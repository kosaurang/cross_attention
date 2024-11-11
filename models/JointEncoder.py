from common_imports import *
import torch.nn.functional as F
from transformers.models.mbart.modeling_mbart import (
    MBartLearnedPositionalEmbedding,
    MBartEncoderLayer, MBartEncoder, MBartDecoder,
    MBartModel, MBartForConditionalGeneration,
    MBartConfig,
    ACT2FN,
    shift_tokens_right, _expand_mask
)

class JointEncoder(MBartEncoder):
    """
    MBartEncoder + visual embedding
    """
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.config = config
        self.visual_embedding = AutoModel.from_pretrained(config.vision_model)
        for param in self.visual_embedding.parameters():
            param.requires_grad = False
        self.project_vision=nn.Linear(config.d_vision,config.d_model)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        vis_attention_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_ids)
        inputs_embeds = inputs_embeds + embed_pos

        B, L = inputs_embeds.size()[:-1]

        img_feat = self.visual_embedding(pixel_values).last_hidden_state
        img_feat = self.project_vision(img_feat)
        V_L = img_feat.size(1)

        if self.config.share_vis_lang_layer_norm:
            inputs_embeds = torch.cat([inputs_embeds, img_feat], dim=1)
            inputs_embeds = self.layernorm_embedding(inputs_embeds)
        else:
            inputs_embeds = self.layernorm_embedding(inputs_embeds)
            inputs_embeds = torch.cat([inputs_embeds, img_feat], dim=1)

        hidden_states = F.dropout(inputs_embeds, p=self.dropout, training=self.training)

        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        vis_attention_mask = torch.ones(B, V_L, dtype=inputs_embeds.dtype, device=inputs_embeds.device)

        attention_mask = torch.cat([attention_mask, vis_attention_mask], dim=1)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if getattr(self.config, "gradient_checkpointing", False):

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                    )
                else:
                    layer_outputs = encoder_layer(hidden_states, attention_mask,
                                                  layer_head_mask= None ,output_attentions=output_attentions)

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )
