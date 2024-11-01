from common_imports import *
from models.VisionBartForConditionalGeneration import VisionBartForConditionalGeneration

def Bart_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
    return tokenizer

def Bart_Embedding(config):
    model_config = AutoConfig.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
    model_config.update({
        'share_vis_lang_layer_norm': True,
        'vision_model' : config.VISION_EMBEDDING.PRETRAINED_NAME,
        'd_vision': config.VISION_EMBEDDING.D_PRETRAINED_FEATURE
        })
    embedding = VisionBartForConditionalGeneration.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME, config=model_config)
    return embedding

class Bart_Encode_Feature(nn.Module):
    def __init__(self, config):
        super(Bart_Encode_Feature, self).__init__()
        self.tokenizer=Bart_tokenizer(config)
        self.padding = config.TOKENIZER.PADDING
        self.max_input_length = config.TOKENIZER.MAX_INPUT_LENGTH
        self.max_target_length = config.TOKENIZER.MAX_TARGET_LENGTH
        self.truncation = config.TOKENIZER.TRUNCATION
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_text: List[str], text_pair: List[str]=None, answers: List[str]=None):
        encoded_inputs = self.tokenizer(
                                input_text,text_pair,
                                padding= self.padding,
                                max_length=self.max_input_length,
                                truncation=self.truncation,
                                return_tensors='pt',
                            ).to(self.device)
        if answers is not None:
            encoded_targets = self.tokenizer(
                                    answers,
                                    padding= self.padding,
                                    max_length=self.max_target_length,
                                    truncation=self.truncation,
                                    return_tensors='pt',
                                ).to(self.device)
            encoded_targets[encoded_targets == self.tokenizer.pad_token_id] = -100
            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask,
                'labels': encoded_targets.input_ids,
                'decoder_attention_mask': encoded_targets.attention_mask,
            }
        else:
            encodings = {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask
            }
        return encodings