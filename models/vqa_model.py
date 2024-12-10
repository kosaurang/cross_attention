# models/vqa_model.py
from common_imports import *
from utils.device_utils import GLOBAL_DEVICE, select_device
from models.Vision_Encode_Pixel import Vision_Encode_Pixel
from models.Bart_Encode_Feature import Bart_Encode_Feature, Bart_Embedding, Bart_tokenizer

class MBart_BEiT_Model(nn.Module):
    def __init__(self, config):
        super(MBart_BEiT_Model, self).__init__()
        self.config = config
        self.vision_encoder_pixel = Vision_Encode_Pixel(config)
        self.text_encoder = Bart_Encode_Feature(config) 
        self.device = GLOBAL_DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
        self.embedding = Bart_Embedding(config)

        self.generator_args = {
            'max_length': config.GENERATOR.MAX_LENGTH,
            'min_length': config.GENERATOR.MIN_LENGTH,
            'num_beams': config.GENERATOR.NUM_BEAMS,
            'length_penalty': config.GENERATOR.LENGTH_PENALTY,
            'no_repeat_ngram_size': config.GENERATOR.NO_REPEAT_NGRAM_SIZE,
            'early_stopping': config.GENERATOR.EARLY_STOPPING,
        }

    def forward(self, questions, images, labels=None):
        # Di chuyển dữ liệu sang device
        images = images.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        encoding_pixel = self.vision_encoder_pixel(images)
        inputs = self.text_encoder(questions, None, labels)
        inputs.update({'pixel_values': encoding_pixel})

        if labels is not None:
            outputs = self.embedding(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids = self.embedding.generate(**inputs, **self.generator_args)
            pred_tokens = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens
