# models/vqa_model.py
import types
from common_imports import *
from utils.device_utils import GLOBAL_DEVICE, select_device
from models.Vision_Encode_Pixel import Vision_Encode_Pixel
from models.Bart_Encode_Feature import Bart_Encode_Feature, Bart_Embedding, Bart_tokenizer

class MBart_BEiT_Model(nn.Module):
    def __init__(self, config):
        super(MBart_BEiT_Model, self).__init__()
        self.config = config
        
        # Chuyển model sang device ngay khi khởi tạo
        self.vision_encoder_pixel = Vision_Encode_Pixel(config).to(GLOBAL_DEVICE)
        self.text_encoder = Bart_Encode_Feature(config).to(GLOBAL_DEVICE)
        self.embedding = Bart_Embedding(config).to(GLOBAL_DEVICE)
        
        self.device = GLOBAL_DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
        
        # Chuyển generator_args thành immutable
        self.generator_args = types.MappingProxyType({
            'max_length': config.GENERATOR.MAX_LENGTH,
            'min_length': config.GENERATOR.MIN_LENGTH,
            'num_beams': config.GENERATOR.NUM_BEAMS,
            'length_penalty': config.GENERATOR.LENGTH_PENALTY,
            'no_repeat_ngram_size': config.GENERATOR.NO_REPEAT_NGRAM_SIZE,
            'early_stopping': config.GENERATOR.EARLY_STOPPING,
        })

    def forward(self, questions, images, labels=None):
        # Đảm bảo dữ liệu trên đúng device
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
            # Chuyển generator_args thành dict thông thường
            pred_ids = self.embedding.generate(**inputs, **dict(self.generator_args))
            pred_tokens = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens
