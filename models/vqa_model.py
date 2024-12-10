# models/vqa_model.py
from common_imports import *
from utils.device_utils import GLOBAL_DEVICE, select_device
from models.Vision_Encode_Pixel import Vision_Encode_Pixel
from models.Bart_Encode_Feature import Bart_Encode_Feature, Bart_Embedding, Bart_tokenizer

class MBart_BEiT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Truy cập config generator an toàn hơn
        generator_config = config.MODEL.GENERATOR
        
        # Chuyển model sang device ngay khi khởi tạo
        self.vision_encoder_pixel = Vision_Encode_Pixel(config).to(GLOBAL_DEVICE)
        self.text_encoder = Bart_Encode_Feature(config).to(GLOBAL_DEVICE)
        self.embedding = Bart_Embedding(config).to(GLOBAL_DEVICE)
        
        self.device = GLOBAL_DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL.TEXT_EMBEDDING.PRETRAINED_NAME)
        
        # Chuyển generator_args thành dict từ CfgNode
        self.generator_args = {
            'max_length': generator_config.MAX_LENGTH,
            'min_length': generator_config.MIN_LENGTH,
            'num_beams': generator_config.NUM_BEAMS,
            'length_penalty': generator_config.LENGTH_PENALTY,
            'no_repeat_ngram_size': generator_config.NO_REPEAT_NGRAM_SIZE,
            'early_stopping': generator_config.EARLY_STOPPING,
        }

    def forward(self, questions, images, labels=None):
        # Đảm bảo dữ liệu trên đúng device
        images = images.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        encoding_pixel = self.vision_encoder_pixel(images)
        
        # Kiểm tra và xử lý input
        inputs = self.text_encoder(questions, None, labels)
        inputs.update({'pixel_values': encoding_pixel})
        
        if labels is not None:
            outputs = self.embedding(**inputs)
            return outputs.logits, outputs.loss
        else:
            # Sử dụng dict thông thường
            pred_ids = self.embedding.generate(**inputs, **self.generator_args)
            pred_tokens = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens
