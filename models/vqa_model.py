# models/vqa_model.py
from common_imports import *
from utils.device_utils import GLOBAL_DEVICE, select_device
from models.Vision_Encode_Pixel import Vision_Encode_Pixel
from models.Bart_Encode_Feature import Bart_Encode_Feature, Bart_Embedding, Bart_tokenizer

class MBart_BEiT_Model(nn.Module):
    def __init__(self, config):
        super().__init__()  # Sửa lại cú pháp super()
        self.config = config
        
        self.vision_encoder_pixel = Vision_Encode_Pixel(config).to(GLOBAL_DEVICE)
        self.text_encoder = Bart_Encode_Feature(config).to(GLOBAL_DEVICE)
        self.embedding = Bart_Embedding(config).to(GLOBAL_DEVICE)
        
        self.device = GLOBAL_DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
        
        self.generator_args = {
            'max_length': config.GENERATOR.MAX_LENGTH,
            'min_length': config.GENERATOR.MIN_LENGTH,
            'num_beams': config.GENERATOR.NUM_BEAMS,
            'length_penalty': config.GENERATOR.LENGTH_PENALTY,
            'no_repeat_ngram_size': config.GENERATOR.NO_REPEAT_NGRAM_SIZE,
            'early_stopping': config.GENERATOR.EARLY_STOPPING,
        }

    def forward(self, questions, images, labels=None):
        # Chuyển đổi questions và labels sang tensor nếu chưa phải
        if not isinstance(questions, torch.Tensor):
            questions = torch.tensor(questions, device=self.device)
        
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=self.device)
        
        images = images.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        
        encoding_pixel = self.vision_encoder_pixel(images)
        
        # Kiểm tra và xử lý input cho text_encoder
        inputs = self.text_encoder(questions, None, labels)
        inputs.update({'pixel_values': encoding_pixel})
        
        if labels is not None:
            outputs = self.embedding(**inputs)
            return outputs.logits, outputs.loss
        else:
            pred_ids = self.embedding.generate(**inputs, **self.generator_args)
            pred_tokens = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            return pred_tokens
