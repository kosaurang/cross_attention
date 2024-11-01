# models/encoders.py
# import torch
# import torch.nn as nn
# from transformers import AutoTokenizer, BeitImageProcessor, AutoConfig
# from PIL import Image
# import os
from common_imports import *

class Bart_Encode_Feature(nn.Module):
    def __init__(self, config):
        super(Bart_Encode_Feature, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(config.TEXT_EMBEDDING.PRETRAINED_NAME)
        self.padding = config.TOKENIZER.PADDING
        self.max_input_length = config.TOKENIZER.MAX_INPUT_LENGTH
        self.max_target_length = config.TOKENIZER.MAX_TARGET_LENGTH 
        self.truncation = config.TOKENIZER.TRUNCATION
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, input_text, text_pair=None, answers=None):
        encoded_inputs = self.tokenizer(
            input_text, text_pair,
            padding=self.padding,
            max_length=self.max_input_length,
            truncation=self.truncation,
            return_tensors='pt'
        ).to(self.device)

        if answers is not None:
            encoded_targets = self.tokenizer(
                answers,
                padding=self.padding, 
                max_length=self.max_target_length,
                truncation=self.truncation,
                return_tensors='pt'
            ).to(self.device)
            encoded_targets[encoded_targets == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': encoded_inputs.input_ids,
                'attention_mask': encoded_inputs.attention_mask,
                'labels': encoded_targets.input_ids,
                'decoder_attention_mask': encoded_targets.attention_mask
            }
        
        return {
            'input_ids': encoded_inputs.input_ids,
            'attention_mask': encoded_inputs.attention_mask
        }

class Vision_Encode_Pixel(nn.Module):
    def __init__(self, config):
        super(Vision_Encode_Pixel, self).__init__()
        self.preprocessor = BeitImageProcessor.from_pretrained(config.VISION_EMBEDDING.PRETRAINED_NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images):
        processed_images = self.preprocessor(
            images=[Image.open(img_path).convert('RGB') for img_path in images],
            return_tensors="pt"
        ).to(self.device)
        return processed_images.pixel_values