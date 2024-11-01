from common_imports import *

class Vision_Encode_Pixel(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Encode_Pixel,self).__init__()
        self.preprocessor = BeitImageProcessor.from_pretrained(config.VISION_EMBEDDING.PRETRAINED_NAME)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_folder = os.path.join("vivqa-dataset", "images")

    def forward(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[
                Image.open(image_path).convert('RGB') for image_path in images
            ],
            return_tensors="pt",
        ).to(self.device)
        return processed_images.pixel_values