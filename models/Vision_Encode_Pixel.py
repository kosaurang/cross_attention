from common_imports import *
from utils.device_utils import GLOBAL_DEVICE, select_device

class Vision_Encode_Pixel(nn.Module):
    def __init__(self, config: Dict):
        super(Vision_Encode_Pixel,self).__init__()
        self.preprocessor = BeitImageProcessor.from_pretrained(config.VISION_EMBEDDING.PRETRAINED_NAME)
        self.device = GLOBAL_DEVICE
        self.image_folder = os.path.join("vivqa-dataset", "images")

    def forward(self, images: List[str]):
        processed_images = self.preprocessor(
            images=[
                Image.open(image_path).convert('RGB') for image_path in images
            ],
            return_tensors="pt",
        ).to(self.device)
        return processed_images.pixel_values
