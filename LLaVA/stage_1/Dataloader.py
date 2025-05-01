import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageCaptionDataset(Dataset):
    def __init__(self, images, descriptions, transform, prompt="Mô tả chi tiết bức ảnh:"):
        super(ImageCaptionDataset, self).__init__()
        assert len(images) == len(descriptions)
        self.images = images
        self.descriptions = descriptions
        self.transform = transform
        self.prompt = prompt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        image_tensor = self.transform(image)
        description = self.descriptions[idx]
        return image_tensor, self.prompt, description
