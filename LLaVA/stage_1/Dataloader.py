import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class ImageCaptionDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image'].convert('RGB')
        image_tensor = self.transform(image)
        description = item['description']
        return image_tensor, description
