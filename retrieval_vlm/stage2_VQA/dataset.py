import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class VisualQuestionAnsweringDataset(Dataset):
    def __init__(self, images, descriptions, transform, questions, answers):
        super(VisualQuestionAnsweringDataset, self).__init__()
        assert len(images) == len(descriptions)
        self.images = images
        self.descriptions = descriptions
        self.transform = transform
        self.question = questions
        self.answer = answers

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].convert("RGB")
        image_tensor = self.transform(image)
        description = self.descriptions[idx]
        question = self.question[idx]
        answer = self.answer[idx]
        return image_tensor, description, question, answer
