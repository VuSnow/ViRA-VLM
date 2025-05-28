import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from transformers import AutoTokenizer
from PIL import Image


class ImageCaptionDataset(Dataset):
    def __init__(self, dataset:Dataset, transform: Compose, tokenizer: AutoTokenizer):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.prompt = "Miêu tả chi tiết bức ảnh này từ hành động, vật thể cho đến cảm giác hoặc bối cảnh. Bao gồm mọi yếu tố như màu sắc, ánh sáng, bố cục, các đối tượng và sự tương tác giữa chúng, cảm xúc hoặc thông điệp mà bức ảnh truyền tải. Hãy chú ý đến cả các yếu tố nhỏ mà có thể mang lại chiều sâu cho bức ảnh. Chỉ miêu tả những gì bạn nhận diện được, không được bịa đặt, ảo tưởng."

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # process image
        image = self.dataset[idx]['image'].convert('RGB')
        image_tensor = self.transform(image).float()

        # proces prompt
        question_enc = self.tokenizer(
            self.prompt,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        input_ids = question_enc['input_ids'].squeeze(0).to(dtype=torch.long)
        attention_mask = question_enc['attention_mask'].squeeze(0).to(dtype=torch.long)
        
        # process description
        description = self.dataset[idx]['description']
        labels_enc = self.tokenizer(
            description,
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors='pt'
        )
        labels = labels_enc['input_ids'].squeeze(0).to(dtype=torch.long)

        return{
            'image_tensor': image_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }