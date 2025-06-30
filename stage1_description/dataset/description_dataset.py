from torch.utils.data import Dataset
from torchvision.transforms import Compose
from PIL import Image


class DescriptionDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Compose = None):
        self.dataset = dataset
        self.transform = transform
        self.prompt = "Mô tả chi tiết bức ảnh này từ hành động, vật thể cho đến cảm giác hoặc bối cảnh. Bao gồm mọi yếu tố như màu sắc, ánh sáng, bố cục, các đối tượng và sự tương tác giữa chúng, cảm xúc hoặc thông điệp mà bức ảnh truyền tải. Hãy chú ý đến cả các yếu tố nhỏ mà có thể mang lại chiều sâu cho bức ảnh. Chỉ miêu tả những gì bạn nhận diện được, không được bịa đặt, ảo tưởng."

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        if isinstance(row['image'], Image.Image):
            image_tensor = self.transform(row['image'].convert('RGB'))
        elif isinstance(row['image'], str):
            image_tensor = self.transform(
                Image.open(row['image']).convert('RGB'))

        label = row['description']
        return {
            "image_tensor": image_tensor,
            "label": label,
            "prompt": self.prompt,
        }
