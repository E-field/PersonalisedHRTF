from PIL import Image
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, image_path: list, image_class: list, transform=None):
        self.images_path = image_path
        self.image_class = image_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        label = self.image_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
