import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from finding_box import find_grid_index

class LandmarkDataset(Dataset):
    """ A custom dataset class that loads images from a specified directory with labels. """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f"Data/{img}" for img in os.listdir(img_dir)]
        self.labels = []  # Assumes labels are part of the filename
        for img in os.listdir(img_dir):
            img = img.removesuffix(".jpg")
            self.labels.append(img)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = self.img_names[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        label = label.split("_")
        label = (float(label[0]), float(label[1]))
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        image = image.to(device)
        label = label.to(device)
        return image, label