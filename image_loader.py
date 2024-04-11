import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from finding_box import find_grid_index

class LandmarkDataset(Dataset):
    """ A custom dataset class that loads images from a specified directory with labels. """

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_names = [f"data/{img}" for img in os.listdir(img_dir)]
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

        # find_grid_index(target_lat, target_lon, min_lat, max_lat, min_lon, max_lon, num_rows, num_cols)
        
        label = find_grid_index(float(label[0]), float(label[1]))
        #box = (10*(float(label[0]) - 1)) + float(label[1])



        if self.transform:
            image = self.transform(image)

        return image, label