import torch
import os
import json
from torchvision import models, transforms
from image_loader import CustomImageDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split


def print_labels(indexes, class_idx):
    for i in indexes:
        print(class_idx[str(i.item())])


def main():

    class_idx = json.load(open("resnet50_labels.json"))

    # Load the pre-trained ResNet50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    #torch.save(model, 'resnet50.pth')
    transform = transforms.Compose([
        # you can add other transformations in this list
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomImageDataset(img_dir='data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    for images, labels in dataloader:
        outputs = model(images)
        print(labels)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.max(probabilities, dim=1)
        print_labels(top_catid, class_idx)

    train_proportion = 0.8
    val_proportion = 0.1
    test_proportion = 0.1

    train_size = int(train_proportion * len(dataset))
    val_size = int(val_proportion * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    for loader in [train_loader, val_loader, test_loader]:
        for images, labels in loader:
            outputs = model(images)
            print(labels)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_catid = torch.max(probabilities, dim=1)
            print_labels(top_catid, class_idx)



    # Set the model to evaluation mode
    model.eval()

if __name__ == '__main__':
    main()