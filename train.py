import torch
import os
import json
from torchvision import models, transforms
from image_loader import LandmarkDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

def print_labels(indexes, class_idx):
    for i in indexes:
        print(class_idx[str(i.item())])

def main():
    # Load the pre-trained ResNet50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #dataset = LandmarkDataset(img_dir='Data', transform=transform)
    #dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataset = LandmarkDataset(img_dir='Data', transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [500, 18799])
    dataloader = DataLoader(train_set, batch_size=50, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(10):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()

    # Set the model to evaluation mode
    model.eval()

if __name__ == '__main__':
    main()
