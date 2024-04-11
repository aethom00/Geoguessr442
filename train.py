import torch
import os
import json
from torchvision import models, transforms
from image_loader import LandmarkDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from Agent import customLoss
from weighted_est_file import weighted_estimate

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

    dataset = LandmarkDataset(img_dir='data', transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # loss_fn = customLoss(prediction_point, actual_point)
    loss_fn = customLoss
    # loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    for epoch in range(5):
        for images, labels in dataloader:
            estimated_lat, estimated_lon = weighted_estimate(outputs) # for the var input the 150 x 1 vector
            outputs = model(images)[0]
            latitute, longitude = labels
            labels = torch.tensor([latitute.item(), longitude.item()])
            print(labels, outputs)
            actual_point = (latitute, longitude) # place in latitude, longitude
            optimizer.zero_grad()
            loss = customLoss((estimated_lat, estimated_lon), actual_point)
            #loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

    # Set the model to evaluation mode
    model.eval()

if __name__ == '__main__':
    main()