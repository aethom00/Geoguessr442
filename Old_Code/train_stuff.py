import torch
import os
import json
import sys
from torchvision import models, transforms
from image_loader import LandmarkDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import math
import matplotlib.pyplot as plt
# from Agent import haversine_distance
from Old_Code.weighted_est_file import weighted_estimate

def print_labels(indexes, class_idx):
    for i in indexes:
        print(class_idx[str(i.item())])

def customLoss(prediction_lat, prediction_lon, actual_lat, actual_lon):
    return haversine_distance(prediction_lat, prediction_lon, actual_lat, actual_lon)

def haversine_distance(predicted_lat, predicted_long, actual_lat, actual_long, in_meters=False): # predicted and true should be tuples
    # distance haversine 
    R = 6371 # Radius of the Earth in km
    rad = 2 * R 
    # convert lat and longitudes to radians 
    predicted_lat, predicted_long  = torch.deg2rad(predicted_lat), torch.deg2rad(predicted_long)
    actual_lat, actual_long  = torch.deg2rad(actual_lat), torch.deg2rad(actual_long)
    dlat = 1 - torch.cos(actual_lat - predicted_lat)
    dlon = torch.cos(predicted_lat) * torch.cos(actual_lat) * (1 - torch.cos(actual_long - predicted_long))
    distance = rad * torch.arcsin(torch.sqrt((dlat + dlon)/2))
    return distance * 1000 if in_meters else distance # Convert to meters if required

def train_model(batch_size: int = 50, lr: float = 1e-2) -> None:
    # Load the pre-trained ResNet50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = LandmarkDataset(img_dir='Data', transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [500, 18799])
    dataloader = DataLoader(train_set, batch_size=1, shuffle=False)
    # loss_fn = customLoss
    softmax = nn.Softmax(dim=0) 
    loss = torch.nn.CrossEntropyLoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    iteration = 0
    plt_loss = [] 
    plt.xlabel('iteration')
    plt.ylabel('Loss - Cross Entropy')

    for epoch in range(5):
        for images, labels in dataloader:
            optimizer.zero_grad()
            
            #outputs = softmax(model(images)[0])
            #predicted_point = weighted_estimate(outputs) # for the var input the 150 x 1 vector
            #estimated_lat, estimated_lon = predicted_point
            outputs = model((images)[0])
            model_lat, model_long = outputs
            

            #print(estimated_lat.dtype)
            #print(estimated_lon.dtype)
            #print(labels[0].dtype)
            #print(labels[1].dtype)
            #print("---")
            
            #loss = customLoss(estimated_lat, estimated_lon, labels[0], labels[1])

            
            #loss = torch.tensor([loss])

            #print(f"loss is: {loss}")
            loss_new = loss.detach().numpy()
            plt_loss.append([iteration, loss_new])
            loss.backward()
            optimizer.step()
            iteration += 1
            print(iteration)
            #plt.scatter(iteration, loss_new)
            #plt.show()
        print("Gone through one epoch")
        torch.save(model.state_dict(), f'checkpoints/resnet50_{iteration}.pth')
            

    # Set the model to evaluation mode
        new_x = []
        new_y = []
        for loss in (plt_loss):
            loss_0, loss_1 = loss
            plt.scatter(loss_0, loss_1, color='blue')
            new_x.append(loss_0)
            new_y.append(loss_1)
        plt.plot(new_x, new_y, color='red', label='Curve of Best Fit')


    
    plt.show()
    model.eval()

if __name__ == '__main__':
    train_model()