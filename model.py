import torch.nn as nn
import torch

class final_model(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(final_model, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=32)
        self.fc5 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = nn.functional.relu(x1)
        x1 = self.fc2(x1)
        x1 = nn.functional.relu(x1)
        x1 = self.fc3(x1)
        x1 = nn.functional.relu(x1)
        x1 = self.fc4(x1)
        x1 = nn.functional.relu(x1)
        x1 = self.fc5(x1)
        return x1
    
    
class CustomGeoMSELoss(nn.Module):
    def __init__(self, penalty=3.0):
        super(CustomGeoMSELoss, self).__init__()
        self.penalty = penalty

    def forward(self, predictions, targets):
        # Unpack predictions and targets into longitude and latitude
        pred_lon, pred_lat = predictions[:, 0], predictions[:, 1]
        true_lon, true_lat = targets[:, 0], targets[:, 1]

        # Calculate base MSE loss for both longitude and latitude
        loss_lon = (pred_lon - true_lon) ** 2
        loss_lat = (pred_lat - true_lat) ** 2

        # Initialize penalty mask for longitude
        penalty_mask_lon = torch.zeros_like(pred_lon)
        # Apply penalties for specific longitude conditions
        penalty_mask_lon[(pred_lon < -90) | (true_lon < 75)] = 1.0
        
        # Initialize penalty mask for latitude
        penalty_mask_lat = torch.zeros_like(pred_lat)
        # Apply penalties for specific latitude conditions
        penalty_mask_lat[(true_lat > 40)] = 1.0

        # Calculate the penalized loss
        penalized_loss_lon = loss_lon * (1 + penalty_mask_lon * (self.penalty - 1))
        penalized_loss_lat = loss_lat * (1 + penalty_mask_lat * (self.penalty - 1))

        # Return the mean of the combined penalized losses
        return (penalized_loss_lon.mean() + penalized_loss_lat.mean()) / 2

