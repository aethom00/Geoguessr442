import torch.nn as nn
import torch

class final_model(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(final_model, self).__init__()
        self.fc = nn.Linear(in_features, 2)

    def forward(self, x):
        x1 = self.fc(x)
        return x1
    
    
class CustomGeoMSELoss(nn.Module):
    def __init__(self, penalty=3.0):
        super(CustomGeoMSELoss, self).__init__()
        self.penalty = penalty

    def forward(self, predictions, targets):
        # Unpack predictions and targets into longitude and latitude
        pred_lon, pred_lat = predictions[:, 0], predictions[:, 1]
        true_lon, true_lat = targets[:, 0], targets[:, 1]

        loss_lon = (pred_lon - true_lon) ** 2
        loss_lat = (pred_lat - true_lat) ** 2

        penalty_mask_lon = torch.zeros_like(pred_lon)
        # Apply penalties for specific longitude conditions
        penalty_mask_lon[(pred_lon < -90) | (true_lon < 75)] = 1.0
        
        # Initialize penalty mask for latitude
        penalty_mask_lat = torch.zeros_like(pred_lat)
        penalty_mask_lat[(true_lat > 40)] = 1.0

        penalized_loss_lon = loss_lon * (1 + penalty_mask_lon * (self.penalty - 1))
        penalized_loss_lat = loss_lat * (1 + penalty_mask_lat * (self.penalty - 1))

        return (penalized_loss_lon.mean() + penalized_loss_lat.mean()) / 2

