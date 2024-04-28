import torch
import os
import json
from model import final_model, CustomGeoMSELoss
from torchvision import models, transforms
from image_loader import LandmarkDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from gui import GUI
from train import unfreeze_conv5_x


def main(count, is_independent):
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device)
    num_ftrs = model.fc.in_features
    model.fc = final_model(num_ftrs, 2).to(device)
    
    # # latest path
    # last_path = os.listdir('checkpoints')[0]
    # last_path = 'checkpoints/' + last_path
    # checkpoint_cnn = torch.load('checkpoints/resnet50_cnn_0.pth')
    # checkpoint_fc = torch.load('checkpoints/resnet50_fc_0.pth')
    unfreeze_conv5_x(model)
    model.fc.requires_grad = True
    model.layer4.load_state_dict(torch.load('checkpoints/resnet50_cnn_6.pth'))
    model.fc.load_state_dict(torch.load('checkpoints/resnet50_fc_6.pth'))


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = LandmarkDataset(img_dir='CombinedFiles', transform=transform)
    train_set_val = int(0.8 * len(dataset))
    test_set_val = len(dataset) - train_set_val
    _, test_set = torch.utils.data.random_split(dataset, [train_set_val, test_set_val])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    loss_fn = nn.MSELoss()
    total_loss = 0
    model.eval()
    if not is_independent:
        gui = GUI(num_rects_width=1, num_rects_height=1)
        gui.init()
        gui.clear_output()

        with torch.no_grad():
            for i, (images, true_label) in enumerate(test_loader):
                outputs = model(images)
                outputs = outputs.to(device)

                output_val = outputs[0].tolist()
                true_val = true_label[0].tolist()
        
                print(f"true_label: {true_val}, predicted_label: {output_val}")

                # red is our guess
                gui.place_dot(output_val[1], output_val[0], color='red', r=10)
                
                # green is the correct
                gui.place_dot(true_val[1], true_val[0], color='green', r=10)

                loss = loss_fn(outputs, true_label)
                total_loss += loss
                gui.show(display_coords=False, show_boxes=True, index=i)

                if i == (count-1):
                    break

    else:
        with torch.no_grad():
            for i, (images, true_label) in enumerate(test_loader):
                gui = GUI(num_rects_width=1, num_rects_height=1)
                gui.init()
                gui.clear_output()
                
                outputs = model(images)
                outputs = outputs.to(device)

                output_val = outputs[0].tolist()
                true_val = true_label[0].tolist()
        
                print(f"true_label: {true_val}, predicted_label: {output_val}")

                # red is our guess
                gui.place_dot(output_val[1], output_val[0], color='red', r=10)
                
                # # green is the correct
                gui.place_dot(true_val[1], true_val[0], color='green', r=10)

                # plt.plot([output_val[0], true_val[0]], [output_val[1], true_val[1]], 'ro-')
                # plt.savefig('b.png')

                loss = loss_fn(outputs, true_label)
                total_loss += loss

                gui.show(display_coords=False, show_boxes=True, index=i)

                if i == (count-1):
                    break

    print(total_loss/len(test_loader))
if __name__ == '__main__':
    main()
