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

def remove_checkpoint(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print("File has been deleted.")

def unfreeze_conv5_x(model):
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the layers in conv5_x
    for param in model.layer4.parameters():  # layer4 is conv5_x in ResNet-50
        param.requires_grad = True

    # Optionally, unfreeze the fully connected layer as well
    for param in model.fc.parameters():
        param.requires_grad = True

#def main(num_training, num_epochs, batch_size, learning_rate, weight_decay):
def main(num_epochs):
    torch.manual_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # Load the pre-trained ResNet50 model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT').to(device)
    # Unfreeze the last block of the ResNet50 model
    
    num_ftrs = model.fc.in_features
    model.fc = final_model(num_ftrs, 2).to(device)
    unfreeze_conv5_x(model)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    dataset = LandmarkDataset(img_dir='CombinedFiles', transform=transform)
    train_set_val = int(0.8 * len(dataset))
    test_set_val = len(dataset) - train_set_val
    train_set, test_set = torch.utils.data.random_split(dataset, [train_set_val, test_set_val])
    dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-4}
        ], weight_decay=1e-4)
    iteration = 0
    plt_loss = [] 
    plt.xlabel('iteration')
    plt.ylabel('Loss - MSE Loss')
    plt.title('Loss Graph')
    iteration = 0

    model.fc.requires_grad = True
    model.layer4.load_state_dict(torch.load('checkpoints/resnet50_cnn_1.pth'))
    model.fc.load_state_dict(torch.load('checkpoints/resnet50_fc_1.pth'))

    for epoch in range(num_epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.to(device)
            loss = loss_fn(outputs, labels)
            if iteration % 100 == 0:
                print(loss)
                loss_new = loss.cpu().detach().numpy()
                plt_loss.append([iteration, loss_new])
            loss.backward()
            optimizer.step()
            iteration += 1
        torch.save(model.layer4.state_dict(), f'checkpoints/resnet50_cnn_{epoch}.pth')
        torch.save(model.fc.state_dict(), f'checkpoints/resnet50_fc_{epoch}.pth')
    # Set the model to evaluation mode
    model.eval()
    new_x = []
    new_y = []
    for loss in (plt_loss):
        loss_0, loss_1 = loss
        plt.scatter(loss_0, loss_1, color='blue')
        new_x.append(loss_0)
        new_y.append(loss_1)
    plt.plot(new_x, new_y, color='red', label='Curve of Best Fit')
    plt.savefig('loss.png')

    #model.load_state_dict(torch.load('checkpoints/resnet50_2.pth'))
    total_loss = 0

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
            gui.place_dot(output_val[1], output_val[0], color='red', r=5)
            
            # green is the correct
            gui.place_dot(true_val[1], true_val[0], color='green', r=10)

            loss = loss_fn(outputs, true_label)
            total_loss += loss
            
    gui.show(display_coords=False, show_boxes=True)

    print(total_loss/len(test_loader))

if __name__ == '__main__':
    main()
