import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets ,models, transforms
from PIL import Image
import cv2
import numpy as np
import csv
import os
import pandas as pd


transforms = transforms.Compose([transforms.Resize((480, 480)), transforms.ToTensor()])


class imgdataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return (len(self.csv))

    def __getitem__(self, index):
        name = '{}.jpg'.format(index)
        img = Image.open(os.path.join(self.root_dir, name)).convert('L')
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(self.csv.iloc[index, 1: 7])
        y = y.float()
        return img, y

batch_size = 64
train_data = imgdataset("./train_image", "train_data.csv", transform=transforms)
val_data = imgdataset("./val_image", "val_data.csv", transform=transforms)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)



class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(230400, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'

# Initialize a model, and put it on the device specified.
model = CNN_Model().to(device)
model.device = device
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

model_path = './model.ckpt'

n_epochs = 80

best_loss = 10000.0

for epoch in range(n_epochs):
    model.train()

    # These are used to record information in training.
    train_loss = 0
    val_loss = 0

    # training
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        batch_loss = criterion(outputs, labels)
        # _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward()
        optimizer.step()

        train_loss += batch_loss.item()
        print(train_loss)

    if len(val_data) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)

                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Loss: {:3.6f} | Val loss: {:3.6f}'.format(
                epoch + 1, n_epochs, train_loss, val_loss))

            # if the model improves, save a checkpoint at this epoch
            if val_loss > best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_loss))
    else:
        print('[{:03d}/{:03d}] Train Loss: {:3.6f}'.format(
            epoch + 1, n_epochs, train_loss))

# if not validating, save the last epoch
if len(val_data) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
    # ---------- Validation ----------
