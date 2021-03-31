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



transforms = transforms.Compose([transforms.Resize((480,480)), transforms.ToTensor()])


class imgdataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.csv = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return (len(self.csv))

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, index - 1)).convert('L')
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(self.csv.iloc[index-1, 17])
        return img, y

train = imgdataset("train_image", "train.csv", transform=transforms)


class CNN(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(3,224,224)
        self.cnn1 = nn.Conv2d(3, 16, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # Convolution 2
        self.cnn2 = nn.Conv2d(16, 8, kernel_size=11, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # Fully connected 1 ,#input_shape=(8*50*50)
        self.fc = nn.Linear(8 * 50 * 50, 2)
        # 列出forward的路徑，將init列出的層代入

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out