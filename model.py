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
    def __init__(self, train_CNN=False, num_classes=1):
        super(CNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        features = self.inception(images)
        return self.sigmoid(self.dropout(self.relu(features))).squeeze(1)