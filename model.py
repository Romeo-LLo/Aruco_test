import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, models
import torchvision.transforms as transforms
from PIL import Image

import os
import pandas as pd
import gc
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure




class imgdataset(Dataset):
    def __init__(self, root_dir, csv_file, model, transform=None):
        self.root_dir = root_dir
        self.csv = pd.read_csv(csv_file)
        self.transform = transform
        self.model = model

    def __len__(self):

        return (len(self.csv))

    def __getitem__(self, index):
        name = '{}.jpg'.format(index)
        # img = Image.open(os.path.join(self.root_dir, name)).convert('L')
        img = Image.open(os.path.join(self.root_dir, name))
        if self.transform:
            img = self.transform(img)

        if self.model == 0:
            y = torch.tensor(self.csv.iloc[index, 1: 7])
        elif self.model == 1:
            y = torch.tensor(self.csv.iloc[index, 1: 4])
        elif self.model == 2:
            y = torch.tensor(self.csv.iloc[index, 4: 7])


        y = y.float()

        return img, y

class CNN_Model1(nn.Module):
    def __init__(self, model):
        super(CNN_Model1, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0),
            nn.Dropout(0.1),

        )

        if model == 0:
            self.fc_layers = nn.Sequential(
                nn.Linear(256 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 6),
                nn.Dropout(0.1),

             )
        elif model == 1 or model == 2:
            self.fc_layers = nn.Sequential(
                nn.Linear(256 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3),

            )

    def forward(self, x):

        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x

class CNN_Model2(nn.Module):
    def __init__(self, model):
        super(CNN_Model2, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
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

        if model == 0:
            self.fc_layers = nn.Sequential(
                nn.Linear(256 * 8 * 8, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 6),

            )
        elif model == 1 or model == 2:
            self.fc_layers = nn.Sequential(
                nn.Linear(256 * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 3),
            )

    def forward(self, x):

        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x


def train(train_loader, val_loader, val_len, config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config['model'] == 1:
        model = CNN_Model1(model=config['model']).to(device)
    elif config['model'] == 2:
        model = CNN_Model2(model=config['model']).to(device)

    model.device = device
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    model_path = config["model_path"]

    n_epochs = config['n_epochs']
    best_loss = 10000.0
    loss_record = {'train': [], 'dev': []}


    for epoch in range(n_epochs):
        model.train()

        train_loss = 0
        val_loss = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)
            # print(batch_loss)
            # _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
            batch_loss.backward()
            optimizer.step()
            loss_record['train'].append(batch_loss.detach().cpu().item())

            train_loss += batch_loss.item()

        if val_len > 0:
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                    val_loss += batch_loss.item()
                loss_record['dev'].append(val_loss/len(val_loader))

                print('[{:03d}/{:03d}] Train Loss: {:3.6f} | Val loss: {:3.6f}'.format(
                    epoch + 1, n_epochs, train_loss/len(train_loader), val_loss/len(val_loader)))

                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(model.state_dict(), model_path)
                    print('saving model with loss = {:.3f}'.format(best_loss/len(val_loader)))
        else:
            print('[{:03d}/{:03d}] Train Loss: {:3.6f}'.format(
                epoch + 1, n_epochs, train_loss/len(train_loader)))

    if val_len == 0:
        torch.save(model.state_dict(), model_path)
        print('saving model at last epoch')
    return loss_record

def test(test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = './model.ckpt'
    criterion = nn.MSELoss()
    test_loss = 0


    model = CNN_Model2().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels)

            test_loss += batch_loss.item()

        print('Test Loss: {:3.6f}'.format(test_loss/len(test_loader)))

def plot_learning_curve(loss_record, title=''):
    ''' Plot learning curve of your DNN (train & dev loss) '''
    total_steps = len(loss_record['train'])
    x_1 = range(total_steps)
    x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
    figure(figsize=(6, 4))
    plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
    plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
    plt.ylim(0.0, 5.)
    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title('Learning curve of {}'.format(title))
    plt.legend()
    plt.show()

def Config():
    config = {
        'batch_size': 32,
        'lr': 0.0001,
        'weight_decay': 0,
        'n_epochs': 120,
        'early_stop': 2,
        'model': 1,
        #model 0 = rvec+tvec   model 1 = rvec   model 2 = tvec
        'model_path': './model_r.ckpt'
    }
    #
    # config = {
    #     'batch_size': 32,
    #     'lr': 0.0003,
    #     'weight_decay': 5e-4,
    #     'n_epochs': 120,
    #     'early_stop': 2,
    #     'model': 2,
    #     #model 0 = rvec+tvec   model 1 = rvec   model 2 = tvec
    #     'model_path': './model_t.ckpt'
    #
    # }

    return config

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    seed = 73
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    config = Config()

    train_transforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    test_tansforms = transforms.Compose([transforms.Resize((128, 128)),
                                          transforms.ToTensor()])

    dataset = imgdataset("./train_image", "train_data.csv", model=config['model'], transform=train_transforms)
    val_len = int(0.2 * len(dataset))
    train_len = len(dataset) - val_len
    dataset_train, dataset_valid = random_split(dataset, (train_len, val_len))


    train_loader = DataLoader(dataset_train, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(dataset_valid, batch_size=config['batch_size'], shuffle=False)

    del dataset_train, dataset_valid
    gc.collect()

    model_loss_record = train(train_loader, val_loader, val_len, config)
    plot_learning_curve(model_loss_record, title='deep model')

# tt_dataset = imgdataset("./test_image", "test_data.csv", transform=test_tansforms)
# test_loader = DataLoader(tt_dataset, batch_size=batch_size, shuffle=False)
# test(test_loader)

