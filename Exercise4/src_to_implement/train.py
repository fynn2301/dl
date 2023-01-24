import torch
from sklearn.model_selection import train_test_split
import matplotlib as plt
import torch.optim as optim
import pandas as pd
from data import ChallengeDataset
from model import ResNet
from trainer import Trainer
import torch.nn as nn
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device object
print(device)

# load label data
all_data_df = pd.read_csv("drive/MyDrive/deep_learning/solar_panel_images/data.csv", sep=";")
all_data_df.sample(frac=1)
all_data_len = len(all_data_df)
train_val_data_len = int(all_data_len * 0.8)
test_data_len = int(all_data_len * 0.2)
train_data_len = int(train_val_data_len * 0.8)

# split the data
train_val_data_df, test_data_df  = train_test_split(all_data_df, train_size=train_val_data_len)
train_data_df, val_data_df = train_test_split(train_val_data_df, train_size=train_data_len)


# split the data into train and validation 
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset objects
path = "drive/MyDrive/deep_learning/solar_panel_images/images"
train_dataset = ChallengeDataset(train_data_df, 'train', device, path)
val_dataset = ChallengeDataset(val_data_df, 'val', device, path)
test_dataset = ChallengeDataset(test_data_df, 'test', device, path)

train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=128, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=128, drop_last=True)
# create an instance of our ResNet model

num_epochs = 100

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
fc_in_features = model.fc.in_features
new_output = nn.Sequential(
    nn.Linear(fc_in_features, out_features=2),
    nn.Sigmoid()
)
model.fc = new_output

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)
criterion = nn.BCELoss()

trainer = Trainer(model, criterion, optimizer, train_dl, val_dl)
dict_log = trainer.fit(num_epochs)

# plot the results
plt.plot(np.arange(len(dict_log[0])), dict_log[0], label='train loss')
plt.plot(np.arange(len(dict_log[1])), dict_log[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')

        