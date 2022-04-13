# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms


class CNN_CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(16 * 16 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 16 * 16 * 64)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)

if __name__ == '__main__':
    train_dataset = CIFAR10('~/.pytorch/CIFAR10', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = CIFAR10('~/.pytorch/CIFAR10', train=True, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=False)

    model = CNN_CIFAR10().to(device)
    model.train()
    print(model)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    no = 0

    for train_x, train_label in train_loader:
        optimizer.zero_grad()
        train_x = train_x.to(device)
        #print(train_x.size(), train_label)
        y = model(train_x)
        loss = F.nll_loss(y, train_label)
        loss.backward()
        optimizer.step()
        #print(y.size())
        #print(y)
        print(loss.item())
        if no > 2000:
            break
        no += 1

    print(y)

    model.eval()

    with torch.no_grad():
        for test_x, test_target in test_loader:
            test_y = model(test_x)
            predict = test_y.argmax(dim=1, keepdim=True)

            print(predict)
            print(test_target)
            break


