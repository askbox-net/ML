# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import nn


class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(2, 5)
        self.layer_2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.sigmoid(self.layer_2(x))
        return x

    def fit(self, x_train, y_train):
        criterion = nn.MSELoss()

        optimizer = optim.SGD(model.parameters(), lr=0.1)

        x = torch.Tensor(x_train)
        y = torch.Tensor(y_train)

        self.train()
        for i in range(5001):
            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print("[%d] loss: %f" % (i, loss))

    def predict(self, x_data):
        self.eval()
        for x in x_data:
            print(x)
            x_in = torch.Tensor(x)
            y_out = model(x_in)
            y = y_out.data
            print("%d %d = %f %f" % (x[0], x[1], y[0], y[1]))


if __name__ == '__main__':
    x = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]])

    # AND
    y_and = np.array([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1]])
    model = Simple()
    model.fit(x, y_and)
    model.predict(x)

    # OR
    y_or = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [0, 1]])
    model = Simple()
    model.fit(x, y_or)
    model.predict(x)

    # XOR
    y_xor = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]])
    model = Simple()
    model.fit(x, y_xor)
    model.predict(x)
