import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepQlearning(nn.Module):
    def __init__(self, learning_rate):
        super(DeepQlearning, self).__init__()
        self.conv1 = nn.Conv2d(8, 32)
        self.conv2 = nn.Conv2d(32, 4)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss

    def forward(self, state):
        x = F.relu(self.conv1(state))
        return self.conv2(x)
