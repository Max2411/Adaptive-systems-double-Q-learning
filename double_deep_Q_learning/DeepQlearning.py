import torch.nn as nn
import torch.nn.functional as F


class DeepQlearning(nn.Module):
    def __init__(self):
        super(DeepQlearning, self).__init__()
        self.layer1 = nn.Linear(8, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 32)
        self.layer4 = nn.Linear(32, 4)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)
