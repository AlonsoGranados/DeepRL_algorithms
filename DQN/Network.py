import torch
import torch.nn as nn
import torch.nn.functional as F

class Q_Network(nn.Module):
    def __init__(self, observation_space, action_space, device):
        super(Q_Network, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(observation_space,200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, action_space)

    def forward(self,x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x