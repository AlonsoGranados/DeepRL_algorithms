import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor_Network(nn.Module):
    def __init__(self, observation_space, action_space, device):
        super(Actor_Network, self).__init__()
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

class Critic_Network(nn.Module):
    def __init__(self, observation_space, action_space, device):
        super(Critic_Network, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(observation_space + action_space,200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1)

    def forward(self,x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x