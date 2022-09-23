import random
import numpy as np
from collections import deque
from collections import namedtuple
import torch

class experience_replay():
    def __init__(self,size):
        self.buffer = deque([],maxlen=size)
        self.transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        self.buffer.append(self.transition(*args))

    def sample(self,batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def epsilon_greedy(state, network, epsilon, action_space, device):
    sample = np.random.uniform()
    if(sample < epsilon):
        return torch.tensor(random.randrange(action_space), device=device, dtype=torch.long).view(-1,1)
    else:
        with torch.no_grad():
            return network(state).argmax().view(-1,1)


def exploratory_policy(state, network, action_space, device, std):
    with torch.no_grad():
        return network(state) + torch.normal(torch.zeros(action_space), torch.zeros(action_space) + std).to(device)
