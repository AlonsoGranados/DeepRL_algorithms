import torch
import torch.nn
import gym
import numpy as np
from utils import experience_replay
from Network import Q_Network
from DQN import gradient_step
import torch.optim as optim
from utils import epsilon_greedy
import matplotlib.pyplot as plt
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.envs.make('LunarLander-v2')
# env = gym.envs.make('MountainCar-v0')

buffer = experience_replay(100000)
network = Q_Network(env.observation_space.shape[0],env.action_space.n, device).to(device)
target_network = Q_Network(env.observation_space.shape[0],env.action_space.n, device).to(device)
target_network.eval()

optimizer = optim.RMSprop(network.parameters())

gamma = 0.9
epsilon = 1
batch_size = 128
max_iter = 200
num_episodes = 50000
target_update = 5000

state = env.reset()
state = torch.from_numpy(state)
G = 0
G_list = []

for e in range(num_episodes):
    start_time = time.time()
    if(e % 100 == 0):
        print(e)
    if (e % 1000 == 999):
        plt.plot(G_list)
        plt.show()
    for t in range(max_iter):
        # env.render()
        if(e % 2000 == 0 and e > 0):
            epsilon = 0.1
            # env.render()
        # action = np.random.randint(env.action_space.n)
        action = epsilon_greedy(state, network, epsilon, env.action_space.n, device)
        # print(action)
        # next_state, reward, done, info = env.step(action)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.from_numpy(next_state)
        reward = torch.tensor([float(reward)], device=device)
        G += reward.item()
        # Store transition
        buffer.push(state.view(1,-1), action, next_state.view(1,-1), reward)

        if(done):
            next_state = None
            break
        gradient_step(buffer,network, target_network, batch_size, device, gamma, optimizer)
        state = next_state

    if e % target_update == 0:
        target_network.load_state_dict(network.state_dict())

    G_list.append(G)
    G = 0


    state = env.reset()
    state = torch.from_numpy(state)

    print(start_time - time.time())

