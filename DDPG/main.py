import torch
import torch.nn
import gym
import numpy as np
from utils import experience_replay
from Network import Actor_Network
from Network import Critic_Network
from DDPG import critic_step
from DDPG import actor_step
import torch.optim as optim
from utils import exploratory_policy
import matplotlib.pyplot as plt
import time

tau = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#env = gym.envs.make('LunarLander-v2')
env = gym.envs.make('MountainCarContinuous-v0')

# Parameters
obs_space = env.observation_space.shape[0]
act_space = env.action_space.shape[0]

buffer = experience_replay(10000)
actor_network = Actor_Network(obs_space, act_space, device).to(device)
actor_target = Actor_Network(obs_space, act_space, device).to(device)

critic_network = Critic_Network(obs_space, act_space, device).to(device)
critic_target = Critic_Network(obs_space, act_space, device).to(device)

for target_param, param in zip(actor_target.parameters(), actor_network.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(critic_target.parameters(), critic_network.parameters()):
    target_param.data.copy_(param.data)


actor_target.eval()
critic_target.eval()

actor_optimizer = optim.RMSprop(actor_network.parameters())
critic_optimizer = optim.RMSprop(critic_network.parameters())


gamma = 0.9
epsilon = 1
batch_size = 128
max_iter = 200
num_episodes = 50000
target_update = 5000

state = env.reset()
state = torch.from_numpy(state).to(device)
G = 0
G_list = []
for e in range(num_episodes):
    if(e % 100 == 0):
        print(e)
    if (e % 1000 == 999):
        plt.plot(G_list)
        plt.show()
    for t in range(max_iter):

        if(e % 1000 == 999):
            epsilon = 0.1
            # env.render()
        # action = np.random.randint(env.action_space.n)
        action = exploratory_policy(state, actor_network, act_space, device)

        # next_state, reward, done, info = env.step(action)
        next_state, reward, done, info = env.step([action.item()])
        next_state = torch.from_numpy(next_state).to(device)
        reward = torch.tensor([float(reward)], device=device)
        G += reward.item()
        # Store transition
        buffer.push(state.view(1,-1), action, next_state.view(1,-1), reward)

        if(done):
            next_state = None
            break
        critic_step(buffer,critic_network, critic_target, actor_target, batch_size, device, gamma, critic_optimizer)
        actor_step(buffer, critic_network, actor_network, batch_size, device, gamma, actor_optimizer)
        state = next_state

        for target_param, param in zip(actor_target.parameters(), actor_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(critic_target.parameters(), critic_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    if(e % 20 == 0):
        G_list.append(G)
    G = 0

    state = env.reset()
    state = torch.from_numpy(state).to(device)

