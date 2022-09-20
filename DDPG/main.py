import torch
import torch.nn
import gym
import numpy as np
from utils import experience_replay
from Network import Actor_Network
from Network import Critic_Network
from DDPG import gradient_step
import torch.optim as optim
from utils import exploratory_policy
import matplotlib.pyplot as plt
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#env = gym.envs.make('LunarLander-v2')
env = gym.envs.make('Pendulum-v1')

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

actor_optimizer = optim.Adam(actor_network.parameters())
critic_optimizer = optim.Adam(critic_network.parameters())

tau = 0.001
gamma = 0.9
batch_size = 64
max_iter = 200
num_episodes = 50000
target_update = 5000

G = 0
G_list = []
for e in range(num_episodes):
    # start_time = time.time()
    state = env.reset()
    state = torch.from_numpy(state).to(device)

    if(e % 100 == 0):
        print(e)
    if (e % 1000 == 999):
        plt.plot(G_list)
        plt.show()
    for t in range(max_iter):
        # env.render()
        # action = np.random.randint(env.action_space.n)
        action = exploratory_policy(state, actor_network, act_space, device)

        next_state, reward, done, info = env.step([action.item()])
        next_state = torch.from_numpy(next_state).to(device)
        reward = torch.tensor([float(reward)], device=device)
        G += reward.item()
        # Store transition
        buffer.push(state.view(1,-1), action, next_state.view(1,-1), reward)
        state = next_state

        gradient_step(buffer,critic_network, critic_target, actor_network, actor_target, batch_size, device, gamma, critic_optimizer, actor_optimizer)

        for target_param, param in zip(actor_target.parameters(), actor_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(critic_target.parameters(), critic_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        if (done):
            next_state = None
            break


    if(e % 20 == 0):
        G_list.append(G)
    G = 0



    # print(time.time()-start_time)

