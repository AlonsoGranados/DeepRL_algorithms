import numpy as np
import gym
from utils import transform_data
from utils import create_tiling
from algorithms import COPDAQ

# Parameter setting
exploratory_sigma = 1

env = gym.make('MountainCarContinuous-v0')
print(env.observation_space)
algo = COPDAQ(1,2)
# table

s = env.reset()

cumulative_return = []
for episode in range(10000):
    G = 0
    if(episode%100 == 0):
        print(episode)
    for iteration in range(5000):
        # env.render()
        a = np.random.uniform(-1,1)
        # a = np.random.randn() * exploratory_sigma + algo.mean(s)[0]
        # print(a)
        next_s, reward, done, info = env.step([a])
        G += reward
        # print(next_s)

        algo.step(s.reshape(-1,1),next_s.reshape(-1,1),reward,a)
        s = next_s


        if done:
            break
    s = env.reset()
    cumulative_return.append(G)
import matplotlib.pyplot as plt

plt.plot(cumulative_return)
plt.show()