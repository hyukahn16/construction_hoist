import environment as gym
import time
import numpy as np
import copy
import os
import random
from utility import organize_output, time_wrapper, episode_wrapper
from model import DeepQNetwork
import matplotlib.pyplot as plt
import timeit

#######################################
# Hyperparameters
num_elevators = 1
total_floors = 20
pass_gen_time = 50

nS = total_floors * 4
nA = 3
lr = 0.001
gamma = 0.95
eps = 1
min_eps = 0.1
eps_decay = 0.99999
batch_size = 24
episode_time = 10000
num_episodes = 10000

use_saved = False
# END Hyperparameters
#######################################

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size)
    for _ in range(num_elevators)]
env = gym.make(
    num_elevators, total_floors, total_floors, 
    pass_gen_time, episode_time, nA
)
episode_rewards = [[] for _ in range(num_elevators)] # Stores reward from each episode

print("Speed test starting.")
'''
# Measuring the time of action step
env.reset()
wrapped = time_wrapper(env.step, [1 for i in range(num_elevators)])
print(timeit.timeit(wrapped, number=1))
print("Speed test finished.")
'''


# Measuring the time of an episode
wrapped = episode_wrapper(env, agents, episode_time, nA, nS, batch_size)
print(timeit.timeit(wrapped, number=1))

print("Speed test finished")