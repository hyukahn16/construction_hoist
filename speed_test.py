import environment as gym
import time
import numpy as np
import copy
import os
import random
import matplotlib.pyplot as plt
import timeit
from utility import organize_output, time_wrapper, episode_wrapper
from model import DeepQNetwork
from hyperparam import *

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
agents = [DeepQNetwork(
    nS, nA, lr, gamma, eps, min_eps,
    eps_decay, batch_size, test_mode)
    for _ in range(num_elevators)]
env = gym.make(
    num_elevators, total_floors, total_floors, 
    pass_gen_time, episode_time, nA
)
episode_rewards = [[] for _ in range(num_elevators)] # Stores reward from each episode

print("Speed test starting.\n")

# Measuring the time of action step
print("Measuring 1 action step time")
env.reset()
wrapped = time_wrapper(env.step, [1 for i in range(num_elevators)])
print(timeit.timeit(wrapped, number=1))
print()

# Measuring the time of an episode
print("Measuring EPISODE TIME")
wrapped = episode_wrapper(env, agents, episode_time, nA, nS, batch_size)
print(timeit.timeit(wrapped, number=1))
print("Speed test finished")