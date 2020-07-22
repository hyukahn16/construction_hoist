import environment as gym
import time
import numpy as np
import copy
import os
import random
import matplotlib.pyplot as plt
from model import DeepQNetwork
from utility import organize_output, run_episode, print_hyperparam
from hyperparam import *

print_hyperparam()

agents = [DeepQNetwork(
    nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size, test_mode
    )
    for _ in range(num_elevators)]
env = gym.make(
    num_elevators, total_floors, total_floors, 
    pass_gen_time, episode_time, nA
)
episode_rewards = [[] for _ in range(num_elevators)] # Stores reward from each episode

for e in range(num_episodes):
    print("-----------{} Episode------------".format(e))
    eps_output = run_episode(env, agents, episode_time, nA, nS, batch_size)
    cumul_rewards = eps_output["rewards"]
    cumul_actions = eps_output["actions"]

    # Outside of episode
    for e_id in range(num_elevators):
        print("Elevator_{} Reward: {}".format(e_id, cumul_rewards[e_id]))
        print("Elevator_{} Epsilon: {}".format(e_id, agents[e_id].epsilon))
        print("Elevator_{} Number of passengers served: {}".format(
                e_id,
                env.elevators[e_id].num_served
            )
        )
        print("Elevator_{} Number of passengers carrying: {}".format( 
                e_id,
                len(env.elevators[e_id].passengers)
            )
        )
        print("Elevator_{} Actions: {}\n".format(e_id, cumul_actions[e_id]))
        episode_rewards[e_id].append(cumul_rewards[e_id])
    print("Total passengers generated: {}\n".format(env.generated_passengers))

    if e % 5 == 0:
        plt.figure()
        for e_id in range(num_elevators):
            plt.plot(
                [i for i in range(e + 1)],
                episode_rewards[e_id],
                label=str(e_id)
            )
        plt.legend(loc="upper right")
        plt.pause(0.01)
        plt.draw()
