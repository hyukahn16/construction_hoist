import tensorflow as tf
import numpy as np
import gym
import math
import os
import copy
from matplotlib.pyplot as plt

from model import PPONetwork
import environment as gym
from utility import organize_output, run_episode, print_hyperparam
from hyperparam import *

# FIXME: add ppo_steps
print_hyperparam()

agents = [PPONetwork(
    nS, nA, batch_size, lr, gamma, lmbda
    ) for _ in range(num_elevators)]
env = gym.make(
    num_elevators, total_floors, total_floors, 
    pass_gen_time, episode_time, nA
)
episode_rewards = [[] for _ in range(num_elevators)] # Stores reward from each episode

# FIXME: until all episodes done or TARGET REWARD REACHED
for e in range(num_episodes):
    print("-----------{} Episode------------".format(e))
    output = {}
    organize_output(output, env.reset())
    neg_action = [-1 for i in range(len(agents))]
    cumul_rewards = [0 for _ in range(len(agents))]
    cumul_actions = {i: [j for j in range(nA)] for i in range(len(agents))}

    # Get a batch of experience
    for iter in range(ppo_steps):
        # Get actions and critic's values for each agent
        actions = copy.deepcopy(neg_action)
        act_dists = np.zeros(num_elevators)
        for e_id, e_output in output.items():
            if e_output["last"]:
                state = e_output["state"]
                state = np.reshape(state, [1, nS])
                act_dists[e_id] = agents[e_id].get_act_dist(state) # FIXME: need input
                actions[e_id] = agents[e_id].get_action(act_dists[e_id])
                cumul_actions[e_id][new_action] += 1

        # Take action in the environment
        if render:
            env.render()
        new_output = env.step(actions)

        # Store experience
        for e_id, e_output in new_output.items():
            # FIXME: does this need to be same state as input state?
            state = copy.deepcopy(output[e_id]["state"])
            action = actions[e_id]
            value = agents[e_id].get_value(state) # FIXME: need input
            mask = 0 # FIXME
            reward = e_output["reward"]
            act_dist = act_dists[e_id]
            agents[e_id].store(state, action, value, mask, reward, act_dist)

        # Overwrite old output with new output
        organize_output(output, new_output)

    # Predict value for the last state
    for e_id, e_output in output.items():        
        last_value = agents[e_id].get_value(e_output["state"])
        returns, advs = agents[e_id].get_advantages(last_value)
    
    # Save models when rewards are good
    # Clear agents' experience memories

        

