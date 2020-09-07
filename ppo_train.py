import tensorflow as tf
import numpy as np
import gym
import math
import os
import copy
import matplotlib.pyplot as plt

from model import PPONetwork
import environment as gym
from utility import organize_output, run_episode, print_hyperparam
from hyperparam import *

print_hyperparam()

agents = [PPONetwork(
    nS, nA, batch_size, lr, gamma, lmbda, ppo_epochs, clip_range
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
        act_dists = [[0, 0, 0] for _ in range(num_elevators)])
        for e_id, e_output in output.items():
            if e_output["last"]:
                state = e_output["state"]
                state = np.reshape(state, [1, nS])
                act_dists[e_id] = agents[e_id].get_act_dist(state)[0]
                actions[e_id] = agents[e_id].get_action(act_dists[e_id])
                new_action = actions[e_id]
                cumul_actions[e_id][new_action] += 1

        # Take action in the environment
        if render:
            env.render()
        new_output = env.step(actions)

        # Store experience
        for e_id, e_output in new_output.items():
            state = np.reshape(output["state"], [1, nS])
            action = actions[e_id]
            value = agents[e_id].get_value(state)
            mask = 0 # FIXME
            reward = e_output["reward"]
            act_dist = act_dists[e_id]
            agents[e_id].store(state, action, value, mask, reward, act_dist)

        # Overwrite old output with new output
        organize_output(output, new_output)

    # Train agents
    for e_id, e_output in output.items():  
        state = np.reshape(output["state"], [1, nS])      
        last_value = agents[e_id].get_value(state)
        returns, advs = agents[e_id].get_advantages(last_value)
        agents[e_id].learn(returns, advs)
    
    # Save models when rewards are good
    # TODO

    # Clear agents' experience memories for the next episode
    for agent in agents:
        agent.clear_memory()

    # Print statistics
    for e_id in range(num_elevators):
        print("Elevator_{} Reward: {}".format(e_id, cumul_rewards[e_id]))
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

    # Print graph
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
    

        

