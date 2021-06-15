import environment as gym
import time
import numpy as np
import copy
import os
import random
import simpy
from utility import organize_output, split_state
from model import DeepQNetwork
import matplotlib.pyplot as plt

#######################################
# Hyperparameters
num_elevators = 2
total_floors = 20
pass_gen_time = 10

nS = total_floors * 4
nA = 3

lr = 0.001
gamma = 0.95
eps = 1
min_eps = 0.1
eps_decay = 0.9999
batch_size = 24
episode_time = 1000
num_episodes = 100

use_saved = False
# END Hyperparameters
#######################################

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size)
    for _ in range(num_elevators)]
env = gym.Environment(
    simpy.Environment(),
    num_elevators,
    total_floors,
    total_floors, 
    pass_gen_time,
    nA,
    episode_time,
)
episode_rewards = [[] for _ in range(num_elevators)] # Stores reward from each episode

for e in range(num_episodes):
    print("-----------{} Episode------------".format(e))
    # env.reset() and env.step() return tuple:
    # ( state, [reward list], done, {info}, [decision_elevator list] )
    output = env.reset() # Reset returns only the state
    output = [output, [], False, {}, [i for i in range(num_elevators)]]
    cumul_rewards = [0 for _ in range(num_elevators)]
    cumul_actions = {i: [0,0,0] for i in range(num_elevators)}
    old_states = split_state(output, num_elevators, total_floors)

    while env.now() <= episode_time: # Stop episode if time is over
        # 1. Get actions for the decision agents
        actions = copy.deepcopy(neg_action)
        old_dec_elevs = output[4]
        for e_id in old_dec_elevs:
            actions[e_id] = agents[e_id].action(
                np.reshape(old_states[e_id], [1, nS])
            )
            cumul_actions[e_id][actions[e_id]] += 1 # stats keeping
        
        # 2. Take action in the Environment
        new_output = env.step(actions)
        new_states = split_state(new_output, num_elevators, total_floors)
        new_dec_elevs = new_output[4]
        # 3. Experience replay
        for e_id in new_dec_elevs:
            reward = new_output[1][e_id]
            cumul_rewards[e_id] += reward # stats keeping
            old_action = actions[e_id]
            old_state = [copy.deepcopy(old_states[e_id])]
            new_state = [copy.deepcopy(new_states[e_id])]
            done = new_output[2]

            agents[e_id].store(
                old_state,
                old_action,
                reward,
                new_state,
                done
            )
            if len(agents[e_id].memory) > batch_size: 
                agents[e_id].experience_replay(batch_size)

        # 4. overwrite old state with new state
        # decay epsilon
        old_states = new_states
        eps *= eps_decay


    # After a single episode finishes - print relevant stats
    print("Rewards: {}\n".format(cumul_rewards))
    for e_id in range(num_elevators):
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
    print("Epsilon value: {}\n".format(eps))
    print("Total passengers generated: {}\n".format(env.generated_passengers))

    '''
    if e % 1 == 0:
        plt.figure()
        for i in range(num_elevators):
            print(len(episode_rewards[i]))
        print(len([i for i in range(e + 1)]))
        plt.plot([i for i in range(e + 1)], episode_rewards[0], label="0")
        plt.plot([i for i in range(e + 1)], episode_rewards[1], label="1")
        plt.legend(loc="upper right")
        plt.pause(0.01)
        plt.draw()
    '''
