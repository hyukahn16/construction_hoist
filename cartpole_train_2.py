import environment as gym
import time
import numpy as np
import copy
import os
import random
from utility import organize_output
from model import DeepQNetwork
import matplotlib.pyplot as plt

#######################################
# Hyperparameters
num_elevators = 2
total_floors = 40
pass_gen_time = 30

nS = total_floors * 4
nA = 4
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
    pass_gen_time, episode_time
)
episode_rewards = [[] for _ in range(num_elevators)] # Stores reward from each episode

for e in range(num_episodes): # number of episodes == 100
    print("-----------{} Episode------------".format(e))
    output = {}
    organize_output(output, env.reset())
    
    cumul_rewards = [0 for _ in range(num_elevators)]
    cumul_actions = {i: [j for j in range(nA)] for i in range(num_elevators)}
    while env.now() <= episode_time: # Stop episode if time is over
        # 1. Get actions for the decision agents
        actions = copy.deepcopy(neg_action)
        for e_id, e_output in output.items():
            if e_output["last"] == False:
                continue
            new_action = agents[e_id].action(
                np.reshape(e_output["state"], [1, nS])
            )
            actions[e_id] = new_action

            cumul_actions[e_id][new_action] += 1
        
        # 2. Take action in the Environment
        new_output = env.step(actions)
        # 3. (Skipped)
        for e_id, e_output in new_output.items():
            cumul_rewards[e_id] += e_output["reward"]
            replay_action = actions[e_id]

            state = copy.deepcopy(output[e_id]["state"])
            new_state = copy.deepcopy(e_output["state"])
            state = [state]
            new_state = [new_state]

            agents[e_id].store(state, replay_action, e_output["reward"], 
                                new_state, 1.0)
            if len(agents[e_id].memory) > batch_size: 
                agents[e_id].experience_replay(batch_size)

        # 4. overwrite old output with new output
        organize_output(output, new_output)

    # Outside of episode
    for e_id in range(num_elevators):

        print("Elevator_{} Reward: {}\n".format(e_id, cumul_rewards[e_id]))
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

    if e % 1 == 0:
        plt.figure()
        plt.plot([i for i in range(e + 1)], episode_rewards[0], label="0")
        plt.plot([i for i in range(e + 1)], episode_rewards[1], label="1")
        plt.legend(loc="upper right")
        plt.pause(0.01)
        plt.draw()
