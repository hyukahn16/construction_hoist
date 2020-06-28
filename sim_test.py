# File used for testing the environment and the simulation

import environment as gym
import time
import numpy as np
import copy
import os
import random
from utility import organize_output
from heuristic import ScanAgent

#######################################
# Hyperparameters
num_elevators = 2
total_floors = 20
pass_gen_time = 50

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
agents = [ScanAgent(total_floors) for _ in range(num_elevators)]
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
            legal = env.elevators[e_id].legal_actions()
            new_action = random.choice(legal)
            actions[e_id] = new_action

            cumul_actions[e_id][new_action] += 1
        
        # 2. Take action in the Environment
        new_output = env.step(actions)
        # 3. (Skipped)
        for e_id, e_output in new_output.items():
            cumul_rewards[e_id] += e_output["reward"]
        # 4. overwrite old output with new output
        organize_output(output, new_output)
        env.render()
        time.sleep(0.1)
        print("\n\n")

    # Outside of episode
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
    print("Total passengers generated: {}\n".format(env.generated_passengers))


