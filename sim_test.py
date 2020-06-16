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
    cumul_actions = {i: [0,0,0] for i in range(num_elevators)}
    while env.now() <= episode_time: # Stop episode if time is over
        #print("-----------------------------------")
        #print(output)
        # 1. Get actions for the decision agents
        actions = copy.deepcopy(neg_action)
        for e_id, e_output in output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
            if e_output["last"] == False:
                continue
            #new_action = agents[e_id].action(
             #   np.reshape(e_output["state"], [1, nS])
            #)
            new_action = random.choice(range(0, 3, 1))
            actions[e_id] = new_action

            cumul_actions[e_id][new_action] += 1
        
        # 2. Take action in the Environment
        new_output = env.step(actions)
        # 3. (Skipped)
        for e_id, e_output in new_output.items():
            cumul_rewards[e_id] += e_output["reward"]
        # 4. overwrite old output with new output
        organize_output(output, new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time

    # Outside of episode
    print("Rewards: ", cumul_rewards)
    print("Elevator_1 Number of passengers served: ", 
        env.elevators[0].num_served
    )
    print("Elevator_1 Number of passengers carrying: ", 
        len(env.elevators[0].passengers)
    )
    print("Elevator_1 Actions: ", cumul_actions[0])
    print("Elevator_2 Number of passengers served: ", 
        env.elevators[1].num_served
    )
    print("Elevator_2 Number of passengers carrying: ", 
        len(env.elevators[1].passengers)
    )
    print("Elevator_1 Actions: ", cumul_actions[1])
    print("Total passengers generated:", env.generated_passengers)
    episode_rewards.append(cumul_rewards[0])
    episode_rewards.append(cumul_rewards[0])

