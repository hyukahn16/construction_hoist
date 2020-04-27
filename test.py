# DQN vs. Heuristic Test

import environment as gym
import time
import numpy as np
from model import *
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # FIXME: is this needed for tensorflow model save?
import matplotlib.pyplot as pyplot
from heuristic import ScanAgent
import sys

def organize_output(output, new_output):

    for e_id, e_output in output.items():
        e_output["last"] = False

    for e_id, e_output in new_output.items():
        output[e_id] = e_output
        output[e_id]["last"] = True

num_elevators = 1
total_floors = 10
pass_gen_time = 75

nS = total_floors * 4
nA = 3
lr = 0.001
gamma = 0.95
min_eps = 0.01
eps = min_eps # Epsilon is already at MINIMUM value
eps_decay = 0.99999 # Not used since this is testing
batch_size = 24
test = True

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
dqn_agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size, test)]
scan_agents = [ScanAgent(total_floors) for _ in range(num_elevators)]
dqn_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, True)
scan_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, True)

if os.path.exists('training_1.index'):
    dqn_agents[0].model.load_weights(dqn_agents[0].checkpoint_dir)
    print("Successfully loaded saved model")
else:
    print("Couldn't find saved model")
    print("Exiting test")
    sys.exit("Couldn't find saved model. Exiting.")

dqn_output = {}
scan_output = {}
organize_output(dqn_output, dqn_env.reset())
organize_output(scan_output, scan_env.reset())

dqn_cumul_rewards = [0 for _ in range(num_elevators)]
scan_cumul_rewards = [0 for _ in range(num_elevators)]
dqn_cumul_actions = {0: [0,0,0]}
scan_cumul_actions = {0: [0,0,0]}

while dqn_env.now() <= 1000:
    # 1. Get actions for the decision agents
    dqn_actions = copy.deepcopy(neg_action)
    for e_id, e_output in dqn_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
        if e_output["last"] == False:
            continue
        legal = dqn_env.elevators[e_id].legal_actions()
        new_action = dqn_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
        dqn_actions[e_id] = new_action

        dqn_cumul_actions[e_id][new_action] += 1
    
    # 2. Take action in the Environment
    dqn_new_output = dqn_env.step(dqn_actions)

    # 3. DQN - Update Replay Memory and train agent
    for e_id, e_output in dqn_new_output.items():
        dqn_cumul_rewards[e_id] += e_output["reward"]

    # 4. overwrite old output with new output
    organize_output(dqn_output, dqn_new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time

while scan_env.now() <= 1000:
    # 1. Get actions for the decision agents
    scan_actions = copy.deepcopy(neg_action)
    for e_id, e_output in scan_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
        if e_output["last"] == False:
            continue
        legal = scan_env.elevators[e_id].legal_actions()
        new_action = scan_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
        scan_actions[e_id] = new_action

        scan_cumul_actions[e_id][new_action] += 1

    # 2. Take action in the Environment
    scan_new_output = scan_env.step(scan_actions)

    # 3. SCAN
    for e_id, e_output in scan_new_output.items():
        scan_cumul_rewards[e_id] += e_output["reward"]

    # 4. overwrite old output with new output
    organize_output(scan_output, scan_new_output)

print("DQN ------")
print("Rewards: ", dqn_cumul_rewards)
print("Number of passengers served: ", dqn_env.elevators[0].num_served)
print("Number of passengers carrying: ", len(dqn_env.elevators[0].passengers))
print()
print("SCAN ------")
print("Rewards: ", scan_cumul_rewards)
print("Number of passengers served: ", scan_env.elevators[0].num_served)
print("Number of passengers carrying: ", len(scan_env.elevators[0].passengers))

# FIXME: plot dqn and scan graphs together
'''
plt.plot([i for i in range(e + 1)], episode_rewards)
plt.pause(0.01)
plt.draw()
'''