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

test_mode = "intermediate"
modes = ["intermediate", "uppeak", "downpeak", "lunch"]
if len(sys.argv) == 2:
    if sys.argv[1] not in modes:
        sys.exit("Mode name incorrect. Pick one of the following: {}".format(modes))
    test_mode = sys.argv[1]

#######################################
# Hyperparameters
num_elevators = 1
total_floors = 20
pass_gen_time = 75

nS = total_floors * 4
nA = 3
lr = 0.001
gamma = 0.95
min_eps = 0.01
eps = min_eps # DO NOT CHANGE! Epsilon is already at MINIMUM value.
eps_decay = 0.99999 # Not used since this is testing
batch_size = 24
test = True
episode_time = 10000
# End of hyperparameters
#######################################

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
dqn_agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size, test)]
scan_agents = [ScanAgent(total_floors) for _ in range(num_elevators)]
dqn_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, episode_time, test_mode)
scan_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, episode_time, test_mode)

if os.path.exists('training_1.index'):
    dqn_agents[0].model.load_weights(dqn_agents[0].checkpoint_dir)
    print("Successfully loaded saved model")
    print("Using model =", test_mode)
else:
    sys.exit("Couldn't find saved model. Exiting.")

dqn_output = {}
scan_output = {}
organize_output(dqn_output, dqn_env.reset())
organize_output(scan_output, scan_env.reset())

dqn_cumul_rewards = [0 for _ in range(num_elevators)]
scan_cumul_rewards = [0 for _ in range(num_elevators)]
dqn_cumul_actions = [0, 0, 0]
scan_cumul_actions = [0, 0, 0]
dqn_step_rewards = []
scan_step_rewards = []
dqn_lift_time = []
scan_lift_time= []
dqn_wait_time = []
scan_wait_time = []

while dqn_env.now() <= episode_time:
    # 1. Get actions for the decision agents
    dqn_actions = copy.deepcopy(neg_action)
    for e_id, e_output in dqn_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
        if e_output["last"] == False:
            continue
        legal = dqn_env.elevators[e_id].legal_actions()
        new_action = dqn_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
        dqn_actions[e_id] = new_action

    # 2. Take action in the Environment
    dqn_new_output = dqn_env.step(dqn_actions)
    dqn_cumul_actions[dqn_actions] += 1

    # 3. Update values
    for e_id, e_output in dqn_new_output.items():
        dqn_cumul_rewards[e_id] += e_output["reward"]
        dqn_step_rewards.append(e_output["reward"])
        dqn_lift_time.append(e_output["lift_time"])
        dqn_wait_time.append(e_output["wait_time"])

    # 4. overwrite old output with new output
    organize_output(dqn_output, dqn_new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time

while scan_env.now() <= episode_time:
    # 1. Get actions for the decision agents
    scan_actions = copy.deepcopy(neg_action)
    for e_id, e_output in scan_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
        if e_output["last"] == False:
            continue
        legal = scan_env.elevators[e_id].legal_actions()
        new_action = scan_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
        scan_actions[e_id] = new_action

    # 2. Take action in the Environment
    scan_new_output = scan_env.step(scan_actions)
    scan_cumul_actions[scan_actions] += 1

    # 3. Update values
    for e_id, e_output in scan_new_output.items():
        scan_cumul_rewards[e_id] += e_output["reward"]
        scan_step_rewards.append(e_output["reward"])
        scan_lift_time.append(e_output["lift_time"])
        scan_wait_time.append(e_output["wait_time"])

    # 4. overwrite old output with new output
    organize_output(scan_output, scan_new_output)

print("DQN ------")
print("Rewards: ", dqn_cumul_rewards)
print("Number of passengers served: ", dqn_env.elevators[0].num_served)
print("Number of passengers carrying: ", len(dqn_env.elevators[0].passengers))
print("Cumulative actions:", dqn_cumul_actions)
print()
print("SCAN ------")
print("Rewards: ", scan_cumul_rewards)
print("Number of passengers served: ", scan_env.elevators[0].num_served)
print("Number of passengers carrying: ", len(scan_env.elevators[0].passengers))
print("Cumulative actions:", scan_cumul_actions)

# Wait Time Graph
plt.figure(0)
plt.plot([i for i in range(len(dqn_lift_time))], dqn_lift_time, label='DQN')
plt.plot([i for i in range(len(scan_lift_time))], scan_lift_time, label='SCAN')
plt.legend(loc='upper right')
plt.title("Average Lift Time")

# Lift Time Graph
plt.figure(1)
plt.plot([i for i in range(len(dqn_wait_time))], dqn_wait_time, label='DQN')
plt.plot([i for i in range(len(scan_wait_time))], scan_wait_time, label='SCAN')
plt.legend(loc='upper right')
plt.title("Average Wait Time")

# Total Time Graph
dqn_total_time = [dqn_lift_time[i] + dqn_wait_time[i] for i in range(len(dqn_wait_time))]
scan_total_time = [scan_lift_time[i] + scan_wait_time[i] for i in range(len(scan_wait_time))]
plt.figure(2)
plt.plot([i for i in range(len(dqn_total_time))], dqn_total_time, label='DQN')
plt.plot([i for i in range(len(scan_total_time))], scan_total_time, label='SCAN')
plt.legend(loc='upper right')
plt.title("Average Wait + Lift Time")

#plt.pause(0.01)
plt.draw()
