# DQN vs. Heuristic Test

import environment as gym
import time
import numpy as np
from model import *
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # FIXME: is this needed for tensorflow model save?
import matplotlib.pyplot as pyplot
from heuristic import ScanAgent, HumanAgent
import sys

def organize_output(output, new_output):

    for e_id, e_output in output.items():
        e_output["last"] = False

    for e_id, e_output in new_output.items():
        output[e_id] = e_output
        output[e_id]["last"] = True

def sanitize(array):
    str_arr = str(array)
    remove_char = "[],"
    for char in remove_char:
        str_arr = str_arr.replace(char, "")
    return str_arr

def read_stats(filename):
    with open(filename, 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if line in ["dqn", "scan", "human"]:
                #print(line)
                continue

            time_list = [float(i) for i in line.split(' ')]
            #print(time_list)
            print(np.average(time_list))



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
pass_gen_time = 50

nS = total_floors * 4
nA = 3
lr = 0.001
gamma = 0.95
min_eps = 0.1
eps = min_eps # DO NOT CHANGE! Epsilon is already at MINIMUM value.
eps_decay = 0.99999 # Not used since this is testing
batch_size = 24
test = True
episode_time = 10000
# End of hyperparameters
#######################################

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions

dqn_agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, 
    batch_size, test)]
scan_agents = [ScanAgent(total_floors) for _ in range(num_elevators)]
human_agents = [HumanAgent(total_floors) for _ in range(num_elevators)]

dqn_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, 
    episode_time, test_mode)
scan_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time,
    episode_time, test_mode)
human_mode = True
human_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, 
    episode_time, test_mode, human_mode, human_agents[0])



if os.path.exists('training_1.index'):
    dqn_agents[0].model.load_weights(dqn_agents[0].checkpoint_dir)
    print("Successfully loaded saved model")
    print("Using model =", test_mode)
else:
    sys.exit("Couldn't find saved model. Exiting.")

dqn_output = {}
scan_output = {}
human_output = {}
organize_output(dqn_output, dqn_env.reset())
organize_output(scan_output, scan_env.reset())
organize_output(human_output, human_env.reset())

dqn_cumul_rewards = [0 for _ in range(num_elevators)]
scan_cumul_rewards = [0 for _ in range(num_elevators)]
human_cumul_rewards = [0 for _ in range(num_elevators)]
dqn_cumul_actions = [0, 0, 0]
scan_cumul_actions = [0, 0, 0]
human_cumul_actions = [0, 0, 0]
dqn_step_rewards = []
scan_step_rewards = []
human_step_rewards = []
dqn_lift_time = []
scan_lift_time= []
human_lift_time= []
dqn_wait_time = []
scan_wait_time = []
human_wait_time = []

while dqn_env.now() <= episode_time:
    # 1. Get actions for the decision agents
    dqn_actions = copy.deepcopy(neg_action)
    for e_id, e_output in dqn_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
        if e_output["last"] == False:
            continue
        legal = dqn_env.elevators[e_id].legal_actions()
        new_action = dqn_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
        dqn_actions[e_id] = new_action
        dqn_cumul_actions[new_action] += 1

    # 2. Take action in the Environment
    dqn_new_output = dqn_env.step(dqn_actions)

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
        scan_cumul_actions[new_action] += 1

    # 2. Take action in the Environment
    scan_new_output = scan_env.step(scan_actions)

    # 3. Update values
    for e_id, e_output in scan_new_output.items():
        scan_cumul_rewards[e_id] += e_output["reward"]
        scan_step_rewards.append(e_output["reward"])
        scan_lift_time.append(e_output["lift_time"])
        scan_wait_time.append(e_output["wait_time"])

    # 4. overwrite old output with new output
    organize_output(scan_output, scan_new_output)

while human_env.now() <= episode_time:
    '''
    if human_agents[0].serving_passenger:
      print("Passenger current floor:", human_agents[0].serving_passenger.curr_floor)
      print("Passenger destination floor:", human_agents[0].serving_passenger.dest_floor)
      print("Elevator ID:", human_agents[0].serving_passenger.elevator)
    '''
    # 1. Get actions for the decision agents
    human_actions = copy.deepcopy(neg_action)
    for e_id, e_output in human_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
        if e_output["last"] == False:
            continue
        new_action = human_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
        human_actions[e_id] = new_action
        human_cumul_actions[new_action] += 1

    # 2. Take action in the Environment
    human_new_output = human_env.step(human_actions)

    # 3. Update values
    for e_id, e_output in human_new_output.items():
        human_cumul_rewards[e_id] += e_output["reward"]
        human_step_rewards.append(e_output["reward"])
        human_lift_time.append(e_output["lift_time"])
        human_wait_time.append(e_output["wait_time"])

    # 4. overwrite old output with new output
    organize_output(human_output, human_new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time

print("DQN ------")
print("Rewards: ", dqn_cumul_rewards)
print("Number of passengers served: ", dqn_env.elevators[0].num_served)
print("Number of passengers carrying: ", len(dqn_env.elevators[0].passengers))
print("Actions: ", dqn_cumul_actions)
print("Average Wait Time: ", np.average(dqn_wait_time))
print("Average Lift Time: ", np.average(dqn_lift_time))
print("Average Total Time: ", np.average(np.add(dqn_wait_time, dqn_lift_time)))
print()
print("SCAN ------")
print("Rewards: ", scan_cumul_rewards)
print("Number of passengers served: ", scan_env.elevators[0].num_served)
print("Number of passengers carrying: ", len(scan_env.elevators[0].passengers))
print("Actions: ", scan_cumul_actions)
print("Average Wait Time: ", np.average(scan_wait_time))
print("Average Lift Time: ", np.average(scan_lift_time))
print("Average Total Time: ", np.average(np.add(scan_wait_time, scan_lift_time)))
print()
print("HUMAN ------")
print("Rewards: ", human_cumul_rewards)
print("Number of passengers served: ", human_env.elevators[0].num_served)
print("Number of passengers carrying: ", len(human_env.elevators[0].passengers))
print("Actions: ", human_cumul_actions)
print("Average Wait Time: ", np.average(human_wait_time))
print("Average Lift Time: ", np.average(human_lift_time))
print("Average Total Time: ", np.average(np.add(human_wait_time, human_lift_time)))


# Wait Time Graph
fig1 = plt.figure(0, dpi=1200)
plt.plot([i for i in range(len(dqn_wait_time))], dqn_wait_time, label='DQN')
plt.plot([i for i in range(len(scan_wait_time))], scan_wait_time, label='SCAN')
plt.plot([i for i in range(len(human_wait_time))], human_wait_time, label='HUMAN')
plt.legend(loc='upper right')
plt.title("Average Wait Time")
fig1.savefig("{}_wait.png".format(test_mode))

# Lift Time Graph
fig2 = plt.figure(3, dpi=1200)
plt.plot([i for i in range(len(dqn_lift_time))], dqn_lift_time, label='DQN')
plt.plot([i for i in range(len(scan_lift_time))], scan_lift_time, label='SCAN')
plt.plot([i for i in range(len(human_lift_time))], human_lift_time, label='HUMAN')
plt.legend(loc='upper right')
plt.title("Average Lift Time")
fig2.savefig("{}_lift.png".format(test_mode))

# Total Time Graph
dqn_total_time = [dqn_lift_time[i] + dqn_wait_time[i]
    for i in range(len(dqn_wait_time))]
scan_total_time = [scan_lift_time[i] + scan_wait_time[i]
    for i in range(len(scan_wait_time))]
human_total_time = [human_lift_time[i] + human_wait_time[i]
    for i in range(len(human_wait_time))]
fig3 = plt.figure(2, dpi=1200)
plt.plot([i for i in range(len(dqn_total_time))], dqn_total_time, label='DQN')
plt.plot([i for i in range(len(scan_total_time))], scan_total_time, label='SCAN')
plt.plot([i for i in range(len(human_total_time))], human_total_time, label='HUMAN')
plt.legend(loc='upper right')
plt.title("Average Wait + Lift Time")
fig3.savefig("{}_total.png".format(test_mode))

plt.pause(0.01)
plt.draw()

remove_char = "[],"
with open('{}_data.txt'.format(test_mode), 'w') as f:
    f.write("dqn\n")
    f.write(str(dqn_cumul_rewards[0]) + "\n")
    f.write(str(dqn_env.elevators[0].num_served) + "\n")
    f.write(str(len(dqn_env.elevators[0].passengers)) + "\n")
    f.write(sanitize(dqn_cumul_actions) + "\n")
    f.write(sanitize(dqn_lift_time) + "\n")
    f.write(sanitize(dqn_wait_time) + "\n")
    f.write(sanitize(dqn_total_time) + "\n")
    f.write("scan\n")
    f.write(str(scan_cumul_rewards[0]) + "\n")
    f.write(str(scan_env.elevators[0].num_served) + "\n")
    f.write(str(len(scan_env.elevators[0].passengers)) + "\n")
    f.write(sanitize(scan_cumul_actions) + "\n")
    f.write(sanitize(scan_lift_time) + "\n")
    f.write(sanitize(scan_wait_time) + "\n")
    f.write(sanitize(scan_total_time) + "\n")
    f.write("human\n")
    f.write(str(human_cumul_rewards[0]) + "\n")
    f.write(str(human_env.elevators[0].num_served) + "\n")
    f.write(str(len(human_env.elevators[0].passengers)) + "\n")
    f.write(sanitize(human_cumul_actions) + "\n")
    f.write(sanitize(human_lift_time) + "\n")
    f.write(sanitize(human_wait_time) + "\n")
    f.write(sanitize(human_total_time) + "\n")
