# DQN vs. Heuristic Test
import environment as gym
import time
import numpy as np
from model import *
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # FIXME: is this needed for tensorflow model save?
import matplotlib.pyplot as pyplot

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

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
dqn_agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size)]
scan_agents = [ScanAgent(total_floors) for _ in range(num_elevators)]
dqn_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time)
scan_env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time)

if os.path.exists('training_1.index')
    dqn_agents[0].model.load_weights(agents[0].checkpoint_dir)
    print("Successfully loaded saved model")
else:
    print("Couldn't find saved model")
    print("Exiting test")
    exit()

dqn_episode_rewards = [] # Stores rewards from all episodes
scan_episode_rewards = []
for e in range(10000): # number of episodes
    print("-----------{} Episode------------".format(e))
    dqn_output = {}
    scan_output = {}
    organize_output(dqn_output, env.reset())
    organize_output(scan_output, env.reset())
    
    dqn_cumul_rewards = [0 for _ in range(num_elevators)]
    scan_cumul_rewards = [0 for _ in range(num_elevators)]
    dqn_cumul_actions = {0: [0,0,0]}
    scan_cumul_actions = {0: [0,0,0]}
    while env.now() <= 1000:
        # 1. Get actions for the decision agents
        dqn_actions = copy.deepcopy(neg_action)
        scan_actions = copy.deepcopy(neg_action)
        for e_id, e_output in dqn_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
            if e_output["last"] == False:
                continue
            legal = dqn_env.elevators[e_id].legal_actions()
            new_action = dqn_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
            dqn_actions[e_id] = new_action

            dqn_cumul_actions[e_id][new_action] += 1
        
        for e_id, e_output in scan_output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
            if e_output["last"] == False:
                continue
            legal = scan_env.elevators[e_id].legal_actions()
            new_action = scan_agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
            scan_actions[e_id] = new_action

            scan_cumul_actions[e_id][new_action] += 1
        # 2. Take action in the Environment
        dqn_new_output = dqn_env.step(dqn_actions)
        scan_new_output = scan_env.step(scan_actions)

        # 3. Update Replay Memory and train agent
        for e_id, e_output in dqn_new_output.items():
            dqn_cumul_rewards[e_id] += e_output["reward"]

            # Data used for replay memory
            replay_action = dqn_actions[e_id]
            state = [copy.deepcopy(output[e_id]["state"])]
            new_state = [copy.deepcopy(e_output["state"])]
            #state = [state]
            #new_state = [new_state]
            dqn_agents[e_id].store(state, replay_action, e_output["reward"], 
                                new_state, 1.0)
            
            # Train model
            if len(dqn_agents[e_id].memory) > batch_size: 
                dqn_agents[e_id].experience_replay(batch_size)

        for e_id, e_output in scan_new_output.items():
            scan_cumul_rewards[e_id] += e_output["reward"]

        # 4. overwrite old output with new output
        organize_output(dqn_output, dqn_new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time
        organize_output(scan_output, scan_new_output)

    # Outside of episode
    print("DQN ------")
    print("Rewards: ", dqn_cumul_rewards)
    print("Number of passengers served: ", dqn_env.elevators[0].num_served)
    print("Number of passengers carrying: ", len(dqn_env.elevators[0].passengers))
    dqn_episode_rewards.append(dqn_cumul_rewards[0])
    print("SCAN ------")
    print("Rewards: ", scan_cumul_rewards)
    print("Number of passengers served: ", scan_env.elevators[0].num_served)
    print("Number of passengers carrying: ", len(scan_env.elevators[0].passengers))
    scan_episode_rewards.append(scan_cumul_rewards[0])

    # FIXME: plot dqn and scan graphs together
    '''
    plt.plot([i for i in range(e + 1)], episode_rewards)
    plt.pause(0.01)
    plt.draw()
    '''