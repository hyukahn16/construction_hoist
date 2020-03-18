import environment as gym
import time
import logging
from model import *
import numpy as np
import copy
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def organize_output(output, new_output):
    for e_id, e_output in output.items():
        e_output["last"] = False

    for e_id, e_output in new_output.items():
        output[e_id] = e_output
        output[e_id]["last"] = True


if __name__=="__main__":
    num_elevators = 1
    curr_floors = 10 # FIXME: doesn't use this right now
    total_floors = 10
    lr = 0.001
    spawnRates = [1/1000]*total_floors
    avgWeight = 62 # Kilograms
    weightLimit = 907.185 # Kilograms
    loadTime = 1
    beta = 0.01 # exponential decay factor for target computation
    pass_gen_time = 75

    # initialize environment
    env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time)
    obs_space_size = env.observation_space_size
    action_space_size = env.action_space_size
    print("state space dimension", obs_space_size)
    print("action space size", action_space_size)

    update_freq = 4
    lr = 0.00025
    epsilon = 1.0
    min_epsilon = 0.1
    eps_dec_rate = 0.99995
    gamma = 0.999

    REPLAY_MEMORY_SIZE = 500
    MIN_REPLAY_MEMORY_SIZE = 70 # Start using replay memory once it reaches this number
    MINIBATCH_SIZE = 70

    neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
    zero_actions = [0 for _ in range(action_space_size)] # Used for ReplayMemory

    # initialize replay memory
    replays = [ReplayMemory(REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE) \
                for _ in range(num_elevators)]

    # initialize a NNet for each elevator
    agents = [DQN(update_freq, action_space_size, lr, epsilon, min_epsilon, 
                eps_dec_rate, gamma, curr_floors, total_floors, "fc") \
                for _ in range(num_elevators)]
    

    """
    Begin training
    """
    episode_rewards = []
    for e in range(1000): # number of episodes == 100
        print("-----------{} Episode------------".format(e))
        output = {}
        organize_output(output, env.reset())
        
        cumul_rewards = [0 for _ in range(num_elevators)]
        cumul_actions = {0: [0,0,0]}
        while env.now() <= 1000: # Force stop episode if time is over

            # 1. Get actions for the decision agents
            actions = copy.deepcopy(neg_action)
            for e_id, e_output in output.items(): # FIXME: need distinguish which elevator was decision elevator last time
                if e_output["last"] == False:
                    continue
                legal = env.elevators[e_id].legal_actions()
                new_action = agents[e_id].get_action(e_output["state"], legal)
                actions[e_id] = new_action

                cumul_actions[e_id][new_action] += 1
            
            # 2. Take action in the Environment
            new_output = env.step(actions)

            # 3. Update Replay Memory and train agent
            for e_id, e_output in new_output.items():
                cumul_rewards[e_id] += e_output["reward"]
                #replay_action = copy.deepcopy(zero_actions)
                #replay_action[actions[e_id]] = 1

                replay_action = actions[e_id]

                replays[e_id].push(output[e_id]["state"], e_output["state"], 
                                replay_action, e_output["reward"], 1.0)
                agents[e_id].train(replays[e_id])

            # 4. overwrite old output with new output
            organize_output(output, new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time

            # Print the state of the Environment
            '''
            os.system('cls')
            print("episode {}".format(e))
            print("Rewards: ", cumul_rewards)
            print("Actions: ", cumul_actions)
            env.render()
            '''

        # Outside of episode
        '''
        torch.save({
            'epoch': e,
            'model_state_dict': agents[0].model,
            'optimizer_state_dict': agents[0].optimizer,
            'loss': agents[0].loss,
        }, "./checkpoint/")
        '''
        print("Rewards: ", cumul_rewards)
        print("Epsilon value:", agents[0].epsilon)
        print("Elevator_1 Number of passengers served: ", env.elevators[0].num_served)
        print("Elevator_1 Number of passengers carrying: ", len(env.elevators[0].passengers))
        print("Elevator_1 Max floor visited: ", env.elevators[0].max_visited)
        print("Elevator_1 Min floor visited ", env.elevators[0].min_visited)
        print("Actions: ", cumul_actions)
        print("Total passengers generated:", env.generated_passengers)
        episode_rewards.append(cumul_rewards[0])
        plt.plot([i for i in range(e + 1)], episode_rewards)
        plt.pause(0.05)
        plt.draw()