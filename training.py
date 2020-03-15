import environment as gym
import time
import logging
from model import *
import numpy as np
import copy
import os
import torch

"""
Prep work
"""
def timed_function(func):
    def decorated_func(*args, **kwargs):
        s = time.time()
        output = func(*args, **kwargs)
        e = time.time()
        logger.info("{} finished in {} seconds".format(func.__name__,e-s))
        return output
    return decorated_func

if __name__=="__main__":
    num_elevators = 2
    curr_floors = 40 # FIXME: doesn't use this right now
    total_floors = 40
    lr = 0.001
    spawnRates = [1/1000]*total_floors
    avgWeight = 62 # Kilograms
    weightLimit = 907.185 # Kilograms
    loadTime = 1
    beta = 0.01 # exponential decay factor for target computation

    # initialize environment
    env = gym.make(num_elevators, total_floors, total_floors, 50)
    obs_space_size = env.observation_space_size
    action_space_size = env.action_space_size
    print("state space dimension", obs_space_size)
    print("action space size", action_space_size)

    update_freq = 4
    lr = 0.005
    epsilon = 1.0
    min_epsilon = 0.1
    eps_dec_rate = 0.995
    gamma = 0.99

    REPLAY_MEMORY_SIZE = 2000
    MIN_REPLAY_MEMORY_SIZE = 500 # Start using replay memory once it reaches this number
    MINIBATCH_SIZE = 500

    neg_action = [-1 for i in range(num_elevators)]
    zero_actions = [0 for _ in range(action_space_size)]
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
    for e in range(100): # number of episodes == 100
        print("-----------{} Episode------------".format(e))
        output = env.reset()
        cumul_rewards = [0, 0]
        while env.now() <= 10000: # Force stop episode if time is over

            # 1. Get actions for the decision agents
            actions = []
            for idx in output["decision_agents"]:
                legal = env.elevators[idx].legal_actions()
                new_action = agents[idx].get_action(output["states"], legal)
                actions.append(new_action)
            
            # 2. Take action in the Environment
            new_output = env.step(actions)

            # 3. Update Replay Memory and train agent
            for i, idx in enumerate(new_output["decision_agents"]):
                cumul_rewards[idx] += new_output["rewards"][i]
                replay_action = copy.deepcopy(zero_actions)
                replay_action[i] = 1
                replays[idx].push(output["states"], new_output["states"], 
                                replay_action, new_output["rewards"][i], 1.0)
                agents[idx].train(replays[idx])

            # 4. overwrite old output with new output
            output = new_output

            """
            print("step reward:")
            print("-----------------")
            for i, agent in enumerate(output["decision_agents"]):
                print("Elevator_{} reward: {}".format(agent, output["rewards"][i]))
            """

            # Print the state of the Environment
            #os.system('cls')
            # env.render()

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
        print("Epsilon value: ", agents[1].epsilon)
        print("Elevator_1 Number of passengers served: ", env.elevators[0].num_served)
        print("Elevator_2 Number of passengers served: ", env.elevators[1].num_served)
        print("Total passengers generated:", env.generated_passengers)
        #print("Agent 1's batch size: ", len(replays[0].state_memory))
        #print("Agent 2's batch size: ", len(replays[1].state_memory))