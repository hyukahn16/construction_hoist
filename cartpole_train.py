import environment as gym
import time
import numpy as numpy
from model import *
import copy
import os
import matplotlib.pyplot as pyplot

num_elevators = 1
total_floors = 10
pass_gen_time = 75

nS = 30
nA = 3
lr = 0.001
gamma = 0.95
eps = 1
min_eps = 0.001
dr = 0.9995 # discount rate
batch_size = 24

agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, dr)]
env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time)

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

            agents[e_id].store(output[e_id]["state"], e_output["state"], 
                            replay_action, e_output["reward"], 1.0)
            if len(agents[e_id].memory) > batch_size: 
                agents[e_id].experience_replay(batch_size)

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