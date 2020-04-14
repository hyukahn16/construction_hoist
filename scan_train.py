import environment as gym
import time
import numpy as np
from heuristic import *
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

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

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
agents = [ScanAgent(total_floors) for _ in range(num_elevators)]
env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time)

episode_rewards = []
for e in range(10000): # number of episodes
    print("-----------{} Episode------------".format(e))
    output = {}
    organize_output(output, env.reset())

    cumul_rewards = [0 for _ in range(num_elevators)]
    cumul_actions = {0: [0,0,0]}
    while env.now() <= 1000: # Force-stop episode if time is over
        # 1. Get actions for the decision agents
        actions = copy.deepcopy(neg_action)
        for e_id, e_output in output.items(): # FIXME: need distinguish which elevator was decision elevator last time
            if e_output["last"] == False:
                continue
            legal = env.elevators[e_id].legal_actions()
            new_action = agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
            actions[e_id] = new_action

            cumul_actions[e_id][new_action] += 1
        
        # 2. Take action in the Environment
        new_output = env.step(actions)

        # 3. Update reward
        for e_id, e_output in new_output.items():
            cumul_rewards[e_id] += e_output["reward"]
            
        # 4. overwrite old output with new output
        organize_output(output, new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time

    # Outside of episode
    print("Rewards: ", cumul_rewards)
    print("Elevator_1 Number of passengers served: ", env.elevators[0].num_served)
    print("Elevator_1 Number of passengers carrying: ", len(env.elevators[0].passengers))
    print("Elevator_1 Max floor visited: ", env.elevators[0].max_visited)
    print("Elevator_1 Min floor visited ", env.elevators[0].min_visited)
    print("Actions: ", cumul_actions)
    print("Total passengers generated:", env.generated_passengers)
    episode_rewards.append(cumul_rewards[0])
    plt.plot([i for i in range(e + 1)], episode_rewards)
    plt.pause(0.8)
    plt.draw()

    # Save model
    '''
    for agent in agents:
        agent.model.save_weights('./checkpoints/cp', overwrite=True)
    print("Saved model")
    '''