import environment as gym
import time
import numpy as np
from model import *
import copy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as pyplot

def overwrite(output, new_output):
    '''
    Requires: old output and new output
    Modifies: output
    Effects: output takes on the values of new_output and marks the last
        decision elevators.
    '''
    for e_id, e_output in output.items():
        e_output["last"] = False

    for e_id, e_output in new_output.items():
        output[e_id] = e_output
        output[e_id]["last"] = True

### HYPERPARAMETER ###
num_elevators = 2
total_floors = 20
pass_gen_time = 50

nS = total_floors * (num_elevators * 2 + 2)
nA = 3
lr = 0.001
gamma = 0.95
eps = 1
min_eps = 0.01
eps_decay = 0.99995
batch_size = 24
eps_time = 10000
num_eps = 10000
use_saved = False
### HYPERPARAMETER END ###

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size)]
env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, eps_time)

if use_saved and os.path.exists('training_1.index'):
    print("Loading saved model")
    agents[0].model.load_weights(agents[0].checkpoint_dir)
else:
    print("Starting new model")

eps_rewards = [[] for _ in range(num_elevators)] # Stores rewards from all episodes
for e in range(num_eps):
    print("-----------Episode {}------------".format(e))
    output = {}
    overwrite(output, env.reset())  
    
    eps_reward = [0 for _ in range(num_elevators)]
    eps_actions = {0: [0,0,0]}
    while env.now() <= eps_time: # Force stop episode if time is over
        # 1. Get actions for the decision agents
        actions = copy.deepcopy(neg_action)
        for e_id, e_output in output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
            if e_output["last"] == False:
                continue
            legal = env.elevators[e_id].legal_actions() # FIXME: unused
            new_action = agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
            actions[e_id] = new_action
            eps_actions[e_id][new_action] += 1
        
        # 2. Take action in the Environment
        new_output = env.step(actions)

        # 3. Update Replay Memory and train agent
        for e_id, e_output in new_output.items():
            eps_reward[e_id] += e_output["reward"]
            eps_rewards[e_id].append(e_output["reward"])

            state = copy.deepcopy(output[e_id]["state"])
            new_state = copy.deepcopy(e_output["state"])
            state = [state]
            new_state = [new_state]

            agents[e_id].store(state, actions[e_id], e_output["reward"], 
                                new_state, 1.0)
            if len(agents[e_id].memory) > batch_size: 
                agents[e_id].experience_replay(batch_size)

        # 4. overwrite old output with new output and mark the last decision elevators
        overwrite(output, new_output)

    # Outside of eps
    print("Rewards: ", eps_reward)
    print("Epsilon value:", agents[0].epsilon)
    print("Elevator_1 Number of passengers served: ", env.elevators[0].num_served)
    print("Elevator_1 Number of passengers carrying: ", len(env.elevators[0].passengers))
    print("Actions: ", eps_actions)
    print("Total passengers generated:", env.generated_passengers)
    plt.plot([i for i in range(e + 1)], eps_rewards) # FIXME: eps_rewards holds all rewards for all Elevators
    plt.pause(0.01)
    plt.draw()

    # Save model (outdated)
    '''
    for agent in agents:
        agent.model.save_weights('./checkpoints/cp', overwrite=True)
    print("Saved model")
    '''