import environment as gym
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from model import *
import copy
import os
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
eps = 1
min_eps = 0.01
eps_decay = 0.99999
batch_size = 24
episode_time = 1000

use_saved = True

neg_action = [-1 for i in range(num_elevators)] # Used for state's next actions
agents = [DeepQNetwork(nS, nA, lr, gamma, eps, min_eps, eps_decay, batch_size)]
env = gym.make(num_elevators, total_floors, total_floors, pass_gen_time, episode_time)

if use_saved and os.path.exists('training_1.index'):
    print("Loading saved model")
    agents[0].model.load_weights(agents[0].checkpoint_dir)
else:
    print("Starting new model")

episode_rewards = [] # Stores rewards from all episodes
for e in range(10000): # number of episodes == 100
    print("-----------{} Episode------------".format(e))
    output = {}
    organize_output(output, env.reset())
    
    cumul_rewards = [0 for _ in range(num_elevators)]
    cumul_actions = {0: [0,0,0]}
    while env.now() <= episode_time: # Force stop episode if time is over
        # 1. Get actions for the decision agents
        actions = copy.deepcopy(neg_action)
        for e_id, e_output in output.items(): # FIXME: need distinguish which elevator was decision elevator last time - right now it doesn't matter because it's only 1 elevator but when it becomes multiple elevators we can't tell right now
            if e_output["last"] == False:
                continue
            legal = env.elevators[e_id].legal_actions() # FIXME: not using this
            new_action = agents[e_id].action(np.reshape(e_output["state"], [1, nS]))
            actions[e_id] = new_action

            cumul_actions[e_id][new_action] += 1
        
        # 2. Take action in the Environment
        new_output = env.step(actions)

        # 3. Update Replay Memory and train agent
        for e_id, e_output in new_output.items():
            cumul_rewards[e_id] += e_output["reward"]
            replay_action = actions[e_id]

            state = copy.deepcopy(output[e_id]["state"])
            new_state = copy.deepcopy(e_output["state"])
            state = [state]
            new_state = [new_state]

            agents[e_id].store(state, replay_action, e_output["reward"], 
                                new_state, 1.0)
            if len(agents[e_id].memory) > batch_size: 
                agents[e_id].experience_replay(batch_size)

        # 4. overwrite old output with new output
        organize_output(output, new_output) # FIXME: need to distinguishi which elevator was the decision elevator last time

    # Outside of episode
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
    plt.pause(0.01)
    plt.draw()

    # Save model (outdated)
    '''
    for agent in agents:
        agent.model.save_weights('./checkpoints/cp', overwrite=True)
    print("Saved model")
    '''