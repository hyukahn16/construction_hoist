import logging
# https://realpython.com/python-logging/
import random
import numpy as np
import copy
from environment import Environment, make
from model import DQN
from model import ReplayMemory

#logging.basicConfig(filename="log", filemode='w', level=logging.DEBUG)
logging.basicConfig(filename="log", filemode='w')
logging.debug("train.py: starting train.py")

if __name__ == "__main__":
    logging.debug("train.py: Simulation starting.")
    
    # Initialize replay memory capacity

    # Initialize policy network with random weights

    # Clone the policy network and call it target network

    # For each episode:
        # Initialize starting state

        # For each time step:
            # select an action thru exploration or exploitation
            # Execute selected action in an emulator/simulation
            # Observe reward and the next state
            # Store experience in replay memory
            # Sample random batch from replay memory
            # Preprocess states from batch (might not need for our case)
            # Pass bath of states to policy network
            # Calculate loss between output Q-values and the target Q-values
                # output Q comes from policy network and target Q comes from target network
            # Gradient descent updates weights in the policy network to minimize loss
                # After x time steps, weights in the target network are updated to the weights in the policy network
    

    # variables needed for Environment
    num_elevators = 2
    curr_floors = 50
    total_floors = 60
    passenger_generation_time = 20

    # variables needed for DQN
    update_freq = 4
    action_space = [0,1,2]
    lr = 0.005
    epsilon = 1.0
    min_epsilon = 0.1
    eps_dec_rate = 0.95
    gamma = 0.99

    REPLAY_MEMORY_SIZE = 2000
    MIN_REPLAY_MEMORY_SIZE = 1000
    MINIBATCH_SIZE = 500

    neg_action = [-1 for _ in range(num_elevators)]
    zero_actions = [0 for _ in range(3)]

    replays = [ReplayMemory(REPLAY_MEMORY_SIZE, MIN_REPLAY_MEMORY_SIZE, MINIBATCH_SIZE) \
                for _ in range(num_elevators)]
    agents = [DQN(update_freq, action_space, lr, epsilon, min_epsilon, eps_dec_rate, gamma) \
                for _ in range(num_elevators)] # Deep Q Network agents
    env = make(num_elevators, curr_floors, total_floors, passenger_generation_time)

    for _ in range(100):
        output = env.reset()
        while env.now() <= 1000:
            logging.debug("train.py: About to run env.step()")

            # Reset all actions and choose an action for Elevators
            actions = copy.deepcopy(neg_action)
            for idx in output["decision_agents"]:
                actions[idx] = agents[idx].get_action(output["state"])

            # Run simulation with the chosen actions
            new_output = env.step(actions)

            # Store replay memory for the Elevator that needs next action
            # And train the Elevator's DQN agent
            for idx in new_output["decision_agents"]:
                replay_action = copy.deepcopy(zero_actions)
                replay_action[actions[idx]] = 1
                replays[idx].push(output["state"], replay_action, new_output["state"], new_output["reward"][idx], 1.0)
                agents[idx].train(replays[idx])

            # Overwrite old output with new output
            output = new_output

            logging.debug("train.py: last rewards are {}".format(new_output['reward']))

        eps_end_reward = env.update_end_reward()
        logging.warning("train.py: score: {}".format(env.get_total_reward()))

    logging.debug("train.py: Simulation finished.")
    logging.shutdown()