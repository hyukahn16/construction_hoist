from environment import Environment, make
import logging
# https://realpython.com/python-logging/
import random

logging.basicConfig(filename="log", filemode='w', level=logging.DEBUG)
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
    num_elevators = 2
    env = make(num_elevators, 50, 60, 20) # (num_elevators, curr_floors, total_floors, Pas_gen_time)
    output = env.reset()

    while env.now() <= 1000:
        logging.debug("train.py: About to run env.step()")
        actions = [-1 for _ in range(num_elevators)]
        for idx in output["decision_agents"]:
            actions[idx] = random.randint(0, 2)

        output = env.step(actions) # FIXME: keep moving elevators up
        logging.debug("train.py: Current rewards are {}".format(output['reward']))

    env.update_end_reward()
    logging.debug("train.py: Rewards - {}".format(env.get_reward())) 
    logging.debug("train.py: Simulation finished.")
    logging.shutdown()