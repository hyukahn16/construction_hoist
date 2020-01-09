from env import Environment, make
import gym
print("starting train.py")

if __name__ == "__main__":
    print("Simulation starting.")
    
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

    env = make(2, 100) # (num_elevators, num_floors)
    env.reset()
    proc_pass_gen = env.simul_env.process(env.generate_passengers())
    
    env.simul_env.run(until=10000)
    
    print("Simulation finished.")