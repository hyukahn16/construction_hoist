import simpy
import numpy as np
import torch
from collections import deque
import random

'''
def pass_gen(env):
    while True:
        yield env.timeout(500)
        print("Generating passenger")
    
env = simpy.Environment()
env.process(pass_gen(env))
env.run(until=env)
print("Finished test")
'''
REPLAY_MEMORY_SIZE = 50_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 64

class DQNModel(torch.nn.Module):
    # DQN Pytorch example code: https://github.com/transedward/pytorch-dqn
    def __init__(self):
        super(DQNModel, self).__init__()
                
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = torch.nn.Linear()
        self.fc2 = torch.nn.Linear(, 3)

    def forward(self, input):
        '''Return Q-Values.'''

        output = F.relu(self.conv1(input))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.fc4(output.view(output.size(0), -1)
        return self.fc5(output) # Returns Q-values

class DQN():
    def __init__(self, update_freq):
        self.model = DQNModel() # Fit/Train
        self.target_model = DQNModel() # Predict Q-Values

        self.optimizer = torch.optim.RMSprop(self.model.parameters())

        self.target_update_counter = 0
        self.target_update_freq = 4 # FIXME: replace with update_freq

        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)
                                 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def conv2d_size_out(self, size, kernel_size, stride):
        '''Helper function to calculate size after a layer.'''
        return (size - (kernel_size - 1) -1) // stride + 1

    def train(self):
        '''Train main network every step during episode.'''

        # Train only if there are enough replay memories
        if len(self.replace_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get batch of samples from replay memory
        transitions = self.replay_memory.sample(MINIBATCH_SIZE)
        batch = Transition(*zip(*transitions))

        current_states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.FloatTensor(batch[1]).to(self.device)
        new_states = torch.FloatTensor(batch[2]).to(self.device)
        rewards = torch.FloatTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).to(self.device)

        # get Q-Values from self.model        
        current_Q_values = self.model.forward(current_states).gather(1, actions.unsqueeze(1))

        next_max_q = self.target_model.forward(new_states).detach().max(1)[0]
        next_Q_values = done * next_max_q

        target_Q_values = rewards + (gamma * next_Q_values)

        # Compute Bellman error
        bellman_error =target_Q_values - current_Q_values

        # Clip the Bellman error between [-1, 1]
        bellman_error_clip = bellman_error.clamp(-1, 1)

        d_error = bellman_error_clip * -1

        # Clear previous gradients before backward pass
        self.optimizer.zero_grad()

        # Run backward pass
        current_Q_values.backward(d_error.data.unsqueeze(1))

        # Perform update
        self.optimizer.step()
        self.target_update_counter += 1

        # Periodically, update the target model
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(Q.state_dict())


# initialize NNet for each elevator (2 in current case)
agents = []
for _ in range(2):
    agents.append(DQN())
