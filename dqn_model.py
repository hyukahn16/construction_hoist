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

class Replay():
    def __init__(self, current_state, action, reward, new_state, done):
        self.current_state = None
        self.action = None
        self.reward = None
        self.new_state = None
        self.done = None

class DQNModel(torch.nn.Module):
    # DQN Pytorch example code: https://github.com/transedward/pytorch-dqn
    def __init__(self):
        super(DQNModel, self).__init__()
                
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d()
        self.conv3 = torch.nn.Conv2d()

        self.fc1 = torch.nn.Linear()
        self.fc2 = torch.nn.Linear()

    def forward(self, input):
        '''Return Q-Values.'''

        output = F.relu(self.conv1(input))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = F.relu(self.fc4(output.view(output.size(0), -1)
        return self.fc5(output) # Returns Q-values

class DQN():
    def __init__(self):
        self.model = DQNModel() # Fit/Train
        self.target_model = DQNModel() # Predict Q-Values
        self.target_update_counter = 0
        self.optimizer = torch.optim.RMSprop(self.model.parameters())

        self.replay_memory = ReplayMemory(REPLAY_MEMORY_SIZE)

    def conv2d_size_out(self, size, kernel_size, stride):
        '''Helper function to calculate size after a layer.'''
        return (size - (kernel_size - 1) -1) // stride + 1

    def train(self):
        '''Train main network every step during episode.'''

        # Train only if there are enough replay memories
        if len(self.replace_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get batch of samples from replay memory
        transitions = replay_memory.sample(MINIBATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Get current states from minibatch
        current_states = np.array([memory[0] for memory in minibatch])
        # get Q-Values from self.model        
        current_q = self.model(current_states).gather(1, )

        # Get future states from minibatch
        new_states = np.array(memory[1] for memory in minibatch])
        # get Q-Values from self.target_model
        future_q = self.target_model.predict(new_states)


    def update_replay_memory(self, memory):
        self.replay_memory.append(memory)


# initialize NNet for each elevator (2 in current case)
agents = []
for _ in range(2):
    agents.append(DQN())