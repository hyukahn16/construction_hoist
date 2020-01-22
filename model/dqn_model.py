import simpy
import numpy as np
import torch
from torch.nn.functional import relu
import random
import logging
from .ReplayMemory import Transition

class DQNModel(torch.nn.Module):
    # DQN Pytorch example code: https://github.com/transedward/pytorch-dqn
    def __init__(self):
        super(DQNModel, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=1)
        #self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)

        height = self.conv2d_size_out(self.conv2d_size_out(50, 3, 1), 2, 1)
        width = self.conv2d_size_out(self.conv2d_size_out(4, 3, 1), 2, 1)

        logging.debug(height)
        logging.debug(width)
        logging.debug("dimensions")

        self.fc1 = torch.nn.Linear(height * width * 64 , 512)
        self.fc2 = torch.nn.Linear(512, 3)

    def conv2d_size_out(self, size, kernel_size, stride):
        '''Helper function to calculate size after a layer.'''
        return (size - (kernel_size - 1) -1) // stride + 1

    def forward(self, state):
        '''Return Q-Values.'''

        output = relu(self.conv1(state))
        output = relu(self.conv2(output))
        #output = F.relu(self.conv3(output))
        output = relu(self.fc1(output.view(output.size(0), -1)))
        return self.fc2(output) # Returns Q-values

class DQN():
    def __init__(self, update_freq, action_space, lr, epsilon, min_epsilon, eps_dec_rate, gamma):
        self.model = DQNModel() # Fit/Train
        self.target_model = DQNModel() # Predict Q-Values

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = lr)
        self.loss = torch.nn.MSELoss() # FIXME: Not used      

        self.target_update_counter = 0
        self.target_update_freq = update_freq
        self.action_space = action_space
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.eps_dec_rate = eps_dec_rate
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, state):
        rand = np.random.random()
        if rand < self.epsilon:
            # take random action
            action = np.random.choice(self.action_space)
        else:
            # pick highest Q-value action
            actions = self.model.forward(state)
            action = torch.argmax(actions).items()
        
        return action

    def train(self, replay_memory):
        '''Train main network every step during episode.'''

        # Train only if there are enough replay memories
        if not replay_memory.can_sample():
            return

        # Get batch of samples from replay memory
        transitions = replay_memory.sample()
        batch = Transition(*zip(*transitions))

        current_states = torch.FloatTensor(batch[0]).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        new_states = torch.LongTensor(batch[2]).to(self.device)
        rewards = torch.LongTensor(batch[3]).to(self.device)
        done = torch.FloatTensor(batch[4]).to(self.device)

        # Get the target model's Q-Values   
        current_Q_values = self.model.forward(current_states)
        logging.debug(current_Q_values)
        logging.debug("Q Value")
        current_Q_values = current_Q_values.gather(1, actions.unsqueeze(1))
        next_max_q = self.target_model.forward(new_states).detach().max(1)[0]
        next_Q_values = done * next_max_q
        target_Q_values = rewards + (gamma * next_Q_values)

        # Decrease epsilon by the given rate (self.eps_dec_rate must be less than 1)
        self.epsilon = self.epsilon * self.eps_dec_rate \
                        if self.epsilon > self.min_epsilon \
                        else self.min_epsilon

        # Compute Bellman error
        bellman_error = target_Q_values - current_Q_values
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