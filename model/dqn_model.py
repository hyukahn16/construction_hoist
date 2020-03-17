import simpy
import numpy as np
import torch
from torch.nn.functional import relu
import random
import logging
from .dqn_cnn import DQN_CNN
from .dqn_fc import DQN_FC

class DQN():
    def __init__(self, update_freq, action_space, learning_rate, 
                epsilon, min_epsilon, eps_dec_rate, 
                gamma, current_floors, total_floors, model_type="cnn"):
        self.model_type = model_type

        # CNN or FC
        self.model = DQN_FC(total_floors, action_space) # Fit/Train
        self.target_model = DQN_FC(total_floors, action_space) # Predict Q-Values
        if self.model_type == "cnn":
            self.model = DQN_CNN()
            self.target_model = DQN_CNN()

        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr = learning_rate)
        self.criterion = torch.nn.MSELoss()

        self.current_floors = current_floors
        self.total_floors = total_floors
        self.target_update_counter = 0
        self.target_update_freq = update_freq
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.eps_dec_rate = eps_dec_rate
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_action(self, state, legal_actions):
        rand = np.random.random()
        action = -1
        if rand < self.epsilon:
            action = np.random.choice(list(legal_actions))
        else:
            # pick highest Q-value action
            stateTensor = torch.tensor(state, dtype=torch.float, device=self.device)
            actions = self.model.forward(stateTensor.unsqueeze(0)) # unsqueeze will create [ stateTensor ]
            #action = (torch.argmax(actions[0])).item() # get the highest Q-value action
            highest_Q = float('-inf')
            highest_Q_action = -1
            for a in legal_actions:
                if actions[0][a] > highest_Q:
                    highest_Q_action = a
            action = highest_Q_action
            
        assert(action != -1)
        return action

    def train(self, replay_memory):
        '''Train main network every step during episode.'''

        # Train only if there are enough replay memories
        if not replay_memory.can_sample():
            return

        # Get batch of samples from replay memory
        batch_idx = replay_memory.sample()
        
        #current_states = torch.tensor(replay_memory.state_memory[batch_idx], dtype=torch.float, device=self.device)
        current_states = torch.FloatTensor(replay_memory.state_memory[batch_idx]).to(self.device)
        actions = torch.LongTensor(replay_memory.action_memory[batch_idx]).to(self.device)
        new_states = torch.FloatTensor(replay_memory.new_state_memory[batch_idx]).to(self.device)
        rewards = torch.FloatTensor(replay_memory.reward_memory[batch_idx]).to(self.device)
        done = torch.FloatTensor(replay_memory.done_memory[batch_idx]).to(self.device)

        self.epsilon = self.epsilon * self.eps_dec_rate \
                        if self.epsilon > self.min_epsilon \
                        else self.min_epsilon

        self.optimizer.zero_grad()
        current_Q_values = self.model.forward(current_states)
        current_Q_values = current_Q_values.gather(1, actions)

        new_Q_Values = self.target_model.forward(new_states)
        new_Q_max = new_Q_Values.detach().max(1)[0]
        new_Q = rewards + (self.gamma * new_Q_max)
        loss = self.criterion(current_Q_values, new_Q.unsqueeze(1))
        loss.backward()
        
        self.optimizer.step()
        self.target_update_counter += 1

        # Periodically, update the target model
        if self.target_update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())