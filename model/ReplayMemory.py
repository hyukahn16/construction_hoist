import random
import logging
import numpy as np
from .dqn_fc import DQN_FC

class ReplayMemory():
    def __init__(self, capacity, min_replay_memory_size, batch_size):
        self.capacity = capacity
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size

        # Core memory portion (this is what the agent learns from)
        self.state_memory = np.array([])
        self.new_state_memory = np.array([])
        self.reward_memory = np.array([])
        self.action_memory = np.array([])
        self.done_memory = np.array([])

        self.position = 0

    def push(self, state, new_state, action, reward, done):
        '''Save a Transition.'''

        if len(self.state_memory) == 0:
            self.state_memory = np.array([state])
            self.new_state_memory = np.array([new_state])
            self.action_memory = np.array([action])
            self.reward_memory = np.array([reward])
            self.done_memory = np.array([done])

        if len(self.state_memory) < self.capacity:
            self.state_memory = np.append(self.state_memory, [state], axis=0)
            self.new_state_memory = np.append(self.new_state_memory, [new_state], axis=0)
            self.action_memory = np.append(self.action_memory, [action], axis=0)
            self.reward_memory = np.append(self.reward_memory, [reward], axis=0)
            self.done_memory = np.append(self.done_memory, [done], axis=0)
        else:
            self.state_memory[self.position] = state
            self.new_state_memory[self.position] = new_state
            self.action_memory[self.position] = action
            self.reward_memory[self.position] = reward
            self.done_memory[self.position] = done
        self.position = (self.position + 1) % self.capacity

    def can_sample(self):
        if len(self.state_memory) >= self.min_replay_memory_size:
            return True
        return False

    def sample(self):
        return np.random.choice(self.min_replay_memory_size, self.batch_size)

    def __len__(self):
        return len(self.state_memory)