import random
import logging
import numpy as np

class ReplayMemory():
    def __init__(self, capacity, min_replay_memory_size, batch_size):
        self.capacity = capacity
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.state_memory = np.array([])
        self.new_state_memory = np.array([])
        self.reward_memory = np.array([])
        self.action_memory = np.array([])
        self.done_memory = np.array([])
        self.position = 0

    def push(self, state, new_state, action, reward, done):
        '''Save a Transition.'''
        
        if len(self.state_memory) < self.capacity:
            np.append(self.state_memory, state)
            np.append(self.new_state_memory, new_state)
            np.append(self.action_memory, action)
            np.append(self.reward_memory, reward)
            np.append(self.done_memory, done)
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