import random
from collections import namedtuple
import logging

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory():
    def __init__(self, capacity, min_replay_memory_size):
        self.capacity = capacity
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = min_replay_memory_size
        self.memory = []
        self.position = 0

    def push(self, *args):
        '''Save a Transition.'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def can_sample(self):
        if len(self.memory) >= self.min_replay_memory_size:
            return True
        return False

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)