import random

Transition = namedTuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0 # FIXME: what is this?

    def push(self, *args):
        '''Save a Transition.'''
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)