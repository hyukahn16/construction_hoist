import numpy as np
import torch
from torch.nn.functional import relu

class DQN_CNN(torch.nn.Module):
    # DQN Pytorch example code: https://github.com/transedward/pytorch-dqn
    def __init__(self):
        super(DQN_CNN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=2, stride=1)

        height = self.conv2d_size_out(self.conv2d_size_out(40, 3, 1), 2, 1) # FIXME: change so that the input size is taken from training.py
        width = self.conv2d_size_out(self.conv2d_size_out(8, 3, 1), 2, 1)

        self.fc1 = torch.nn.Linear(height * width * 64 , 512) # 64 comes from output filter of last Conv layer
        self.fc2 = torch.nn.Linear(512, 10) # FIXME: input should be taken from training.py

    def conv2d_size_out(self, size, kernel_size, stride):
        '''Helper function to calculate size after a layer.'''
        return (size - (kernel_size - 1) -1) // stride + 1

    def forward(self, state):
        '''Return Q-Values.'''

        output = relu(self.conv1(state))
        output = relu(self.conv2(output))
        output = relu(self.fc1(output.view(output.size(0), -1)))
        return self.fc2(output) # Returns Q-values