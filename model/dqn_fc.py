
import numpy as np
import torch
from torch.nn.functional import relu


class DQN_FC(torch.nn.Module):
    # https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/

    def __init__(self, total_floor, action_space):
        super(DQN_FC, self).__init__()

        self.fc1 = torch.nn.Linear(total_floor * 3, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, action_space) 

    def forward(self, state):
        '''Return Q-Values.'''
        output = relu(self.fc1(state))
        output = relu(self.fc2(output))
        output = relu(self.fc3(output))
        output = self.fc4(output)
        return output
    
    @staticmethod
    def cnn_to_fc(state):
        fc_state = []
        for floor in state[0]:
            for val in floor:
                fc_state.append(val)
        return fc_state