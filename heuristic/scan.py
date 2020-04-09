'''
Heuristic algorithm for Elevator which follows the SCAN algorithm
1. proceed in the same direction until the last request in that direction.
2. If there is no request, stop and proceed towards other direction if there is any requests from the other direction
Source: https://www.quora.com/What-algorithm-is-used-in-modern-day-elevators
'''

class ScanAgent():
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        self.curr_dir = 0 # -1: down, 0: idle, 1: up

    def get_action(self, state):
        pass
    


