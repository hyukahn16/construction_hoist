'''
We will first try DQN with CNN architecture mentioned in the
Atari game research paper
'''

import tensorflow as tf

'''
Input state: current hoist position(s), hoist current weight capacity,
and call request (up/down) and its index of floor

- Current hoist position: (#_floors x #_elevators)
- hoist current weight capacity: (#_elevators x 1) - might need to fill up to #_floors 
- Call request for up: (#_floors x 1)
- Call request for down: (#_floors x 1)

- Total: (#_floors x #_elevators + 1 + 1 + 1)


height = #_floors
width = (DOWN calls = 1) + (UP calls = 1) + (#_elevators = 1) 
depth = (DOWN, UP calls = 0 or 1) or (Each elevator's capacity normalized)
---------------------
|P |                |
|A |                |
|S |                | 
|S |                |
|  |                |
|  |                |
|  |                |
---------------------
'''

class DQN():
    def __init__():

