'''
Input state: current hoist position(s), hoist current weight capacity,
and call request (up/down) and its index of floor

- Current hoist position: (#_floors x #_elevators)
- hoist current weight capacity: (#_elevators x 1) - might need to fill up to #_floors 
- Call request for up: (#_floors x 1)
- Call request for down: (#_floors x 1)

- Total: (#_floors x #_elevators + 1 + 1 + 1)

---------------------------------------

TURN SIMULATION STATE INTO AN IMAGE (LIKE ATARI GAME)
height = #_floors
width = (DOWN calls = 1) + (UP calls = 1) + (#_elevators = 2 for now) 
depth = 
    - DOWN, UP calls = 0 or 1
    - Each elevator's capacity normalized = [0, 1]
---------------------
|D |   | E |        |
|O |   | L |        |
|W |   | E |        | 
|N |   |   |        |
|  |   |   | E |    |
|  | U |   | L |    |
|  | P |   | E |    |
---------------------

First conv layer: kernel size = 3x3

Cautions:
- Will CNN work fine on a narrow image?

Deep reinforcmenet learning on changing data sizes
https://datascience.stackexchange.com/questions/61536/deep-reinforcement-learning-on-changing-data-sizes
'''

import tensorflow as tf

class DQN():
    def create_model():
        

    def __init__():

