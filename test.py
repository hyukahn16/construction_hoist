import simpy
import numpy as np
import torch
'''
def pass_gen(env):
    while True:
        yield env.timeout(500)
        print("Generating passenger")
    
env = simpy.Environment()
env.process(pass_gen(env))
env.run(until=env)
print("Finished test")
'''
class DQN(torch.nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = torch.nn.Conv2d()

