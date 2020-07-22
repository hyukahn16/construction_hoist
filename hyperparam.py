# Environment variables
num_elevators = 1
total_floors = 20
pass_gen_time = 80
episode_time = 5000
num_episodes = 10000

# DQN variables
nS = total_floors * 4
nA = 3
lr = 0.001
gamma = 0.95
eps = 1
min_eps = 0.1
eps_decay = 0.99999
batch_size = 24

# Testing variables
use_saved = False
# Set this to False if you want to save the trained model
test_mode = True 