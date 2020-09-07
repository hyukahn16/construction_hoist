# Environment variables
num_elevators = 1
total_floors = 20
pass_gen_time = 160
episode_time = 5000
num_episodes = 10000

# general network variables
nS = total_floors * 4
nA = 3
lr = 0.001
gamma = 0.95
batch_size = 24

# DQN exclusive variables
eps = 1
min_eps = 0.1
eps_decay = 0.99999

# PPO exclusive variables
ppo_epochs = 8
ppo_steps = 256
critic_discount = 0.5
entropy_beta = 0.001
clip_range = 0.2
lmbda = 0.95

# Testing variables
use_saved = False
# Set this to False if you want to save the trained model
test_mode = True 