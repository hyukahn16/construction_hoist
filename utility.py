import copy
import numpy as np
from hyperparam import *

def organize_output(output, new_output):
    for e_id, e_output in output.items():
        e_output["last"] = False

    for e_id, e_output in new_output.items():
        output[e_id] = e_output
        output[e_id]["last"] = True

def time_wrapper(func, args):
    def time_wrapped():
        return func(args)
    return time_wrapped

def run_episode(env, agents, episode_time, nA, nS, batch_size):
    output = {}
    organize_output(output, env.reset())
    neg_action = [-1 for i in range(len(agents))]
    cumul_rewards = [0 for _ in range(len(agents))]
    cumul_actions = {i: [j for j in range(nA)] for i in range(len(agents))}
    while env.now() <= episode_time:
        # 1. Get actions for the decision agents
        actions = copy.deepcopy(neg_action)
        for e_id, e_output in output.items():
            if e_output["last"] == False:
                continue
            new_action = agents[e_id].action(
                np.reshape(e_output["state"], [1, nS])
            )
            actions[e_id] = new_action

            cumul_actions[e_id][new_action] += 1
        
        # 2. Take action in the Environment
        new_output = env.step(actions)

        # 3. Train agents
        for e_id, e_output in new_output.items():
            cumul_rewards[e_id] += e_output["reward"]
            replay_action = actions[e_id]

            state = copy.deepcopy(output[e_id]["state"])
            new_state = copy.deepcopy(e_output["state"])
            state = [state]
            new_state = [new_state]

            agents[e_id].store(state, replay_action, e_output["reward"], 
                                new_state, 1.0)
            if len(agents[e_id].memory) > batch_size: 
                agents[e_id].experience_replay(batch_size)

        # 4. overwrite old output with new output
        organize_output(output, new_output)

    return { "rewards": cumul_rewards, "actions": cumul_actions}

def episode_wrapper(env, agents, episode_time, nA, nS, batch_size):
    def wrapped():
        return run_episode(env, agents, episode_time, nA, nS, batch_size)
    return wrapped

def print_hyperparam():
    print("Number of elevators: ", num_elevators)
    print("Total floors: ", total_floors)
    print("Passenger generation speed: ", pass_gen_time)
    print("Episode time: ", episode_time)
    print("Max number of episodes: ", num_episodes)
    print()
    print("Learning rate: ", lr)
    print("Gamma: ", gamma)
    print("Max starting epsilon: ", eps)
    print("Min epsilon: ", min_eps)
    print("Epsilon decay rate: ", eps_decay)
    print("Batch size: ", batch_size)
    print()
    print("Use saved: ", use_saved)
    print("Save model: ", test_mode)
    print("\n")