# 0. Agent type
# 1. Total reward 
# 2. Number of passengers served
# 3. Number of passengers in the elevator
# 4. Number of actions
# 5. lift time
# 6. wait time
# 7. total time
import sys
import numpy as np

counter = 0
filename = sys.argv[1]
agent_type = ""
total_reward = 0
pass_served = 0
pass_loaded = 0
actions = []
total_time = []
with open(filename) as file:
    for line in file:
        if counter == 0:
            if agent_type != "":
                # Print stats
                print("Agent type: " + agent_type, end="")
                print("Total reward: " + total_reward, end="")
                print("Number of passengers served: " + pass_served, end="")
                print("Number of passengers loaded: " + pass_loaded, end="")
                print("Average total time: ", np.average(total_time))
                print()
            agent_type = line
        elif counter == 1:
            total_reward = line
        elif counter == 2:
            pass_served = line
        elif counter == 3:
            pass_loaded = line
        elif counter == 4:
            actions = line.split()
        elif counter == 5:
            pass
        elif counter == 6:
            pass
        elif counter == 7:
            total_time = [float(num) for num in line.split()]
        counter = (counter + 1) % 8
        
    if agent_type != "":
        print("Agent type: " + agent_type, end="")
        print("Total reward: " + total_reward, end="")
        print("Number of passengers served: " + pass_served, end="")
        print("Number of passengers loaded: " + pass_loaded, end="")
        print("Average total time: ", np.average(total_time))
        print()
                