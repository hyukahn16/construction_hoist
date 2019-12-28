from env import Environment, make
import gym
print("starting train.py")

if __name__ == "__main__":
    print("Simulation starting.")
    
    env = make(1, 3) # env is Environment class
    env.reset()
    proc_pass_gen = env.simul_env.process(env.generate_passengers())
    
    env.simul_env.run(until=10000)
    
    print("Simulation finished.")