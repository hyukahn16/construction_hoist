from env import Environment, make

if __name__ == "__main__":
    print("Simulation starting.")
    env = make(1, 2) # env is Environment class
    proc_pass_gen = env.simul_env.process(env.generate_passengers())
    proc_reset = env.simul_env.process(env.reset())
    env.simul_env.run(until=10000)
    print("Simulation finished.")
