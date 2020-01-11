import simpy

def pass_gen(env):
    while True:
        yield env.timeout(500)
        print("Generating passenger")
    
env = simpy.Environment()
env.process(pass_gen(env))
env.run(until=env)
print("Finished test")

