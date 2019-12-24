import simpy

def make(num_elevators, num_floors):
    '''Generate new simpy.Environment.'''
    pass

class Environment():

    def __init__(self, simul_env, num_elevators, num_floors):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = num_floors


        self.action_space = None
        self.observation_space = None

    def reset(self):
        '''Resets the environment to its initial state,
           Returning the initial observation vector.
        '''
        pass

    def step(self):
        '''Receive an action from the agent
           and return the information about the outcome
           of the action:
           - next observation
           - local reward
           - end of episode flag (if ended)

           1. Tell environment which action will be taken by agent(s)
           2. Get new observation from the action(s)
           3. Get reward for the action
        '''
        pass

    def render(self):
        '''Render visualization for the environment.'''
        pass

    # Simulation Functions below
    def generate_passengers(self):
        '''Generate random passengers.
           
        This function will run as a simpy process:
        Ex: self.simul_env.process(self.generate_passengers())


        '''
        
        while True:
            delay_time = None # FIXME: set delay time
            yield self.simul_env.timeout(delay_time)

            # Create new instance of Passenger
            # Add Passenger to appropriate floor group (This may imply that
            # the RL model's observation may include the # of people on each floor
            # which is not tru in the real world. In real world, it is unknown to the agents
            # about how many people may be on each floor (Ex: passengers took stairs.)

    
