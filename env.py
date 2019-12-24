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
        '''Render visualization for the environment."""
        pass
    
