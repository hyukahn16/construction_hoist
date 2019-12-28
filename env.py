# TODO: agent sees how many people on each floor have called for the button

import simpy
import random
from passenger import Passenger

def make(num_elevators, num_floors):
    '''Generate new simpy.Environment.'''
    simpy_env = simpy.Environment()
    env = Environment(simpy_env, num_elevators, num_floors)
    return env

class Environment():

    def __init__(self, simul_env, num_elevators, num_floors):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = num_floors
        
        # these variables underneath will be initialized in "self.reset()"
        self.floors = {} # each floor has list of passengers waiting an elevator
        self.epoch_events = {} # key: event name, value: simpy event


        self.action_space = None
        self.observation_space = None


    def reset(self):
        '''Resets the environment to its initial state,
           Returning the initial observation vector.

           - includes the simpy process for generate_passengers()
        '''
        # initialize each floor
        for i in range(self.num_floors):
            self.floors[i] = []

        # Initialize epoch_events dictionary (which event should the simulation stop?)
        # 1. When elevator arrives at a floor
        # 2. when passenger requests elevator
        # 3. when passenger arrives at destination
        # FIXME: some epoch_events creation might not be in right for-loop
        for i in range(self.num_elevators):
            self.epoch_events["ElevatorArrival_{}".format(i)] = self.simul_env.event()
            self.epoch_events["PassengerRequest_{}".format(i)] = self.simul_env.event()
            self.epoch_events["PassengerArrival_{}".format(i)] = self.simul_env.event()
        for i in range(self.num_floors):
            pass

        # Initialize observation space
        """
        Observation space (what the agent will see to make decisions):
        - which floor the elevator request came from
        - which direction (up or down) the request wants
        """
        self.req_floor_up = [] # req_floor_down[i] == 1 if ith floor requested elevator going up
        self.req_floor_down = []

        self.simul_env = simpy.Environment()
        print("Resetting!")

    def step(self, action):
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
        
        while True: # run until a decision epoch is reached
            finished_events = self.simul_env.run(until=self.epoch_events.values()).events


    def render(self):
        '''Render visualization for the environment.'''
        pass

    # Simulation Functions below
    def generate_passengers(self):
        '''Generate random passengers on a random floor.
           
        This function will run as a simpy process:
        Ex: self.simul_env.process(self.generate_passengers())
        '''
        
        print("Generating new passengers")
        while True:
            delay_time = 100 # FIXME: set delay time
            yield self.simul_env.timeout(delay_time)

            # Create new instance of Passenger at random floor
            curr_fl = random.randrange(0, self.num_floors, 1) # get new current floor for this passenger
            print(curr_fl)
            # get new destination floor for this passenger
            dest_fl = curr_fl;
            while dest_fl == curr_fl:
                dest_fl = random.randrange(0, self.num_floors, 1)
            print(dest_fl)
            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            print(self.floors)
            self.floors[p.curr_floor].append(p)
            print("Created new Passenger at {}, going to {}!".format(p.curr_floor, p.dest_floor))

    
