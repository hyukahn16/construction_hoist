import simpy
import random
from enum import Enum
from passenger import Passenger
from elevator import Elevator

#Request = Enum(EMPTY=0, UP=1, DOWN=2)

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
        
        # These variables underneath will be initialized in "self.reset()"
        self.floors = {} # Key: floor number, value: list of Passenger objects
        self.epoch_events = {} # key: event name, value: simpy event, this is what gets triggered to stop the simulation
        self.elevators = [] # List of Elevator objects
        self.call_requests = [] # List of call requests for each floor - holds Request Enum types

        # FIXME These are used for the DQN I think?
        self.action_space = None # 2 for up or down
        self.observation_space = None


    def reset(self):
        '''Resets the environment to its initial state,
           Returning the initial observation vector.

           - includes the simpy process for generate_passengers()
        '''
        # initialize each floor that holds Passenger objects
        for i in range(self.num_floors):
            self.floors[i] = []

        # Initialize Elevator objects
        for _ in range(self.num_elevators):
            self.elevators.append(Elevator(0))

        # Initialize epoch_events dictionary
        # (which event should the simulation stop to figure out the next decision?)
        for i in range(self.num_elevators):
            # 1. When elevator arrives at a floor
            self.epoch_events["ElevatorArrival_{}".format(i)] = self.simul_env.event()
            # 2. when passenger arrives at destination (similar to #1)
            self.epoch_events["LoadingFinished_{}".format(i)] = self.simul_env.event()
        # 3. when passenger requests elevator
        self.epoch_events["PassengerRequest"] = self.simul_env.event()

        # TODO: Initialize observation space
        # Observation space (what the agent will see to make decisions):
        # - Call requests at each floor
        # - Hoists' current capacity
        # - Hoists' positions

        # Create a process that will generate passengers on random floors
        self.simul_env.process(self.generate_passengers())
        self.simul_env = simpy.Environment()
        print("Reset Complete!")

    def step(self, actions):
        '''Receive an action from the agent
           and return the information about the outcome
           of the action:
           - next observation
           - local reward
           - end of episode flag (if ended)

           1. Create processes for Elevators that have actions
           2. Run until decision epoch is reached
           3. if event type is ElevatorArrival or LoadingFinished, then 
           2. Get new observation from the action(s)
           3. Get reward for the action
        '''

        # Create processes for each elevators' actions
        for action in actions:
            self.simul_env.process(self.elevators.act(action))

        while True: # run until a decision epoch is reached
            finished_events = self.simul_env.run(until=AnyOf(self.simul_env, self.epoch_events.values())).events
            
            decision_reached = False
            for event in finished_events:
                if "ElevatorArrival" in event.value:
                    decision_reached = True
                    self._process_elevator_arrival(event.value)
                elif "LoadingFinished" in event.value:
                    decision_reached = True
                    self._process_loading_finished(event.value)
                elif "PassengerRequest" in event.value:
                    self._process_passenger_request(event.value)

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
            # get new destination floor for this passenger
            dest_fl = curr_fl;
            while dest_fl == curr_fl:
                dest_fl = random.randrange(0, self.num_floors, 1)

            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            print("Created new Passenger at {}, going to {}!".format(p.curr_floor, p.dest_floor))

    def _process_elevator_arrival(self, event_type):
        '''Process when an elevator stops at any floor.'''
        return True

    def _process_loading_finished(self, event_type):
        '''Process when an elevator has finished loading or unloading.
        
        Returns True because 
        LoadingFinished requires the next decision for the elevators.
        '''
        elevator_idx = int(event_type.split('_')[-1])
        self.decision_elevators(elevator_idx)
        return True

    def _process_passenger_request(self, event_type):
        '''Process when a passenger requests for an elevator.

        - Updates the state to reflect this request
        '''
        return False # FIXME: shouldn't PassengerRequest ask for next action of the elevators?

    def get_state(self):
        '''Return the state in multi-dimensional array.

        Returns:
        - Hoist call requests (Up, down) (FIXME potentially the floor number as well)
        - Hoists' positions (34F, 64F)
        - Hoists' current weight WHEN they got a call request (1/1.5)
        '''
        pass

    def get_reward(self):
        '''Calculate and return the reward for the elevators
        
        Used in self.step()
        '''
        pass

    def render(self):
        '''Render visualization for the environment.'''
        pass
    
