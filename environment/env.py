import simpy
import random
import logging
import numpy as np
import torch
from .passenger import Passenger
from .elevator import Elevator
import environment

def make(num_elevators, curr_floors, total_floors, pas_gen_time, episode_time,
    mode=None, human_mode=False, human_agent=None):
    '''Generate new simpy.Environment.'''
    assert curr_floors <= total_floors

    simpy_env = simpy.Environment()
    env = None
    if human_mode:
        if mode == "uppeak":
            return environment.HumanUpPeakEnvironment(simpy_env, num_elevators, 
                curr_floors, total_floors, pas_gen_time, human_agent, episode_time)
        elif mode == "downpeak":
            return environment.HumanDownPeakEnvironment(simpy_env, num_elevators, 
                curr_floors, total_floors, pas_gen_time, human_agent, episode_time)
        elif mode == "intermediate":
            return environment.HumanIntermediateEnvironment(simpy_env, num_elevators, 
                curr_floors, total_floors, pas_gen_time, human_agent, episode_time)
        elif mode == "lunch": 
            return environment.HumanLunchEnvironment(simpy_env, num_elevators, 
                curr_floors, total_floors, pas_gen_time, human_agent, episode_time)

    if not mode or "intermediate":
        env = Environment(simpy_env, num_elevators, curr_floors, total_floors, 
            pas_gen_time)
    elif mode == "schedule":
        env = environment.TestEnvironment(simpy_env, num_elevators, 
            curr_floors, total_floors, pas_gen_time, "schedule.txt")
    elif mode == "uppeak":
        env = environment.UpPeakEnvironment(simpy_env, num_elevators,
                            curr_floors, total_floors, pas_gen_time)
    elif mode == "downpeak":
        env = environment.DownPeakEnvironment(simpy_env, num_elevators,
                            curr_floors, total_floors, pas_gen_time)
    elif mode == "lunch":
        env = environment.LunchEnvironment(simpy_env, num_elevators,
                            curr_floors, total_floors, pas_gen_time, episode_time)
    return env


class Environment():

    def __init__(self, simul_env, num_elevators, curr_floors, total_floors, pas_gen_time):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = curr_floors
        self.total_floors = total_floors
        self.pas_gen_time = pas_gen_time
        
        self.action_space_size = 3 # idle, up, down
        self.observation_space_size = total_floors

    def reset(self):
        '''Resets the environment to its initial state,
           Returning the initial observation vector.

           - includes the simpy process for generate_passengers()
        '''
        self.total_wait_time = 0
        self.total_wait_passengers = 0

        self.simul_env = simpy.Environment()
        self.floors = {} # Key: floor number, value: list of Passenger objects
        self.epoch_events = {} # key: event name, value: simpy event, this is what gets triggered to stop the simulation
        self.elevators = [] # List of Elevator objects
        self.call_requests = [] # List of call requests for each floor | Ex: self.call_requests[0][0] == 1 or 0
        self.decision_elevators = [] # Contains the indicies of elevators that need next action
        self.generated_passengers = 0

        # initialize each floor that holds Passenger objects
        for i in range(self.num_floors):
            self.floors[i] = []
            self.call_requests.append([0, 0]) # first value is UP call, second value is DOWN call

        # Initialize Elevator objects
        for i in range(self.num_elevators):
            self.elevators.append(Elevator(self, i))

        # Initialize epoch_events dictionary
        # (which event should the simulation stop to figure out the next decision?)
        for i in range(self.num_elevators):
            # 1. When elevator arrives at a floor
            self.epoch_events["ElevatorArrival_{}".format(i)] = self.simul_env.event()
        # 2. when passenger requests elevator
        self.epoch_events["PassengerRequest"] = self.simul_env.event()

        # TODO: Initialize observation space
        # Observation space (what the agent will see to make decisions):
        # - Call requests at each floor
        # - Hoists' current capacity
        # - Hoists' positions

        # Create a process that will generate passengers on random floors
        self.simul_env.process(self.generate_passengers())
        logging.debug("env.py: Reset Complete!")
        return self.step([-1])

    def step(self, actions):
        '''Receive an action from the agent
           and return the information about the outcome
           of the action:
           - next observation
           - local reward
           - end of episode flag (if ended)

           1. Create processes for Elevators that have actions
           2. Run until decision epoch is reached
           3. if event type is ElevatorArrival or LoadingFinished, then finish function 
           4. 
        '''
        # Create processes for each elevators' actions
        for idx, action in enumerate(actions):
            if action == -1:
                continue
            self.simul_env.process(self.elevators[idx].act(action))

        while True: # run until a decision epoch is reached
            self.decision_elevators = []
            finished_events = self.simul_env.run( \
                until=simpy.events.AnyOf(self.simul_env, self.epoch_events.values())).events

            decision_reached = False
            for event in finished_events:
                if "ElevatorArrival" in event.value:
                    decision_reached = True
                    self._process_elevator_arrival(event.value)
                elif "PassengerRequest" in event.value:
                    self._process_passenger_request(event.value)
                else:
                    logging.debug("Unimplemented event type!")
            
            if decision_reached:
                break

        # return state, reward, and the decision agents
        output = {}
        for e_id in self.decision_elevators:
            output[e_id] = {"state": np.array(self.get_elevator_state(e_id)),
                            "reward": self.get_elevator_reward(e_id),
                            # FIXME: create multiple inheritance for 
                            # TestEnvironment for efficiency so that
                            # training Environment doesn't have to 
                            # calculate unused numbers
                            "lift_time": self.get_elevator_lift_time(e_id),
                            "wait_time": self.get_elevator_wait_time(),
                            }
        self.decision_elevators = []
        return output

    def generate_passengers(self):
        '''Generate random passengers on a random floor.
           
        This function will run as a simpy process:
        Ex: self.simul_env.process(self.generate_passengers())
        '''

        while True:
            curr_fl = random.randrange(0, self.num_floors, 1) # get new current floor for this passenger
            # get new destination floor for this passenger
            # make sure that the destination floor is NOT the same as 
            # current floor
            dest_fl = curr_fl
            while dest_fl == curr_fl:
                dest_fl = random.randrange(0, self.num_floors, 1)

            # Create new instance of Passenger at random floor
            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            logging.debug("Created new Passenger at {}, going to {}!".format(p.curr_floor, p.dest_floor))
            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")

            yield self.simul_env.timeout(self.pas_gen_time)

    def _process_elevator_arrival(self, event_type):
        '''Process when an elevator stops at any floor.

        Elevator needs to decide on the next action.
        '''
        logging.debug("env.py: _process_elevator_arrival - {}".format(event_type))
        elevator_idx = int(event_type.split('_')[-1])
        self.decision_elevators.append(elevator_idx)

    def _process_passenger_request(self, event_type):
        '''Process when a passenger requests for an elevator.

        - Wake up potentially idling elevators
        - Updates the state to reflect this request
        '''
        for e in self.elevators:
            # if elevator is idle, then wake it up
            if e.state == e.IDLE:
                e.interrupt_idling()

    def trigger_epoch_event(self, event_type):
        # Check spurious wakeup
        if "ElevatorArrival" in event_type:
            e_id = int(event_type.split('_')[-1])
            if self.elevators[e_id].state != None:
                return

        '''Used by other functions when the epoch events should be triggered.'''
        logging.debug("env.py: trigger_epoch_event() - {}".format(event_type))

        # Trigger the event
        self.epoch_events[event_type].succeed(event_type)
        # Reset the event to be triggered again in the future
        self.epoch_events[event_type] = self.simul_env.event()

    def load_passengers(self, e_id, move=0):
        '''Use by Elevator when idle and ready to load/unload.'''
        carrying = self.elevators[e_id].passengers
        curr_floor = self.elevators[e_id].curr_floor

        # Save the Passengers that should get off on this floor
        unload_p = []
        for p in carrying:
            # Determine if the passenger should get off on the current floor
            if p.dest_floor == curr_floor:
                unload_p.append(p)

        # Unload Passengers
        for p in unload_p:
            self.elevators[e_id].update_reward(100)
            self.elevators[e_id].num_served += 1
            # Remove the passenger from the Elevator
            p.elevator = None
            carrying.remove(p)        
            
        # Load passengers
        for p in self.floors[curr_floor]:
            # FIXME: need to consider which passengers will be getting on the elevator
            # take passenger only if Elevator is NOT full
            if (len(self.elevators[e_id].passengers) + 1) * 62 < \
                self.elevators[e_id].weight_capacity:
                carrying.add(p)
                p.begin_lift_time = self.now() # Start lift time
                self.floors[curr_floor].remove(p)
                p.elevator = e_id
                self.elevators[e_id].update_reward(50)
                self.elevators[e_id].requests[p.dest_floor] = 1
            else:
                break
        
        # Update request calls from Environment and Elevator
        # 1. Handle Environment's call requests
        self.call_requests[curr_floor] = [0, 0] # reset call request for this floor
        for p in self.floors[curr_floor]:
            if p.dest_floor > curr_floor: # UP call
                self.call_requests[curr_floor][0] = 1
            elif p.dest_floor < curr_floor: # DOWN call
                self.call_requests[curr_floor][1] = 1
        # 2. Handle Elevator's call requests for this floor
        self.elevators[e_id].requests[curr_floor] = 0

        # Reward for moving in the CALL direction
        reward_calls_above = 0
        reward_calls_below = 0
        for f in range(self.total_floors):
            if f < curr_floor:
                reward_calls_below += len(self.floors[f])
            else:
                reward_calls_above += len(self.floors[f])
        if move == 0:
            reward_calls_above = reward_calls_above * -1
            reward_calls_below = reward_calls_below * -1
        elif move == 1:
            reward_calls_below = reward_calls_below * -1
        elif move == -1:
            reward_calls_above = reward_calls_above * -1
        self.elevators[e_id].update_reward(
            reward_calls_above + reward_calls_below)
        
        # Reward for moving in the REQUEST direction
        for p in self.elevators[e_id].passengers:
            req_reward = 25 / abs(p.dest_floor - curr_floor)
            if move == 0:
                req_reward = req_reward * -1
            elif move == 1 and p.dest_floor < curr_floor:
                req_reward = req_reward * -1
            elif move == -1 and p.dest_floor > curr_floor:
                req_reward = req_reward * -1
            self.elevators[e_id].update_reward(req_reward)

    def get_elevator_state(self, e_id):
        e_state = []
        # Up calls from the building
        for call in self.call_requests:
            e_state.append(call[0])
        
        # Down calls from the building
        for call in self.call_requests:
            e_state.append(call[1])

        # Calls from within the Elevator
        for request in self.elevators[e_id].requests:
            e_state.append(request)

        # Location of the Elevator
        for f_num in range(self.total_floors):
            if self.elevators[e_id].curr_floor == f_num:
                e_state.append(1)
            else:
                e_state.append(0)
                
        return e_state

    def get_elevator_reward(self, e_id):
        '''Return the rewad from decision Elevators'''
        output = self.elevators[e_id].reward
        self.elevators[e_id].reward = 0
        return output

    def get_reward_prop_time(self, flat_reward, p_time):
        time = self.now() - p_time
        if time == 0:
            return flat_reward
        return flat_reward / time

    def get_elevator_lift_time(self, e_id):
        e = self.elevators[e_id]
        return e.calculate_avg_lift_time()

    def get_elevator_wait_time(self):
        '''Calculate average wait time.'''
        for _, floor in self.floors.items():
            for p in floor: 
                self.update_wait_time(p)
        
        if self.total_wait_passengers == 0:
            return 0
        avg_wait_time = self.total_wait_time / self.total_wait_passengers

        # Reset variables
        self.total_wait_passengers = 0
        self.total_wait_time = 0

        return avg_wait_time

    def update_wait_time(self, p):
        # Save passenger's wait time
        self.total_wait_passengers += 1
        self.total_wait_time += (self.now() - p.begin_wait_time)

    def now(self):
        return self.simul_env.now

    def render(self):
        '''
        Prints some stone age visualization in stdout...
        '''
        DIR_MAP = {self.elevators[0].IDLE: '-', None: '-',
                self.elevators[0].MOVING_UP: '^', self.elevators[0].MOVING_DOWN:'v'}
        for floor in range(self.total_floors):
            num_psngr_going_up = len([p for p in self.floors[floor] if p.dest_floor > floor])
            num_psngr_going_down = len([p for p in self.floors[floor] if p.dest_floor < floor])

            string = ""
            if floor < 10:
                string += "0" + str(floor)
            else:
                string += str(floor)

            for elevator in self.elevators:
                if elevator.curr_floor == floor:
                    string+="|{}{:>2}|".format(DIR_MAP[elevator.state], len(elevator.passengers))
                elif 0 > floor or self.total_floors < floor:
                    string+="     "
                else:
                    string+="|   |"
            string+="^"*num_psngr_going_up
            string+="v"*num_psngr_going_down
            print(string)

    # CNN state version
    def get_cnn_state(self):
        '''Return the state as a (total_floors x 4 x 1) image'''
        
        img = []
        img.append([])
        for fl in range(self.num_floors):
            img[0].append([])
            img[0][fl].append(self.call_requests[fl][0])
            img[0][fl].append(self.call_requests[fl][1]) 

            img[0][fl].append(1 if self.elevators[0].curr_floor == fl else 0) # Elevator floor location
            img[0][fl].append((len(self.elevators[0].passengers) * 62.0 / \
                self.elevators[0].weight_capacity) if self.elevators[0].curr_floor == fl else 0) 
            img[0][fl].append(self.elevators[0].requests[fl]) # floor requests from inside the Elevator

            img[0][fl].append(1 if self.elevators[1].curr_floor == fl else 0) # Elevator floor location
            img[0][fl].append((len(self.elevators[1].passengers) * 62.0 / \
                self.elevators[1].weight_capacity) if self.elevators[1].curr_floor == fl else 0)
            img[0][fl].append(self.elevators[1].requests[fl]) # floor requests from inside the Elevator
        return img