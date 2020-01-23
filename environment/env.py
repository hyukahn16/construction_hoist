import simpy
import random
import logging
import numpy as np
import torch
from enum import Enum
from .passenger import Passenger
from .elevator import Elevator

#Request = Enum(EMPTY=0, UP=1, DOWN=2)

def make(num_elevators, curr_floors, total_floors, pas_gen_time):
    '''Generate new simpy.Environment.'''
    assert curr_floors <= total_floors

    simpy_env = simpy.Environment()
    env = Environment(simpy_env, num_elevators, curr_floors, total_floors, pas_gen_time)
    return env

class Environment():

    def __init__(self, simul_env, num_elevators, curr_floors, total_floors, pas_gen_time):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = curr_floors
        self.total_floors = total_floors
        self.pas_gen_time = pas_gen_time
        
        # These variables underneath will be initialized in "self.reset()"
        self.floors = {} # Key: floor number, value: list of Passenger objects
        self.epoch_events = {} # key: event name, value: simpy event, this is what gets triggered to stop the simulation
        self.elevators = [] # List of Elevator objects
        self.call_requests = [] # List of call requests for each floor | Ex: self.call_requests[0][0] == 1 or 0
        self.decision_elevators = [] # Contains the indicies of elevators that need next action

        self.action_space = 3 # idle, up, down
        self.observation_space = None

    def reset(self):
        '''Resets the environment to its initial state,
           Returning the initial observation vector.

           - includes the simpy process for generate_passengers()
        '''
        self.simul_env = simpy.Environment()

        # initialize each floor that holds Passenger objects
        for i in range(self.num_floors):
            self.floors[i] = []
            self.call_requests.append([0, 0]) # first value is UP call, second value is DOWN call

        # Initialize Elevator objects
        for i in range(self.num_elevators):
            self.elevators.append(Elevator(self, i, 0))

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
        logging.debug("env.py: step() - Starting")
        # Create processes for each elevators' actions
        for idx, action in enumerate(actions):
            if action == -1:
                continue
            logging.debug("Creating proces for Elevator_{} with action {}".format(idx, action))
            self.simul_env.process(self.elevators[idx].act(action))

        while True: # run until a decision epoch is reached
            self.decision_elevators = []
            logging.debug("env.py: step() - Running simulation")
            finished_events = self.simul_env.run( \
                until=simpy.events.AnyOf(self.simul_env, self.epoch_events.values())).events
            logging.debug("env.py: step() - Finished simulation")

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
                else:
                    logging.debug("Unimplemented event type!")
            
            if decision_reached:
                break

        # return state, reward, and the decision agents
        output = {
            "state": np.array(self.get_state()),
            "reward": np.array(self.get_reward()), # List of rewards achieved by the previous action
            "decision_agents": np.array(self.decision_elevators) # List of elevators that need next action
        }
        logging.debug("env.py: step() - Finished")
        return output

    def generate_passengers(self):
        '''Generate random passengers on a random floor.
           
        This function will run as a simpy process:
        Ex: self.simul_env.process(self.generate_passengers())
        '''

        while True:
            yield self.simul_env.timeout(self.pas_gen_time)

            curr_fl = random.randrange(0, self.num_floors, 1) # get new current floor for this passenger
            # get new destination floor for this passenger
            # make sure that the destination floor is NOT the same as 
            # current floor
            dest_fl = curr_fl;
            while dest_fl == curr_fl:
                dest_fl = random.randrange(0, self.num_floors, 1)

            # Create new instance of Passenger at random floor
            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            
            if curr_fl > dest_fl:
                # Passenger needs to go DOWN
                self.call_requests[p.curr_floor][1] = 1
            else:
                # Passenger needs to go UP
                self.call_requests[p.curr_floor][0] = 1

            logging.debug("Created new Passenger at {}, going to {}!".format(p.curr_floor, p.dest_floor))

            # Trigger epoch event for PassengerRequest
            self.trigger_epoch_event("PassengerRequest")

    def _process_elevator_arrival(self, event_type):
        '''Process when an elevator stops at any floor.

        Elevator needs to decide on the next action.
        '''
        logging.debug("env.py: _process_elevator_arrival - {}".format(event_type))
        elevator_idx = int(event_type.split('_')[-1])
        self.decision_elevators.append(elevator_idx)

    def _process_loading_finished(self, event_type):
        '''Process when an elevator has finished loading or unloading.
        FIXME: this function is not used 
        LoadingFinished requires the next decision for the elevators.
        '''
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

        return False # FIXME: shouldn't PassengerRequest ask for next action of the elevators?

    def trigger_epoch_event(self, event_type):
        '''Used by other functions when the epoch events should be triggered.'''
        logging.debug("env.py: trigger_epoch_event() - {}".format(event_type))
        # Trigger the event
        self.epoch_events[event_type].succeed(event_type)
        # Reset the event to be triggered again in the future
        self.epoch_events[event_type] = self.simul_env.event()

    def load_passengers(self, elv_id):
        '''Use by Elevator when idle and ready to load/unload.'''
        logging.debug("env.py: load_passengers() - Elevator_{}".format(elv_id))
        carrying = self.elevators[elv_id].passengers
        curr_floor = self.elevators[elv_id].curr_floor

        # Save the Passengers that should get off on this floor
        to_delete = []
        for p in carrying:
            # Determine if the passenger should get off on the current floor
            if p.dest_floor == curr_floor:
                logging.debug("env.py: load_passengers() - passenger unloaded.")
                to_delete.append(p)

        # Unload Passengers
        reward = 0
        for p in to_delete:
            # Update reward for this elevator
            self.elevators[elv_id].update_reward(p.begin_wait_time - self.simul_env.now)
            reward += (p.begin_wait_time - self.simul_env.now)
            # Remove the passenger from the Elevator
            carrying.remove(p)        

        self.elevators[elv_id].last_reward = reward
        
        # Load passengers
        for p in self.floors[curr_floor]:
            logging.debug("env.py: load_passengers() "
                "- passenger loaded in Elevator_{} at floor {} "
                "going to floor {}.".format(elv_id, curr_floor, p.dest_floor))
            # FIXME: need to consider which passengers will be getting on the elevator
            # take passenger only if Elevator is NOT full
            if (len(self.elevators[elv_id].passengers) + 1) * 62 < \
                self.elevators[elv_id].weight_capacity:
                carrying.add(p)
                self.floors[curr_floor].remove(p)
                self.elevators[elv_id].requests[p.dest_floor] = 1
            else:
                break
        
        # Handle request calls from Environment and Elevator
        # 1. Handle Environment's call requests
        self.call_requests[curr_floor] = [0, 0] # reset call request for this floor
        for p in self.floors[curr_floor]:
            if p.dest_floor > curr_floor:
                self.call_requests[curr_floor][0] = 1
            elif p.dest_floor < curr_floor:
                self.call_requests[curr_floor][1] = 1
        # 2. Handle Elevator's call requests for this floor
        self.elevators[elv_id].requests[curr_floor] = 0

    def get_state(self):
        '''Return the state as a (total_floors x 4 x 1) image'''
        
        img = []
        '''
        for fl in range(self.num_floors):
            img.append([]) # img[i]
            for j in range(4):
                img[fl].append([self.call_requests[fl][0]])
                img[fl].append([self.call_requests[fl][1]])
                img[fl].append([self.elevators[0].curr_floor])
                img[fl].append([self.elevators[1].curr_floor])
        '''
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

    def get_reward(self):
        '''Return the last reward from all Elevators in a list.'''
        return [e.last_reward for e in self.elevators]

    def get_total_reward(self):
        return [e.reward for e in self.elevators]

    def update_all_reward(self):
        '''Calculate and update the reward for each elevator.
        
        Used in self.step()
        FIXME: NOT IMPLEMENTED YET
        '''
        for e in self.elevators:
            e.update_reward()

    def update_end_reward(self):
        '''Update Elevator rewards on Passengers never picked up by the Elevators
        in the episode.
        '''
        reward = 0
        for floor in self.floors:
            for p in self.floors[floor]:
                reward += (p.begin_wait_time - self.now())
        
        for e in self.elevators:
            e.update_reward(reward)

        return reward

    def now(self):
        return self.simul_env.now

    def render(self):
        '''Render visualization for the environment.'''
        pass
    
