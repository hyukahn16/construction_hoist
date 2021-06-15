import simpy
import random
import logging
import numpy as np
import copy
from copy import deepcopy
from gym import spaces
import gym
from collections import deque
from .passenger import Passenger
from .elevator import Elevator

class Environment(gym.Env):
    def __init__(self, simul_env, num_elevators, curr_floors, total_floors,
        pas_gen_time, action_space, total_time):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = curr_floors
        self.total_floors = total_floors
        self.pas_gen_time = pas_gen_time
        self.total_time = total_time
        
        self.action_space = spaces.Discrete(action_space ** self.num_elevators)

        obs_hall = 2 * total_floors
        obs_elev = 2 * total_floors * self.num_elevators
        self.observation_space = spaces.Box(low=0, high=1, 
            shape=(obs_hall + obs_elev,), dtype=np.int8)

    def reset(self):
        '''Resets the environment to its initial state,
           Returning the initial observation vector.

           - includes the simpy process for generate_passengers()
        '''
        self.total_wait_time = 0
        self.total_wait_passengers = 0

        self.simul_env = simpy.Environment()
        self.floors_up = {} # Key: floor number, value: list of Passenger objects
        self.floors_down = {}
        self.epoch_events = {} # key: event name, value: simpy event, this is what gets triggered to stop the simulation
        self.elevators = [] # List of Elevator objects
        self.call_requests = [] # List of call requests for each floor | Ex: self.call_requests[0][0] == 1 or 0
        self.decision_elevators = [] # Contains the indicies of elevators that need next action
        self.generated_passengers = 0

        # initialize each floor that holds Passenger objects
        for i in range(self.num_floors):
                self.floors_up[i] = deque()
                self.floors_down[i] = deque()
                # first value is UP call, second value is DOWN call
                # 0 = no call, 1 = call requested
                self.call_requests.append([0, 0])

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

        # Create a process that will generate passengers on random floors
        self.simul_env.process(self.generate_passengers())

        # reset returns the initial observation
        obs, _, _, _, _ = self.step(-1)
        return obs

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
           4. Return the above output
        '''
        # Create processes for each elevators' actions
        # If input action is -1, then no action process is created
        if isinstance(actions, list):
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
                    print("env.py-step(): Unimplemented event type!")
            
            if decision_reached:
                break

        # return state, reward, and the decision agents
        num_served = 0    
        for e_id in range(len(self.elevators)):
            num_served += self.elevators[e_id].num_served
        done = True if self.now() > self.total_time else False
        output = (
            np.array(self.get_elevator_state(), dtype=bool),
            [self.get_elevator_reward(e_id) for e_id in range(self.num_elevators)],
            done,
            {
                "lift_time": self.get_elevator_lift_time(0),
                "wait_time": self.get_elevator_wait_time(),
                "num_served": num_served,
                "generated_passengers": deepcopy(self.generated_passengers),
            },
            self.decision_elevators
        )

        # Reset environment during learning
        if done:
            #FIXME: self.reset()
            pass

        return output

    def generate_passengers(self):
        '''Generate random passengers on a random floor.
           
        This function will run as a simpy process:
        Ex: self.simul_env.process(self.generate_passengers())
        '''
        while True:
            # create new origin floors
            curr_fl = random.sample(range(0, self.num_floors), 1)
            # get new destination floors for passengers
            # where curr_fl != dest_fl
            dest_fl = copy.deepcopy(curr_fl)
            for i in range(len(dest_fl)):
                while dest_fl[i] == curr_fl[i]:
                    dest_fl[i] = random.randrange(0, self.num_floors, 1)

            # Create new passengers
            passengers = []
            for i in range(len(dest_fl)):
                passengers.append(Passenger(curr_fl[i], dest_fl[i], self.simul_env.now))
            
            # Add Passenger to appropriate floor group
            # and update the hall calls
            for p in passengers:
                if p.curr_floor > p.dest_floor: # DOWN call
                    self.floors_down[p.curr_floor].append(p)
                    self.call_requests[p.curr_floor][1] = 1
                else: # UP call
                    self.floors_up[p.curr_floor].append(p)
                    self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += len(passengers)
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

        1. Wake up potentially idling elevators
        2. Updates the state to reflect this request
        '''
        for e in self.elevators:
            # if elevator is idle, then wake it up
            if e.state == e.LOAD:
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

    def unload_passengers(self, e_id):
        carrying = self.elevators[e_id].passengers
        curr_floor = self.elevators[e_id].curr_floor

        # Save the Passengers that should get off on this floor
        unload_p = []
        for p in carrying:
            # Determine if the passenger should get off on the current floor
            if p.dest_floor == curr_floor:
                unload_p.append(p)
        # Unload Passengers from the elevator
        for p in unload_p:
            # Calculate time porportional reward
            #delivery_time = self.now() - p.begin_wait_time
            #rew = self.get_reward_prop_time(1e6, delivery_time)
            #scaled_rew = self.scale(0, 1e4, 100, 200, rew)
            #self.elevators[e_id].update_reward(scaled_rew)
            self.elevators[e_id].update_reward(5)
            self.elevators[e_id].num_served += 1

            # Remove the passenger from the Elevator
            p.elevator = -1 # extra caution check
            carrying.remove(p)

    def load_passengers(self, e_id):
        carrying = self.elevators[e_id].passengers
        curr_floor = self.elevators[e_id].curr_floor
        move_dir = self.elevators[e_id].direction
        hall_passengers = None
        full = True

        assert(move_dir == 1 or move_dir == -1)
        if move_dir == 1:
            hall_passengers = self.floors_up[curr_floor]
        else:
            hall_passengers = self.floors_down[curr_floor]

        # Load passengers into the elevator
        while not self.elevators[e_id].full() and hall_passengers:
            p = hall_passengers.popleft()
            carrying.add(p)
            full = False
            # Update information about this passenger
            p.begin_lift_time = self.now() # Start lift time
            p.elevator = e_id
            # update reward for pickking up new passenger
            #pickup_time = self.now() - p.begin_wait_time
            #rew = self.get_reward_prop_time(3e5, pickup_time)
            #self.elevators[e_id].update_reward(rew)
            self.elevators[e_id].update_reward(2)
            self.elevators[e_id].requests[p.dest_floor] = 1
        
        #if full:
        #    self.elevators[e_id].update_reward(-100)
    
    def move_full_penalty(self, e_id):
        cap = self.elevators[e_id].weight_capacity
        if len(self.elevators[e_id].passengers) + 1 >= (cap / 62):
            self.elevators[e_id].update_reward(-100)

    def update_req_calls(self, e_id):
        '''Called by Elevator after load and unload functions.
            Update requests from inside Elevator.
            Update calls from the Environment.
        '''
        curr_fl = self.elevators[e_id].curr_floor
        move_dir = self.elevators[e_id].direction
        # 1. Handle Environment's calls
        self.call_requests[curr_fl] = [0, 0] # reset call request for this floor
        if self.floors_up[curr_fl]:
            self.call_requests[curr_fl][0] = 1
        if self.floors_down[curr_fl]:
            self.call_requests[curr_fl][1] = 1

        # 2. Handle Elevator's requests for this floor
        self.elevators[e_id].requests[curr_fl] = 0

    def move_rew_request(self, e_id, move):
        # Reward for moving in the REQUEST direction
        # for passengers inside the Elevator
        # This function is called BEFORE the move happens

        curr_floor = self.elevators[e_id].curr_floor
        total_req_rew = 0
        # 
        for p in self.elevators[e_id].passengers:
            dist = abs(p.dest_floor - curr_floor)

            # if elevator is moving past the destination
            if dist == 0: 
                dist = 1
            req_reward = 20 / dist

            # If moving up but the destination is below
            if move == 1 and p.dest_floor <= curr_floor:
                req_reward *= -1
            # If moving down but the destination is above
            elif move == -1 and p.dest_floor >= curr_floor:
                req_reward *= -1

            total_req_rew += req_reward
        self.elevators[e_id].update_reward(total_req_rew)

    def move_rew_call(self, e_id, move):
        # Reward function for moving in the CALL direction
        # for passengers waiting for the Elevator.
        # This function is called BEFORE the move happens.

        curr_floor = self.elevators[e_id].curr_floor
        calls_above = 0
        calls_below = 0
        # Count the number of calls above and below
        for f in range(self.total_floors):
            if f < curr_floor:
                calls_below += len(self.floors_down[f])
                calls_below += len(self.floors_up[f])
            else:
                calls_above += len(self.floors_up[f])
                calls_above += len(self.floors_down[f])
        # Give different rewards depending on the move
        if move == 0:
            calls_above = 0
            calls_below = 0
            #reward_calls_above = reward_calls_above * -1
            #reward_calls_below = reward_calls_below * -1
        elif move == 1:
            calls_below *= -1
        elif move == -1:
            calls_above *= -1
        self.elevators[e_id].update_reward(
            calls_above + calls_below
        )
        
    def get_elevator_state(self):
        '''Used in step function'''
        e_state = []
        # Up calls from the building
        for call in self.call_requests:
            e_state.append(call[0])
        
        # Down calls from the building
        for call in self.call_requests:
            e_state.append(call[1])

        # Calls from within the Elevator
        for elev in range(len(self.elevators)):
            for request in self.elevators[elev].requests:
                e_state.append(request)

            # Location of the Elevator
            for f_num in range(self.total_floors):
                if self.elevators[elev].curr_floor == f_num:
                    e_state.append(1)
                else:
                    e_state.append(0)
                
        return e_state

    def get_elevator_reward(self, e_id):
        '''Used at the end of the step function
        to return the reward for a specific elevator.'''
        output = self.elevators[e_id].reward
        self.elevators[e_id].reward = 0
        return output

    def get_reward_prop_time(self, flat_reward, p_time):
        time = self.now() - p_time
        if time == 0:
            return flat_reward
        return flat_reward / time

    def scale(self, old_min, old_max, new_min, new_max, val):
        numer = (new_max - new_min) * (val - old_min)
        denom = old_max - old_min
        scaled_val = (numer / denom) + new_min
        if scaled_val > new_max:
            scaled_val = new_max
        return scaled_val

    def get_elevator_lift_time(self, e_id):
        e = self.elevators[e_id]
        return e.calculate_avg_lift_time()

    def get_elevator_wait_time(self):
        '''Calculate average wait time.'''
        for _, floor in self.floors_down.items():
            for p in floor: 
                self.update_wait_time(p)
        for _, floor in self.floors_up.items():
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
        DIR_MAP = {
            self.elevators[0].IDLE: '-',
            None: '@',
            self.elevators[0].MOVING_UP: '^',
            self.elevators[0].MOVING_DOWN: 'v',
            self.elevators[0].LOAD: 'x',
        }

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

        print("\n")