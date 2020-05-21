'''
Different testing environments
'''

from environment import Environment
from .passenger import Passenger
import queue
import random

# Reads a schedule from a file and creates passengers based on the schedule
class ScheduleEnvironment(Environment):
    def __init__(self, simul_env, num_elevators, curr_floors, 
                total_floors, pas_gen_time, schedule_file):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = curr_floors
        self.total_floors = total_floors
        self.pas_gen_time = pas_gen_time
        
        self.action_space_size = 3 # idle, up, down
        self.observation_space_size = total_floors

        self.schedule_file = schedule_file
        self.passenger_schedule = queue.Queue()
        self.read_schedule_from_file()

    def read_schedule_from_file(self):
        ''' Read the schedules from file.

            Requires: self.filename
            Modifies: self.passenger_schedule will store a list of list of current floor and
                        destination floor
                        Ex: self.passenger_schedule[0] == [1, 10]
            Output: void
        '''

        with open(self.schedule_file) as schedule:
            # Omit 1st line
            schedule.readline()

            # Start reading schedule of passengers from 2nd line
            for line in schedule:
                p_data = line.split(',')
                curr_fl = p_data[0]
                dest_fl = p_data[1]
                self.passenger_schedule.put([int(curr_fl), int(dest_fl)])

    def generate_passengers(self):
        while True:
            # Create new instance of Passenger
            p_data = self.passenger_schedule.get()
            curr_fl = p_data[0]
            dest_fl = p_data[1]
            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            
            # Update Environment call requests
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")

            yield self.simul_env.timeout(self.pas_gen_time)

# All passengers spawn from the lowest floor and 
# have randomly assigned destination floor other than the lowest floor.
class UpPeakEnvironment(Environment):
    def generate_passengers(self):
        while True:
            curr_fl = 0
            dest_fl = random.randrange(1, self.num_floors, 1)

            # Create new instance of Passenger
            p = Passenger(curr_fl, dest_fl, self.now())
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")
            yield self.simul_env.timeout(self.pas_gen_time)

# All passengers spawn from floors except for the lowest floor and
# have assigned destination floor of the lowest floor.
class DownPeakEnvironment(Environment):
    def generate_passengers(self):
        while True:
            curr_fl = random.randrange(1, self.num_floors, 1)
            dest_fl = 0

            # Create new instance of Passenger at random floor
            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")

            yield self.simul_env.timeout(self.pas_gen_time)

    pass

# Combination of UpPeakEnvironment and DownPeakEnvironment
class LunchEnvironment(Environment):
    def __init__(self, simul_env, num_elevators, curr_floors, 
                total_floors, pas_gen_time, episode_time):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = curr_floors
        self.total_floors = total_floors
        self.pas_gen_time = pas_gen_time
        self.episode_time = episode_time
        
        self.action_space_size = 3 # idle, up, down
        self.observation_space_size = total_floors

    def generate_passengers(self):
        # DownPeak
        while self.now() < self.episode_time / 2:
            curr_fl = random.randrange(1, self.num_floors, 1)
            dest_fl = 0

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
        
        # UpPeak
        while self.now() >= self.episode_time / 2:
            curr_fl = 0
            dest_fl = random.randrange(1, self.num_floors, 1)

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

class HumanEnvironment(Environment):
    def __init__(self, simul_env, num_elevators, curr_floors, 
                total_floors, pas_gen_time, human_agent, episode_time):
        self.simul_env = simul_env
        self.num_elevators = num_elevators
        self.num_floors = curr_floors
        self.total_floors = total_floors
        self.pas_gen_time = pas_gen_time
        self.episode_time = episode_time
        
        self.action_space_size = 3 # idle, up, down
        self.observation_space_size = total_floors

        self.human_agent = human_agent

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
            p.elevator = -1
            carrying.remove(p)        
            
        # Load passengers
        for p in self.floors[curr_floor]:
            if p == self.human_agent.serving_passenger or \
                (len(self.elevators[e_id].passengers) + 2) * 62 < \
                self.elevators[e_id].weight_capacity:
                carrying.add(p)
                p.begin_lift_time = self.now() # Start lift timer
                self.floors[curr_floor].remove(p)
                p.elevator = e_id
                self.elevators[e_id].update_reward(50)
                self.elevators[e_id].requests[p.dest_floor] = 1
                break
        
        # Update request calls from Environment and Elevator
        # 1. Handle Environment's call requests for the current floor
        self.call_requests[curr_floor] = [0, 0] # reset call request for this floor
        for p in self.floors[curr_floor]:
            if p.dest_floor > curr_floor: # UP call
                self.call_requests[curr_floor][0] = 1
            elif p.dest_floor < curr_floor: # DOWN call
                self.call_requests[curr_floor][1] = 1
        # 2. Handle Elevator's call requests for the current floor
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

class HumanUpPeakEnvironment(HumanEnvironment):
    # https://www.programiz.com/python-programming/methods/built-in/classmethod

    def generate_passengers(self):
        '''Creates passenger instances until environment finishes.'''
        while True:
            curr_fl = 0
            dest_fl = random.randrange(1, self.num_floors, 1)

            p = Passenger(curr_fl, dest_fl, self.now())
            self.floors[p.curr_floor].append(p)
            self.human_agent.passenger_queue.put(p)
            
            # Update the calls based on this passenger
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")
            yield self.simul_env.timeout(self.pas_gen_time)

class HumanDownPeakEnvironment(HumanEnvironment):
    def generate_passengers(self):
        while True:
            dest_fl = 0
            curr_fl = random.randrange(1, self.num_floors, 1)

            p = Passenger(curr_fl, dest_fl, self.now())
            self.floors[p.curr_floor].append(p)
            self.human_agent.passenger_queue.put(p)
            
            # Update the calls based on this passenger
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")
            yield self.simul_env.timeout(self.pas_gen_time)

class HumanIntermediateEnvironment(HumanEnvironment):
    def generate_passengers(self):
        '''Creates passenger instances until environment finishes.'''
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
            self.human_agent.passenger_queue.put(p)
            
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")

            yield self.simul_env.timeout(self.pas_gen_time)

class HumanLunchEnvironment(HumanEnvironment):
    def generate_passengers(self):
        # DownPeak
        while self.now() < self.episode_time / 2:
            curr_fl = random.randrange(1, self.num_floors, 1)
            dest_fl = 0

            # Create new instance of Passenger at random floor
            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            self.human_agent.passenger_queue.put(p)
            
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")

            yield self.simul_env.timeout(self.pas_gen_time)
        
        # UpPeak
        while self.now() >= self.episode_time / 2:
            curr_fl = 0
            dest_fl = random.randrange(1, self.num_floors, 1)

            # Create new instance of Passenger at random floor
            p = Passenger(curr_fl, dest_fl, self.simul_env.now)
            
            # Add Passenger to appropriate floor group
            self.floors[p.curr_floor].append(p)
            self.human_agent.passenger_queue.put(p)
            
            if curr_fl > dest_fl: # DOWN call
                self.call_requests[p.curr_floor][1] = 1
            else: # UP call
                self.call_requests[p.curr_floor][0] = 1

            self.generated_passengers += 1
            self.trigger_epoch_event("PassengerRequest")

            yield self.simul_env.timeout(self.pas_gen_time)