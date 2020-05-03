'''
Different testing environments
'''

from environment import Environment
from .passenger import Passenger
import queue

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

            logging.debug("Created new Passenger at {}, going to {}!".format(p.curr_floor, p.dest_floor))
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