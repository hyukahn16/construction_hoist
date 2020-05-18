'''
Heuristic algorithm for Elevator which follows basic human-like control
1. Maintain a queue of service requests from passengers
2. Deliver passengers to their destination based on the queue
'''

import queue

class HumanAgent():
    def __init__(self, total_floors):
        self.passenger_queue = queue.Queue() # holds Passenger objects
        self.serving_passenger = None
        self.observation_space = total_floors

    def action(self, state):
        '''Choose action based on where the serving passenger is 
        and where their destination floor.
        '''
        while True:
            # Get new passenger to serve
            if not self.serving_passenger:
                if self.passenger_queue.empty():
                    return 0
                self.serving_passenger = self.passenger_queue.get()

            # Find the current floor of the Elevator
            e_floor = state[0][3 * self.observation_space: 4 * self.observation_space]
            curr_floor = -1
            for i, x in enumerate(e_floor):
                if x == 1:
                    curr_floor = i
                    break
            
            # Determine which floor to move to
            destination = self.serving_passenger.curr_floor
            if self.is_passenger_loaded():
                destination = self.serving_passenger.dest_floor

            # Check that current passenger is delivered 
            if not self.is_passenger_loaded() \
                and curr_floor == self.serving_passenger.dest_floor:
                self.serving_passenger = None
                continue

            # Move in the direction that the passenger is headed
            if destination > curr_floor:
                assert(curr_floor != self.observation_space - 1)
                return 1
            elif destination < curr_floor:
                assert(curr_floor != 0)
                return 2
            return 0
            
    def is_passenger_loaded(self):
        '''Check if the passenger is loaded in an elevator'''
        if self.serving_passenger.elevator != None:
            return True
        return False


    