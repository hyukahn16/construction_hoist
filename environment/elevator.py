import logging
import numpy as np

# Potential observation for the agent:
# - Requsted floor and the direction(going up or down)
# - Each floor can have up to 2 requests (up and down)
# - This means that the elevator doesn't know how many people 
# will be requesting on each floor.

class Elevator():
    LOAD = 0
    MOVING_UP = 1
    MOVING_DOWN = -1
    IDLE = 999

    def __init__(self, env, id):
        """Initialize Elevator class."""
        self.curr_floor = 0
        self.passengers = set()
        self.requests = np.zeros(env.num_floors) # floor requests from Passengers inside the Elevator
        self.weight_capacity = 907.185 # Unit: Kilograms, 1 ton == 907.185 kg
        self.env = env
        self.id = id
        self.idling_event = None
        self.state = None
        self.reward = 0 # Per step reward
        self.num_served = 0

        self.total_lift_time = 0
        self.total_lift_passengers = 0

        self.ACTION_FUNCTION_MAP = {
            0: self.load,
            1: self.move_up,
            2: self.move_down,
        }

        # When Elevator is first created, it runs the IDLE action
        self.env.simul_env.process(self.act(0))

    def act(self, action):
        '''Create process for elevator to take action.
        0 IDLE
        1 UP
        2 DOWN
        '''
        if action == -1:
            return # do nothing

        # Check if this is a legal action
        if action not in self.legal_actions():
            # idle if not a legal action
            action = 0

        # Save passenger's lift time for testing
        for p in self.passengers:
            self.update_lift_time(p)

        # If action is idle
        if action == 0:
            self.idling_event = \
                self.env.simul_env.process(self.ACTION_FUNCTION_MAP[action]())
            try:
                yield self.idling_event
            except:
                # if idling is interrupted (by the env), then trigger event to 
                # force a decision from the elevator
                self.state = None
                self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))
                self.idling_event = None

        # If action is not idle
        else:
            yield self.env.simul_env.process(self.ACTION_FUNCTION_MAP[action]())

    def interrupt_idling(self):
        #assert(self.state == self.IDLE)
        self.idling_event.interrupt()

    # FIXME: Unused
    def idle(self):
        #self.state = self.IDLE
        yield self.env.simul_env.timeout(7)
        self.state = None
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def move_up(self):
        assert(self.curr_floor < self.env.num_floors - 1)
        self.state = self.MOVING_UP
        self.env.moving_reward(self.id, self.state)
        yield self.env.simul_env.timeout(20)
        self.curr_floor += 1
        self.state = None
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def move_down(self):
        assert(self.curr_floor > 0)
        self.state = self.MOVING_DOWN
        self.env.moving_reward(self.id, self.state)
        yield self.env.simul_env.timeout(20)
        self.curr_floor -= 1
        self.state = None
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def load(self):
        self.state = self.LOAD
        self.env.load_passengers(self.id)
        yield self.env.simul_env.timeout(50)
        self.state = None
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def legal_actions(self):
        '''Return list of actions that are legal in the current Elevator state.
            0 IDLE
            1 MOVE UP
            2 MOVE DOWN
        '''
        legal = set([i for i in range(len(self.ACTION_FUNCTION_MAP))])
        
        if self.curr_floor == self.env.num_floors-1:
            legal.remove(1)
        if self.curr_floor == 0:
            legal.remove(2)
        
        legal = list(legal)
        return legal

    def update_reward(self, reward):
        '''Update reward.
        Called from Environment.update_all_reward()
        '''
        self.reward += reward
        
    def calculate_avg_lift_time(self):
        '''Calculate average lifting time.'''
        if not self.passengers:
            return 0

        for p in self.passengers:
                self.total_lift_time += (self.env.now() - p.begin_lift_time)
                self.total_lift_passengers += 1

        avg_lift_time = self.total_lift_time / self.total_lift_passengers

        # Reset values
        self.total_lift_passengers = 0
        self.total_lift_time = 0
        return avg_lift_time

    def update_lift_time(self, p):
        self.total_lift_passengers += 1
        self.total_lift_time += (self.env.now() - p.begin_lift_time)