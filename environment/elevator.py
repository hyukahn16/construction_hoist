import logging
import numpy as np

# Potential observation for the agent:
# - Requsted floor and the direction(going up or down)
# - Each floor can have up to 2 requests (up and down)
# - This means that the elevator doesn't know how many people 
# will be requesting on each floor.

class Elevator():
    # num_states = 3 not sure if neeeded
    IDLE = 0
    MOVING_UP = 1
    MOVING_DOWN = -1
    LOADING = 2

    def __init__(self, env, id):
        """Initialize Elevator class."""
        self.curr_floor = 0
        self.passengers = set()
        self.requests = np.zeros(env.num_floors) # floor requests from Passengers inside the Elevator # FIXME implement
        self.weight_capacity = 907.185 # Unit: Kilograms, 1 ton == 907.185 kg
        self.env = env
        self.id = id
        self.idling_event = None
        self.state = None
        self.reward = 0 # Cumulative reward
        self.num_served = 0
        # Testing variables
        self.min_visited = 39
        self.max_visited = 0

        self.ACTION_FUNCTION_MAP = {
            0: self.idle,
            1: self.move_up,
            2: self.move_down
        }

        # When Elevator is first created, it runs in idle action
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

        # If action is idle
        if action == 0:
            self.idling_event = \
                self.env.simul_env.process(self.ACTION_FUNCTION_MAP[action]())
            try:
                yield self.idling_event
            except:
                # if idling is interrupted (by the env), then trigger event to 
                # force a decision from the elevator
                logging.debug("elevator.py: idling is interrupted.")
                self.state = None
                self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))
                self.idling_event = None

        # If action is not idle
        else:
            yield self.env.simul_env.process(self.ACTION_FUNCTION_MAP[action]())

    def interrupt_idling(self):
        assert(self.state == self.IDLE)
        self.idling_event.interrupt()

    def idle(self):
        '''Idle.'''
        logging.debug("elevator.py: idle() - Elevator_{}".format(self.id))
        self.state = self.IDLE
        yield self.env.simul_env.timeout(2)
        self.state = self.LOADING

        logging.debug("elevator.py: idle() - Elevator_{} at floor {}".format(self.id, self.curr_floor))
        self.env.load_passengers(self.id)
        self.state = None
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def move_up(self):
        assert(self.curr_floor < self.env.num_floors - 1)
        logging.debug("elevator.py: move_up() - Elevator_{} from floor {}".format(self.id, self.curr_floor))

        self.state = self.LOADING
        self.env.load_passengers(self.id, self.MOVING_UP)
        self.state = self.MOVING_UP
        yield self.env.simul_env.timeout(2)

        self.curr_floor += 1

        if self.curr_floor > self.max_visited:
            self.max_visited = self.curr_floor

        logging.debug("elevator.py: move_up() - Elevator_{} at floor {}".format(self.id, self.curr_floor))
        self.state = self.LOADING
        self.env.load_passengers(self.id)
        self.state = None
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def move_down(self):
        assert(self.curr_floor > 0)
        logging.debug("elevator.py: move_down() - Elevator_{} from floor {}".format(self.id, self.curr_floor))

        self.state = self.LOADING
        self.env.load_passengers(self.id, self.MOVING_DOWN)
        self.state = self.MOVING_DOWN
        yield self.env.simul_env.timeout(2)

        self.curr_floor -= 1

        if self.curr_floor < self.min_visited:
            self.min_visited = self.curr_floor

        logging.debug("elevator.py: move_down() - Elevator_{} at floor {}".format(self.id, self.curr_floor))
        self.state = self.LOADING
        self.env.load_passengers(self.id)
        self.state = None
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def legal_actions(self):
        '''Return list of actions that are legal in the current Elevator state.
        
            0 IDLE
            1 MOVE UP
            2 MOVE DOWN
        '''
        legal = set([0, 1, 2])
        
        if self.curr_floor == self.env.num_floors-1:
            legal.remove(1)
        if self.curr_floor == 0:
            legal.remove(2)
        
        return legal

    def update_reward(self, reward):
        '''Update reward.
        Called from Environment.update_all_reward()
        '''
        self.reward += reward
        
