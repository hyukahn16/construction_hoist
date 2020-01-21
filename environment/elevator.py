import logging

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

    def __init__(self, env, id, curr_floor):
        """Initialize Elevator class."""
        self.curr_floor = curr_floor
        self.passengers = set()
        self.weight_capacity = 907.185 # Unit: Kilograms, 1 ton == 907.185 kg
        self.velocity = 100 # Unit: meters/minute
        self.env = env
        self.id = id
        self.idling_event = None
        self.state = None # 0 for idling - TODO: create enum class
        self.reward = 0 
        self.last_reward = 0

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
        # Check if this is a legal action
        assert action in self.legal_actions()

        # If action is idle
        if action == 0:
            self.idling_event = self.env.simul_env.process(self.ACTION_FUNCTION_MAP[action]())
            try:
                logging.debug("elevator.py: act() - about to idle")
                yield self.idling_event
            except:
                # if idling is interrupted (by the env), then trigger event to 
                # force a decision from the elevator
                logging.debug("elevator.py: idling is interrupted.")
                self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

        # If action is not idle
        else:
            yield self.env.simul_env.process(self.ACTION_FUNCTION_MAP[action]())

    def interrupt_idling(self):
        self.idling_event.interrupt()

    def idle(self):
        '''Idle.'''
        logging.debug("elevator.py: idle() - Elevator_{}".format(self.id))
        self.state = self.IDLE
        yield self.env.simul_env.timeout(10)

        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def move_up(self):
        logging.debug("elevator.py: move_up() - Elevator_{}".format(self.id))
        self.state = self.MOVING_UP
        self.env.load_passengers(self.id)

        yield self.env.simul_env.timeout(15)

        self.curr_floor += self.MOVING_UP
        logging.debug("elevator.py: move_up() - Elevator_{} at floor {}".format(self.id, self.curr_floor))
        self.env.load_passengers(self.id)
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    def move_down(self):
        logging.debug("elevator.py: move_down() - Elevator_{}".format(self.id))
        self.state = self.MOVING_DOWN
        self.env.load_passengers(self.id)

        yield self.env.simul_env.timeout(15)

        self.curr_floor += self.MOVING_DOWN
        logging.debug("elevator.py: move_down() - Elevator_{} at floor {}".format(self.id, self.curr_floor))
        self.env.load_passengers(self.id)
        self.env.trigger_epoch_event("ElevatorArrival_{}".format(self.id))

    # FIXME: may not need this function
    def legal_actions(self):
        '''Return list of actions that are legal in the current Elevator state.'''
        legal_actions = set([0, 1, 2])
        
        if self.curr_floor >= self.env.num_floors:
            legal_actions.remove(1)
        if self.curr_floor <= 0:
            legal_actions.remove(2)
        
        return legal_actions

    # FIXME: may not need thsi function
    def update_reward(self, reward):
        '''Update reward.
        Called from Environment.update_all_reward()
        '''
        self.reward += reward
        
