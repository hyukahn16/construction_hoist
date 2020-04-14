'''
Heuristic algorithm for Elevator which follows the SCAN algorithm
1. proceed in the same direction until the last request in that direction.
2. If there is no request, stop and proceed towards other direction if there is any requests from the other direction
Source: https://www.quora.com/What-algorithm-is-used-in-modern-day-elevators
'''

class ScanAgent():
    def __init__(self, total_floors):
        self.prev_floor = 0
        self.observation_space = total_floors

    def action(self, state):
        curr_floor = None # Get the actual floor
        up_calls = state[0][0 : self.observation_space]
        down_calls = state[0][self.observation_space: 2 * self.observation_space]
        req_calls = state[0][2 * self.observation_space: 3 * self.observation_space]
        e_floor = state[0][3 * self.observation_space: 4 * self.observation_space]
        
        curr_floor = -1
        for i, x in enumerate(e_floor):
            if x == 1:
                curr_floor = i
                break
        
        assert(curr_floor != -1)

        if self.prev_floor == curr_floor: # IDLE
            # If IDLE, then go whichever direction has more requests
            above = 0
            below = 0
            for i in range(len(up_calls)):
                if up_calls[i] == 1 or down_calls[i] == 1:
                    if i < curr_floor:
                        below += 1
                    elif i > curr_floor:
                        above += 1
            
            if above > below:
                return 1
            else:
                return 2

        elif self.prev_floor < curr_floor: # UP
            for i in range(curr_floor, len(up_calls)):
                if up_calls[i] == 1 or down_calls[i] == 1:
                    return 1 # Return move UP action
            # If there are no calls above, then now start moving DOWN
            return 2

        else: # DOWN
            for i in range(0, curr_floor):
                if up_calls[i] == 1 or down_calls[i] == 1:
                    return 2 # Return move DOWN action
            # If there are no calls below, then now start moving UP
            return 1

        return -1 # Should NEVER go in this case
        # 1 proceed in the same direction until last request is fulfilled in that direction
        # FIXME: need to know which direction the elevator is current moving in


        
        

    


