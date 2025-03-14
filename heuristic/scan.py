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
        
        assert(curr_floor != -1 and curr_floor != self.observation_space)

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
            
            if above > below: # Move UP
                self.prev_floor = curr_floor
                return 1
            elif below > above: # Move DOWN
                self.prev_floor = curr_floor
                return 2
            else:
                return 0

        elif self.prev_floor < curr_floor: # UP
            for i in range(curr_floor, len(up_calls)):
                if up_calls[i] == 1 or down_calls[i] == 1:
                    self.prev_floor = curr_floor
                    return 1 # Return move UP action
            # If there are no calls above, then now start moving DOWN
            for i in range(0, curr_floor):
                if up_calls[i] == 1 or down_calls[i] == 1:
                    self.prev_floor = curr_floor
                    return 2 # Return move DOWN action
            self.prev_floor = curr_floor
            return 0

        else: # DOWN
            for i in range(0, curr_floor):
                if up_calls[i] == 1 or down_calls[i] == 1:
                    self.prev_floor = curr_floor
                    return 2 # Return move DOWN action
            # If there are no calls below, then now start moving UP
            for i in range(curr_floor, len(up_calls)):
                if up_calls[i] == 1 or down_calls[i] == 1:
                    self.prev_floor = curr_floor
                    return 1 # Return move UP action
            self.prev_floor = curr_floor
            return 0

        return 0 # Should NEVER go in this case


        
        

    


