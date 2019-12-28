# Potential observation for the agent:
# - Requsted floor and the direction(going up or down)
# - Each floor can have up to 2 requests (up and down)
# - This means that the elevator doesn't know how many people 
# will be requesting on each floor.

class Elevator():
    def init(self, curr_floor):
        """Initialize Elevator class."""
        self.curr_floor = curr_floor
        self.carrying_p = []