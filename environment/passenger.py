class Passenger():
    def __init__(self, current_floor, dest_floor, begin_wait_time):
        """Initialize the characteristics of a passenger."""
        self.curr_floor = current_floor
        self.dest_floor = dest_floor
        self.begin_wait_time = begin_wait_time
        self.begin_lift_time = None
        self.elevator = None