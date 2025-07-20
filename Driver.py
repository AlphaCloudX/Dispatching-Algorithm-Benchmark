from Calls import Call
from Utils import CallStatus


class Driver:
    def __init__(self, driver_id: int, truck_type: str, start_location=(0, 0)):
        self.driver_id = driver_id
        self.truck_type = truck_type
        self.current_call = None
        self.completed_calls: list[Call] = []
        self.current_location = start_location

        self.current_status = CallStatus.IDLE
        self.call_statistics = [0] * 4

        # New attributes for ticking system
        self.commute_time_left = 0
        self.on_location_time_left = 0
        self.towing_time_left = 0

    def assign_call(self, call: Call, travel_time: int):
        self.current_call = call
        self.current_status = CallStatus.EN_ROUTED
        self.commute_time_left = travel_time
        # Don't change location immediately; will update upon arrival

    def step(self):
        """
        Progress the driver's current task by one tick.
        """
        if self.current_status == CallStatus.EN_ROUTED:
            if self.commute_time_left > 0:
                self.commute_time_left -= 1
                if self.commute_time_left == 0:
                    # Arrived at location
                    self.current_location = self.current_call.breakdown_location
                    self.current_status = CallStatus.ON_LOCATION
                    self.on_location_time_left = self.current_call.time_on_location  # use call's time on location

        elif self.current_status == CallStatus.ON_LOCATION:
            if self.on_location_time_left > 0:
                self.on_location_time_left -= 1
                if self.on_location_time_left == 0:

                    # Is it a tow or local service
                    if self.current_call.breakdown_location == self.current_call.dropoff_location:
                        self.complete_current_call()
                        self.current_status = CallStatus.IDLE

                    else:
                        self.current_status = CallStatus.TOWING
                        # Set towing time based on distance
                        self.towing_time_left = abs(
                            self.current_call.breakdown_location[0] - self.current_call.dropoff_location[0]
                        ) + abs(
                            self.current_call.breakdown_location[1] - self.current_call.dropoff_location[1]
                        )

        elif self.current_status == CallStatus.TOWING:
            if self.towing_time_left > 0:
                self.towing_time_left -= 1
                if self.towing_time_left == 0:
                    self.complete_current_call()
                    self.current_status = CallStatus.IDLE

        # Always increment statistics regardless of status value
        print(f"Updating Driver #{self.driver_id} to {self.call_statistics[self.current_status.value] + 1}")
        self.call_statistics[self.current_status.value] = self.call_statistics[self.current_status.value] + 1

    def complete_current_call(self):
        if self.current_call:
            self.completed_calls.append(self.current_call)
            self.current_call = None
        self.current_status = CallStatus.IDLE  # Ensure status reset

    def is_available(self) -> bool:
        return self.current_call is None and self.current_status == CallStatus.IDLE

    def __str__(self):
        return (
            f"Driver(ID: {self.driver_id}, "
            f"Truck Type: {self.truck_type}, "
            f"Status: {self.current_status}, "
            f"Current Call: {self.current_call}, "
            f"Completed Calls: {len(self.completed_calls)})"
        )

    def __repr__(self):
        return self.__str__()
