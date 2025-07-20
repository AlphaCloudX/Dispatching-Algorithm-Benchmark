from Utils import CallStatus


class Call:
    def __init__(self, tick_when_available, breakdown_location: tuple[int, int], dropoff_location: tuple[int, int],
                 time_on_location: int, required_truck_type: list[str]):
        self.breakdown_location = breakdown_location
        self.dropoff_location = dropoff_location
        self.time_on_location = time_on_location
        self.required_truck_type = required_truck_type

        self.tick_when_available = tick_when_available

        self.current_status = CallStatus.IDLE
        self.call_statistics = [
                                   0] * 1  # Only track the IDLE time, we dont care too much about the other times since we can hard code those later

    def update_call_status(self):
        # If we do not perform a tow, auto mark to complete
        if self.current_status == CallStatus.ON_LOCATION and self.breakdown_location == self.dropoff_location:
            self.current_status = CallStatus.COMPLETED
        else:
            self.call_statistics[self.current_status.value] = self.call_statistics[self.current_status.value] + 1

            # Safely increment status
            next_value = self.current_status.value + 1

            # Ensure we do not go past maximum enum value
            if next_value < len(CallStatus):
                self.current_status = CallStatus(next_value)
            else:
                self.current_status = CallStatus.COMPLETED

    def can_truck_complete_job(self, driver_truck_type: CallStatus) -> bool:
        return driver_truck_type in self.required_truck_type

    def __str__(self):
        return (
            f"Call(\n"
            f"  Tick When Available: {self.tick_when_available},\n"
            f"  Breakdown Location: {self.breakdown_location},\n"
            f"  Dropoff Location: {self.dropoff_location},\n"
            f"  Time on Location: {self.time_on_location},\n"
            f"  Required Truck Type: {self.required_truck_type},\n"
            f"  Current Status: {self.current_status.name},\n"
            f"  Call Statistics: {self.call_statistics}\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()
