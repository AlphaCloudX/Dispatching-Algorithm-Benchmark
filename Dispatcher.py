import heapq

from SimulationLogger import SimulationLogger
from Utils import CallStatus


class Dispatcher:
    def __init__(self, drivers: list, calls: list, grid: list):
        """
        :param drivers: List of Driver objects
        :param calls: List of Call objects
        :param grid: 2D grid representing the environment for shortest path calculation
        """
        self.drivers = drivers
        self.calls = calls
        self.queued_calls = []
        self.grid = grid
        self.tick = 0

        self.stats_collector = SimulationLogger()

    def manhattan_distance(self, loc1, loc2):
        """Returns Manhattan distance as a simple shortest path placeholder."""
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def find_closest_available_driver(self, call):
        """
        Finds the closest available driver who can complete the call.
        Returns (driver, travel_time) or (None, None) if no driver available.
        """
        min_distance = float('inf')
        closest_driver = None

        for driver in self.drivers:
            if driver.is_available() and call.can_truck_complete_job(driver.truck_type):
                distance = self.manhattan_distance(driver.current_location, call.breakdown_location)
                if distance < min_distance:
                    min_distance = distance
                    closest_driver = driver

        if closest_driver:
            return closest_driver, min_distance
        else:
            return None, None

    def step(self):
        self.tick += 1

        # First, update all drivers
        for driver in self.drivers:
            driver.step()

        # Process new calls at this tick
        for call in self.calls[:]:
            if call.tick_when_available <= self.tick:
                driver, travel_time = self.find_closest_available_driver(call)
                if driver:
                    # Assign call
                    driver.assign_call(call, travel_time)
                    call.current_status = CallStatus.EN_ROUTED
                    # call.call_statistics[CallStatus.EN_ROUTED.value] = call.call_statistics[CallStatus.EN_ROUTED.value] + 1
                    call.travel_time = travel_time
                    self.calls.remove(call)
                else:
                    # Still waiting: increment idle time here
                    call.call_statistics[CallStatus.IDLE.value] = call.call_statistics[CallStatus.IDLE.value] + 1

            # Can stop once we exceed the limit
            else:
                break

        # Show calls remaining after processing
        print(f"Tick {self.tick}: {len(self.calls)} calls remaining.")

        # Diagnostics for remaining calls every N ticks
        if self.tick % 100 == 0 or len(self.calls) < 20:
            print("==== Remaining Calls Diagnostic ====")
            for call in self.calls:
                print(f"Call scheduled at tick {call.tick_when_available}, "
                      f"required trucks: {call.required_truck_type}, "
                      f"breakdown location: {call.breakdown_location}, "
                      f"dropoff location: {call.dropoff_location}")

            # Check driver compatibility
            for call in self.calls:
                compatible_drivers = [d for d in self.drivers if call.can_truck_complete_job(d.truck_type)]
                available_drivers = [d for d in compatible_drivers if d.is_available()]
                print(f"Call at {call.breakdown_location} requires {call.required_truck_type}. "
                      f"{len(compatible_drivers)} compatible drivers, {len(available_drivers)} available.")

        # Optional: print driver statistics for debugging
        for driver in self.drivers:
            print(f"Driver {driver.driver_id} status: {driver.current_status}, call stats: {driver.call_statistics}")

        self.stats_collector.record(self.tick, self.drivers, self.calls)

    def run_until_complete(self):
        """
        Runs the simulation for the specified number of ticks.
        """
        while True:
            self.step()
            print(f"Tick {self.tick} completed.")

            # Check if all calls are processed and all drivers are idle
            all_calls_processed = len(self.calls) == 0
            all_drivers_idle = all(driver.is_available() for driver in self.drivers)

            if all_calls_processed and all_drivers_idle:
                print(f"Simulation complete at tick {self.tick}.")
                break
