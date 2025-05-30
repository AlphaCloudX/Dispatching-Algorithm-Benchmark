import multiprocessing
import random
from collections import deque

import numpy as np
from noise import pnoise2
from multiprocessing import Process
from scipy.stats import uniform, norm, skewnorm

from Algorithms import shortest_tick_distance, PriorityQueueAlgorithm, PerDriverQueueAlgorithm


class DistributionConfig:
    def __init__(self, distribution_type="uniform", *, mean=None, scale=None, skew=0, bimodal_params=None):
        self.distribution_type = distribution_type
        self.mean = mean
        self.scale = scale
        self.skew = skew
        self.bimodal_params = bimodal_params

    def compute(self, values: np.ndarray) -> np.ndarray:
        return DistributionHelper.compute_distribution(
            values=values,
            distribution_type=self.distribution_type,
            mean=self.mean,
            scale=self.scale,
            skew=self.skew,
            bimodal_params=self.bimodal_params
        )

    def to_dict(self):
        return {
            "distribution_type": self.distribution_type,
            "mean": self.mean,
            "scale": self.scale,
            "skew": self.skew,
            "bimodal_params": self.bimodal_params
        }

    @staticmethod
    def from_dict(data: dict):
        return DistributionConfig(
            distribution_type=data.get("distribution_type", "uniform"),
            mean=data.get("mean"),
            scale=data.get("scale"),
            skew=data.get("skew", 0),
            bimodal_params=data.get("bimodal_params")
        )

# Constants / Config as a class
class SimulationConfig:
    def __init__(self):
        self.minutes_width = 60
        self.minutes_height = 60
        self.dropOffSameLocationAsPickup = False
        self.drivers = 10
        self.ticksToRun = 500
        self.numberOfCalls = 100
        self.enableTowing = True

        self.min_time_on_scene = 5
        self.max_time_on_scene = 20

        self.distributions = {
            "event_ticks": DistributionConfig(
                distribution_type="normal",
                mean=self.ticksToRun / 2,
                scale=self.ticksToRun * 0.1,
                bimodal_params=(
                    (self.ticksToRun * 0.25, self.ticksToRun * 0.1),
                    (self.ticksToRun * 0.75, self.ticksToRun * 0.1)
                )
            ),
            "time_on_scene": DistributionConfig(
                distribution_type="skewed",
                mean=12,
                scale=4,
                skew=5
            ),
            "driver_location": DistributionConfig(distribution_type="uniform"),
            "call_location": DistributionConfig(distribution_type="uniform")
        }

    def get_distribution(self, key: str) -> DistributionConfig:
        return self.distributions.get(key)

    def to_dict(self):
        return {k: v.to_dict() for k, v in self.distributions.items()}

class DistributionHelper:
    @staticmethod
    def compute_distribution(
        values: np.ndarray,
        distribution_type: str = "uniform",
        *,
        mean: float = None,
        scale: float = None,
        skew: float = 0,
        bimodal_params: tuple = None
    ) -> np.ndarray:
        """
        Compute and normalize a probability distribution over `values`.

        Parameters:
        - values: np.ndarray of numeric values.
        - distribution_type: "uniform", "normal", "bimodal", or "skewed".
        - mean: mean/loc parameter for normal/skewed.
        - scale: scale/std deviation parameter.
        - skew: skewness parameter (only for skewed).
        - bimodal_params: tuple of two (mean, scale) pairs for bimodal distribution.

        Returns:
        - Normalized numpy probability distribution over values.
        """
        if distribution_type == "normal":
            probs = norm.pdf(values, loc=mean, scale=scale)

        elif distribution_type == "bimodal":
            (mean1, scale1), (mean2, scale2) = bimodal_params
            probs = norm.pdf(values, loc=mean1, scale=scale1) + norm.pdf(values, loc=mean2, scale=scale2)

        elif distribution_type == "skewed":
            probs = skewnorm.pdf(values, a=skew, loc=mean, scale=scale)

        else:  # uniform or fallback
            if scale is None:
                scale = values.max() - values.min()
            probs = uniform.pdf(values, loc=values.min(), scale=scale)

        probs /= probs.sum()
        return probs

class CoordinateGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def random_uniform_coord(self):
        x = random.uniform(0, self.width)
        y = random.uniform(0, self.height)
        return x, y

    def weighted_coord(self, dist_config: DistributionConfig):
        x_vals = np.linspace(0, self.width - 1, 100)
        y_vals = np.linspace(0, self.height - 1, 100)

        x_probs = dist_config.compute(x_vals)
        y_probs = dist_config.compute(y_vals)

        x = np.random.choice(x_vals, p=x_probs)
        y = np.random.choice(y_vals, p=y_probs)

        return round(x), round(y)

class TrafficMap:
    def __init__(self, width=60, height=60, scale=10.0, octaves=1, persistence=0.5, lacunarity=2.0):
        self.width = width
        self.height = height
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.noise_map = self._generate_perlin_noise_map()

    def _generate_perlin_noise_map(self):
        noise_map = np.zeros((self.height, self.width))
        for y in range(self.height):
            for x in range(self.width):
                noise_val = pnoise2(x / self.scale, y / self.scale, octaves=self.octaves,
                                    persistence=self.persistence, lacunarity=self.lacunarity,
                                    repeatx=self.width, repeaty=self.height, base=0)
                normalized = (noise_val + 1) / 2
                scaled = normalized * 1.5 + 0.5
                noise_map[y][x] = round(scaled, 1)
        return noise_map

    def get_noise_value(self, location):
        x, y = location
        return self.noise_map[y][x]


class Call:
    def __init__(self, pickup, dropoff, ticksOnScene, is_service_call=False):
        self.pickupLocation = pickup
        self.dropoffLocation = dropoff
        self.ticksOnScene = ticksOnScene
        self.is_service_call = is_service_call


    def __str__(self):
        return f"Call(Pickup={self.pickupLocation}, Dropoff={self.dropoffLocation}, OnScene={self.ticksOnScene})"

    def __repr__(self):
        return self.__str__()

class Driver:
    STATUS_IDLE = 0
    STATUS_EN_ROUTE = 1
    STATUS_ON_LOCATION = 2
    STATUS_TRAVELING_WITH = 3

    STATUS_NAMES = {
        STATUS_IDLE: "Idle",
        STATUS_EN_ROUTE: "En Route",
        STATUS_ON_LOCATION: "On Location",
        STATUS_TRAVELING_WITH: "Transporting"
    }

    def __init__(self, driverid, location, trafficmap):
        self.driverID = driverid
        self.location = location
        self.status = self.STATUS_IDLE
        self.timeLeft = 0
        self.currentCall = None
        self.trafficMap = trafficmap

    def start_call(self, call: Call):
        self.currentCall = call
        self.status = self.STATUS_EN_ROUTE
        self.timeLeft = shortest_tick_distance(self.location, call.pickupLocation, self.trafficMap.noise_map)
        self.location = call.pickupLocation

    def start_er(self, call: Call):
        self.currentCall = call
        self.status = self.STATUS_EN_ROUTE
        self.timeLeft = shortest_tick_distance(self.location, call.pickupLocation, self.trafficMap.noise_map)
        self.location = call.pickupLocation

    def start_ol(self):
        self.status = self.STATUS_ON_LOCATION
        duration = self.trafficMap.get_noise_value(self.location)
        self.timeLeft = int(duration * 20)

    def start_tw(self):
        self.status = self.STATUS_TRAVELING_WITH
        self.timeLeft = shortest_tick_distance(self.location, self.currentCall.pickupLocation,
                                               self.trafficMap.noise_map)
        self.location = self.currentCall.pickupLocation

    def clear(self):
        self.status = self.STATUS_IDLE
        self.timeLeft = 0
        self.currentCall = None

    def tick(self):
        if self.timeLeft > 0:
            self.timeLeft -= 1
            return False  # Still working

        # When timeLeft == 0, advance state machine
        if self.status == self.STATUS_EN_ROUTE:
            self.start_ol()
            return False

        elif self.status == self.STATUS_ON_LOCATION:
            if self.currentCall.is_service_call or self.currentCall.pickupLocation == self.currentCall.dropoffLocation:
                self.clear()
                return True  # Call completed
            else:
                self.start_tw()
                return False

        elif self.status == self.STATUS_TRAVELING_WITH:
            self.location = self.currentCall.dropoffLocation
            self.clear()
            return True  # Call completed

        elif self.status == self.STATUS_IDLE:
            return False

    def __str__(self):
        status_name = self.STATUS_NAMES.get(self.status, "Unknown")
        return (f"Driver {self.driverID}: "
                f"Status={status_name}, "
                f"Location={self.location}, "
                f"TimeLeft={self.timeLeft}, "
                f"Call={'Yes' if self.currentCall else 'No'}")

    def __repr__(self):
        return self.__str__()



# Will create a list of events and at what tick they should be executed, sorted low to high and they are removed from the queue once assigned
# This means that the sim can stop once the events list is empty and all drivers are finished
class Event:
    def __init__(self, simulation_config: SimulationConfig):
        self.simulation_config = simulation_config
        self.grid_width = simulation_config.minutes_width
        self.grid_height = simulation_config.minutes_height
        self.number_of_calls = simulation_config.numberOfCalls
        self.number_of_ticks = simulation_config.ticksToRun

        self.coord_generator = CoordinateGenerator(self.grid_width, self.grid_height)

        # Event timing distribution
        event_tick_dist = simulation_config.get_distribution("event_ticks")
        self.values = np.arange(0, self.number_of_ticks)
        self.probabilities = event_tick_dist.compute(self.values)

        # Sample the ticks at which events will occur
        self.sampled_ticks = np.random.choice(
            self.values, size=self.number_of_calls, replace=False, p=self.probabilities
        )
        self.sampled_ticks.sort()

    def generate_calls(self):
        events = []

        pickup_dist = self.simulation_config.get_distribution("call_location")
        dropoff_dist = self.simulation_config.get_distribution("call_location")

        for tick in self.sampled_ticks:
            pickup = self.coord_generator.weighted_coord(pickup_dist)

            is_service_call = not self.simulation_config.enableTowing
            if is_service_call:
                dropoff = pickup
            else:
                dropoff = self.coord_generator.weighted_coord(dropoff_dist)
                while dropoff == pickup:
                    dropoff = self.coord_generator.weighted_coord(dropoff_dist)

            ticks_on_scene = random.randint(
                self.simulation_config.min_time_on_scene,
                self.simulation_config.max_time_on_scene
            )

            call = Call(pickup, dropoff, ticks_on_scene, is_service_call)
            events.append((tick, call))

        return events


def run_simulation(AlgorithmClass, name, description="", config_seed=None, config: SimulationConfig = None):
    if config_seed is not None:
        random.seed(config_seed)
        np.random.seed(config_seed)

    if config is None:
        config = SimulationConfig()

    traffic_map = TrafficMap(config.minutes_width, config.minutes_height)
    event_queue_gen = Event(config)
    events = event_queue_gen.generate_calls()
    event_queue = deque(events)

    print(f"[{name}] Initial event queue length: {len(event_queue)}")
    print(f"[{name}] Total expected calls: {config.numberOfCalls}")
    print(f"[{name}] Max ticks to run: {config.ticksToRun}")

    driver_location_dist = config.get_distribution("driver_location")
    coord_gen = CoordinateGenerator(config.minutes_width, config.minutes_height)
    drivers = [
        Driver(
            i,
            coord_gen.weighted_coord(driver_location_dist),
            traffic_map
        )
        for i in range(config.drivers)
    ]

    dispatch_algo = AlgorithmClass(drivers, traffic_map)

    tick = 0
    completed_calls = 0
    calls_assigned = 0
    max_ticks = config.ticksToRun

    last_completed_calls = -1
    last_calls_assigned = -1

    while event_queue or any(driver.status != Driver.STATUS_IDLE for driver in drivers):
        while event_queue and event_queue[0][0] == tick:
            event_tick, call = event_queue.popleft()
            dispatch_algo.add_call(call)
            calls_assigned += 1

        completed_calls = dispatch_algo.tick()

        if completed_calls != last_completed_calls or calls_assigned != last_calls_assigned:
            print(f"[{name}] Tick {tick}: Completed calls so far = {completed_calls}, Calls assigned = {calls_assigned}")
            last_completed_calls = completed_calls
            last_calls_assigned = calls_assigned

        tick += 1
        if tick > max_ticks * 5:
            print(f"[{name}] Exceeded tick limit. Terminating.")
            break

    return {
        "name": name,
        "description": description,
        "total_ticks": tick,
        "completed_calls": completed_calls,
        "expected_calls": config.numberOfCalls
    }


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    base_config = SimulationConfig()

    # You can change these high-level config values:
    base_config.drivers = 25
    base_config.ticksToRun = 500
    base_config.numberOfCalls = 100

    # NOTE: If you change ticksToRun, consider updating the distributions that depend on it.
    # For example, if event_ticks is bimodal, the peaks should reflect the new time range:
    # Example: set peaks at 15% and 45% of ticksToRun
    # This simulate a morning and evening rush
    peak1 = int(base_config.ticksToRun * 0.15)
    peak2 = int(base_config.ticksToRun * 0.45)
    std_dev = base_config.ticksToRun * 0.03  # Adjust width of peaks accordingly

    base_config.distributions["event_ticks"] = DistributionConfig(
        distribution_type="bimodal",
        bimodal_params=((peak1, std_dev), (peak2, std_dev))
    )

    # If you change expected scene time behavior, update mean/scale/skew:
    base_config.distributions["time_on_scene"] = DistributionConfig(
        distribution_type="skewed",
        mean=10,    # average scene time
        scale=3,    # spread of time
        skew=7      # right-skewed (long tail)
    )

    # If you change the map size (minutes_width/height), update location means to center of map
    map_center_x = base_config.minutes_width / 2
    map_center_y = base_config.minutes_height / 2
    location_std_dev = 10  # spread around center

    base_config.distributions["driver_location"] = DistributionConfig(
        distribution_type="normal",
        mean=map_center_x,
        scale=location_std_dev
    )
    base_config.distributions["call_location"] = DistributionConfig(
        distribution_type="normal",
        mean=map_center_y,
        scale=location_std_dev
    )

    # Run both algorithms with the same base config
    result1 = run_simulation(
        PriorityQueueAlgorithm,
        "PriorityQueue",
        description="Custom config with skewed and bimodal distributions",
        config_seed=42,
        config=base_config
    )

    result2 = run_simulation(
        PerDriverQueueAlgorithm,
        "PerDriverQueue",
        description="Same config, different algorithm",
        config_seed=42,
        config=base_config
    )

    # Print simulation results
    print("\n--- Simulation Summary ---")
    for res in [result1, result2]:
        print(
            f"{res['name']} completed {res['completed_calls']} calls out of {res['expected_calls']} in {res['total_ticks']} ticks.")
        print(f"  Description: {res['description']}")

    # Print final configuration
    print("\n--- Final Simulation Configuration ---")
    print(f"Drivers: {base_config.drivers}")
    print(f"Ticks to Run: {base_config.ticksToRun}")
    print(f"Number of Calls: {base_config.numberOfCalls}")
    print(f"DropOff Same as Pickup: {base_config.dropOffSameLocationAsPickup}")
    print(f"Enable Towing: {base_config.enableTowing}")
    print(f"Scene Time Range: {base_config.min_time_on_scene} - {base_config.max_time_on_scene}")
    print("Distributions:")
    for key, dist in base_config.distributions.items():
        print(f"  {key}: {dist.to_dict()}")
