import random

import numpy as np

from Calls import Call
from Driver import Driver
from Utils import sample_distribution


def generate_traffic(x: int, y: int, events_distribution: dict):
    return sample_distribution(events_distribution, size=(x, y))


def generate_ticks(calls: dict, ticks_to_run: int, calls_to_queue: int):
    # Parse call frequency distribution to create PDF
    call_freq_dist = calls['call_frequency_distribution']
    dist_type = call_freq_dist['type']

    if dist_type == 'normal':
        pdf = np.random.normal(
            loc=call_freq_dist['mean'],
            scale=call_freq_dist['std'],
            size=ticks_to_run
        )
        pdf = np.clip(pdf, call_freq_dist['min'], call_freq_dist['max'])

    elif dist_type == 'uniform':
        pdf = np.random.uniform(
            low=call_freq_dist['min'],
            high=call_freq_dist['max'],
            size=ticks_to_run
        )
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")

    # Normalize PDF to probabilities
    probabilities = pdf / np.sum(pdf)

    # Create tick indices (each index represents a tick in the simulation)
    ticks = np.arange(ticks_to_run)

    # Validation: cannot sample more calls than available ticks
    if calls_to_queue > ticks_to_run:
        raise ValueError("calls_to_queue exceeds total number of ticks available.")

    # Random sample using weighted probabilities, no repeats
    sampled_ticks = np.random.choice(
        ticks,
        size=calls_to_queue,
        replace=False,
        p=probabilities
    )

    sampled_ticks = np.sort(sampled_ticks)

    return sampled_ticks


def generate_call(tick: int, calls: dict, grid_width: int, grid_height: int) -> Call:
    # Tow or No tow Call
    is_tow = random.choices(
        population=[False, True],
        weights=[calls['no_tow_chance'], calls['tow_chance']],
        k=1
    )[0]

    # Determine vehicle types based on tow status
    vehicle_types = list(calls['call_vehicle_distribution'].keys())
    vehicle_probs = list(calls['call_vehicle_distribution'].values())

    if not is_tow:
        # Get allowed vehicle types for no tow from config
        allowed_no_tow = set(calls['vehicles_allowed_for_no_tow'])

        # Get the vehicles that we are allowed to use for no tow
        filtered_vehicle_types = []
        filtered_vehicle_probs = []

        for vt, prob in zip(vehicle_types, vehicle_probs):
            if vt in allowed_no_tow:
                filtered_vehicle_types.append(vt)
                filtered_vehicle_probs.append(prob)

        # Check if any allowed vehicles remain
        if filtered_vehicle_types:
            # Normalize probabilities to sum to 1
            total_prob = sum(filtered_vehicle_probs)
            vehicle_types = filtered_vehicle_types
            vehicle_probs = [p / total_prob for p in filtered_vehicle_probs]
        else:
            raise ValueError("No allowed vehicle types available for no tow call.")

    # Choose required truck type based on final vehicle_types and vehicle_probs
    required_truck_type = random.choices(vehicle_types, weights=vehicle_probs, k=1)

    # Random time spent on location
    time_on_location = random.randint(calls['min_time_OL'], calls['max_time_OL'])

    # Generate breakdown location as a random grid coordinate
    breakdown_x = random.randint(0, grid_width - 1)
    breakdown_y = random.randint(0, grid_height - 1)
    breakdown_location = (breakdown_x, breakdown_y)

    # No tow -> dropoff = breakdown location
    if not is_tow:
        dropoff_location = breakdown_location
    else:
        dropoff_x = random.randint(0, grid_width - 1)
        dropoff_y = random.randint(0, grid_height - 1)
        dropoff_location = (dropoff_x, dropoff_y)

    # Return the generated Call object
    return Call(
        tick_when_available=tick,
        breakdown_location=breakdown_location,
        dropoff_location=dropoff_location,
        time_on_location=time_on_location,
        required_truck_type=required_truck_type
    )


def generate_calls(grid_width: int, grid_height: int, calls: dict, ticks_to_run: int, calls_to_queue: int):
    """
    Generates a list of Call objects for the simulation.

    Args:
        calls (dict): Calls configuration dictionary.
        ticks_to_run (int): Total simulation ticks.
        calls_to_queue (int): Number of calls to generate.
        grid_width (int): Grid width for random locations.
        grid_height (int): Grid height for random locations.

    Returns:
        List[Call]: Generated call objects.
    """
    # Load ticks to have calls at
    sampled_ticks = generate_ticks(calls, ticks_to_run, calls_to_queue)

    calls_list = []

    # Generate a call for each sampled tick
    for tick in sampled_ticks:
        call = generate_call(tick, calls, grid_width, grid_height)
        calls_list.append(call)

    return calls_list


def generate_drivers(config: dict) -> list:
    drivers = []
    driver_id = 1

    number_of_drivers = config['number_of_drivers']
    vehicle_distribution = config['driver_vehicle_distribution']

    # Build a list of truck types based on the distribution counts
    truck_types = []
    for truck_type, count in vehicle_distribution.items():
        truck_types.extend([truck_type] * count)

    # Safety check: ensure we don't assign more drivers than truck types available
    if len(truck_types) < number_of_drivers:
        raise ValueError("Not enough truck types defined for the number of drivers requested.")

    # If more truck types than drivers, only assign as many as needed
    truck_types = truck_types[:number_of_drivers]

    # Shuffle to randomize truck type assignment
    import random
    random.shuffle(truck_types)

    # Create driver objects
    for truck_type in truck_types:
        driver = Driver(driver_id=driver_id, truck_type=truck_type)
        drivers.append(driver)
        driver_id += 1

    return drivers
