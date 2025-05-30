import heapq
from abc import ABC, abstractmethod
from collections import deque

def dijkstra(start, goal, cost_map):
    height, width = cost_map.shape
    visited = set()
    dist = {start: 0}
    heap = [(0, start)]

    def neighbors(pos):
        x, y = pos
        for nx, ny in [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]:
            if 0 <= nx < width and 0 <= ny < height:
                yield (nx, ny)

    while heap:
        current_dist, current = heapq.heappop(heap)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return current_dist  # shortest cost found

        for n in neighbors(current):
            if n in visited:
                continue
            new_dist = current_dist + cost_map[n[1]][n[0]]
            if new_dist < dist.get(n, float('inf')):
                dist[n] = new_dist
                heapq.heappush(heap, (new_dist, n))

    return float('inf')  # no path found

def shortest_tick_distance(a, b, perlin_map):
    return dijkstra(a, b, perlin_map)

class DispatchingAlgorithm(ABC):
    def __init__(self, drivers, map):
        self.drivers = drivers
        self.trafficMap = map

    @abstractmethod
    def add_call(self, call):
        pass

    @abstractmethod
    def tick(self):
        """Advance the simulation by one tick and return list of (driverID)s who completed calls"""
        pass

class PriorityQueueAlgorithm(DispatchingAlgorithm):
    def __init__(self, drivers, map):
        super().__init__(drivers, map)
        self.queue = []
        self.counter = 0

    def add_call(self, call):
        self.queue.append(call)

    def tick(self):

        # Update drivers
        for driver in self.drivers:
            if driver.tick():
                #completed.append(driver.driverID)
                self.counter +=1

        # Dispatch new calls
        available_drivers = [d for d in self.drivers if d.status == 0]
        remaining_calls = []

        for call in self.queue:
            if not available_drivers:
                remaining_calls.append(call)
                continue

            nearest_driver = min(
                available_drivers,
                key=lambda d: shortest_tick_distance(d.location, call.pickupLocation, self.trafficMap.noise_map),
                default=None
            )
            if nearest_driver:
                nearest_driver.start_call(call)
                available_drivers.remove(nearest_driver)
            else:
                remaining_calls.append(call)

        self.queue = remaining_calls
        return self.counter


class PerDriverQueueAlgorithm(DispatchingAlgorithm):
    def __init__(self, drivers, map):
        super().__init__(drivers, map)
        self.queues = {d.driverID: deque() for d in drivers}
        self.counter = 0

    def add_call(self, call):
        def last_location(driver):
            # If driver has calls queued, get dropOffLocation of last call
            if self.queues[driver.driverID]:
                return self.queues[driver.driverID][-1].dropoffLocation
            else:
                # Otherwise use driver's current location
                return driver.location

        nearest_driver = min(
            self.drivers,
            key=lambda d: shortest_tick_distance(last_location(d), call.pickupLocation, self.trafficMap.noise_map)
        )
        self.queues[nearest_driver.driverID].append(call)
        return nearest_driver.driverID

    def tick(self):
        for driver in self.drivers:
            if driver.tick():
                self.counter += 1

            if driver.status == 0 and self.queues[driver.driverID]:
                next_call = self.queues[driver.driverID].popleft()
                driver.start_call(next_call)

        return self.counter
