# Dispatching-Algorithm-Benchmark
Working as a dispatcher I noticed that manually locating tow trucks for members isn't efficient. I used 2 scheduling algorithms to compare their efficiency.

Here is an output that shows using a single queue and assigning calls one at a time is 18.5% more efficient than stacking drivers with calls based on their tow destination.
```
[PriorityQueue] Initial event queue length: 100
[PriorityQueue] Total expected calls: 100
[PriorityQueue] Max ticks to run: 500
[PriorityQueue] Tick 0: Completed calls so far = 0, Calls assigned = 0
[PriorityQueue] Tick 37: Completed calls so far = 0, Calls assigned = 1
...
[PriorityQueue] Tick 349: Completed calls so far = 99, Calls assigned = 100
[PriorityQueue] Tick 351: Completed calls so far = 100, Calls assigned = 100

[PerDriverQueue] Initial event queue length: 100
[PerDriverQueue] Total expected calls: 100
[PerDriverQueue] Max ticks to run: 500
[PerDriverQueue] Tick 0: Completed calls so far = 0, Calls assigned = 0
[PerDriverQueue] Tick 37: Completed calls so far = 0, Calls assigned = 1
...
[PerDriverQueue] Tick 423: Completed calls so far = 99, Calls assigned = 100
[PerDriverQueue] Tick 431: Completed calls so far = 100, Calls assigned = 100

--- Simulation Summary ---
PriorityQueue completed 100 calls out of 100 in 352 ticks.
  Description: Custom config with skewed and bimodal distributions
PerDriverQueue completed 100 calls out of 100 in 432 ticks.
  Description: Same config, different algorithm

--- Final Simulation Configuration ---
Drivers: 25
Ticks to Run: 500
Number of Calls: 100
DropOff Same as Pickup: False
Enable Towing: True
Scene Time Range: 5 - 20
Distributions:
  event_ticks: {'distribution_type': 'bimodal', 'mean': None, 'scale': None, 'skew': 0, 'bimodal_params': ((75, 15.0), (225, 15.0))}
  time_on_scene: {'distribution_type': 'skewed', 'mean': 10, 'scale': 3, 'skew': 7, 'bimodal_params': None}
  driver_location: {'distribution_type': 'normal', 'mean': 30.0, 'scale': 10, 'skew': 0, 'bimodal_params': None}
  call_location: {'distribution_type': 'normal', 'mean': 30.0, 'scale': 10, 'skew': 0, 'bimodal_params': None}
```
