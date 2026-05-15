# Solutions

## Abstract Solution Type

```@docs
AbstractSolution
```

## GNEP Solution

```@docs
GNEPSolution
Trajectory
get_trajectory
get_cost
get_costs
get_strategy
has_strategy
has_shared_state
is_feedback
is_open_loop_solution
first_step_state
```

## Strategy Types

```@docs
AbstractStrategy
OpenLoopStrategy
FeedbackStrategy
zero_open_loop_strategy
zero_feedback_strategy
apply_strategy
to_open_loop
get_nominal_control
get_gain
get_feedforward
get_nominal_state
get_times
get_control_dims
control_offsets
```
