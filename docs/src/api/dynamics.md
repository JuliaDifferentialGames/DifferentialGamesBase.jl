# Dynamics

## Abstract Types

```@docs
AbstractPlayerDynamics
ContinuousPlayerDynamics
DiscretePlayerDynamics
CoupledPlayerDynamics
LinearPlayerDynamics
is_continuous
is_discrete
is_linear
is_separable_dynamics
```

## Concrete Dynamics

```@docs
LinearDynamics
SeparableDynamics
CoupledNonlinearDynamics
get_A
get_B
get_B_concatenated
is_ltv
total_state_dim
total_control_dim
```

## Time Horizon

```@docs
TimeHorizon
DiscreteTime
ContinuousTime
```

## Discretization

```@docs
AbstractDiscretizationMethod
ZOHDiscretization
MatrixExpDiscretization
DiffEqDiscretization
DiscreteApproximation
discretize
da_step
```

## Dynamics Interface

```@docs
evaluate_dynamics
dynamics_jacobian
rollout
rollout_strategy
```

## Trajectory Expansion

```@docs
TrajectoryExpansion
DynamicsExpansion
CostExpansion
expand
linearize_dynamics
quadraticize_costs
assemble_lq_game
reference_trajectory
```
