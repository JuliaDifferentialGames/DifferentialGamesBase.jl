# Constraints

## Abstract Hierarchy

```@docs
AbstractConstraint
AbstractPrivateConstraint
AbstractPrivateInequality
AbstractPrivateEquality
AbstractSharedConstraint
AbstractSharedInequality
AbstractSharedEquality
is_private
is_shared
is_equality
is_inequality
is_convex
get_player
get_players
evaluate_constraint
constraint_jacobian
constraint_output_dim
is_active
constraint_violation
```

## Private Constraints

```@docs
ControlBounds
StateBounds
PrivateNonlinearInequality
PrivateNonlinearEquality
PrivateInequality
PrivateEquality
control_bounds
state_bounds
```

## Shared Constraints

```@docs
ProximityConstraint
CommunicationConstraint
LinearCoupling
SharedNonlinearInequality
SharedNonlinearEquality
SharedInequality
SharedEquality
collision_avoidance
keep_in_range
linear_coupling
```

## Continuous-Time Constraints

```@docs
ContinuousTimeConstraint
continuous_time_constraint
```

## Penalty Functions

```@docs
exterior_penalty_cubic
exterior_penalty_cubic_grad
```
