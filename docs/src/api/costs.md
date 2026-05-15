# Costs & Objectives

## Abstract Types

```@docs
AbstractCost
AbstractStageCost
AbstractTerminalCost
```

## LQ Costs

```@docs
LQStageCost
LQTerminalCost
DiagonalLQStageCost
DiagonalLQTerminalCost
get_Q
get_R
get_M
get_q
get_r
evaluate_stage_cost
evaluate_terminal_cost
stage_cost_gradient
terminal_cost_gradient
stage_cost_hessian
terminal_cost_hessian
is_ltv
```

## Nonlinear Costs

```@docs
NonlinearStageCost
NonlinearTerminalCost
```

## Player Objective

```@docs
PlayerObjective
total_cost
get_objective
```

## Cost Term DSL

```@docs
AbstractCostTerm
AbstractTerminalCostTerm
CompositeCostTerm
CompositeTerminalCostTerm
evaluate_cost_term
cost_term_gradient
cost_term_hessian
is_quadratic
is_separable_term
player_slice
minimize
```

## Standard Cost Terms

```@docs
QuadraticStateCost
QuadraticControlCost
ProximityCost
CommunicationCost
ControlBarrierCost
QuadraticTerminalCost
ProximityTerminalCost
track_goal
regularize_input
avoid_proximity
maintain_proximity
terminal_goal
```
