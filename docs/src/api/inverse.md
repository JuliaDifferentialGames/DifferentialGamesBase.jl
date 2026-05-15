# Inverse Games

Inverse game theory asks: given observed trajectories from a Nash equilibrium, recover the players' unknown cost functions.

DifferentialGamesBase provides the problem specification and interface. Concrete inverse solvers (e.g., MONGOOSE) are implemented in separate packages.

## Player Knowledge

```@docs
PlayerKnowledge
KnownObjective
UnknownObjective
```

## Observation Models

```@docs
ObservationModel
observe
observation_dim
FullStateObservation
NoisyObservation
```

## Forward Solver Interface

```@docs
ForwardSolverWrapper
solve_forward
predict_next_state
```

## Inverse Game Problem

```@docs
InverseGameProblem
InversePDGNEProblem
unknown_players
known_players
n_unknown
known_objective
as_forward_problem
```

## Solver State

```@docs
InverseSolverState
ObservationData
push_observation!
```

## Inverse Solution

```@docs
InverseGameSolution
get_weights
get_weight_history
```
