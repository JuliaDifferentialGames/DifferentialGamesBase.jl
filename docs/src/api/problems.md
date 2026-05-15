# Problem Types

## Abstract Hierarchy

```@docs
AbstractGameProblem
AbstractDeterministicGame
AbstractStochasticGame
AbstractPartiallyObservableGame
AbstractInverseGameProblem
AbstractPotentialGame
is_deterministic
is_stochastic
is_partially_observable
is_inverse
```

## Game Problem Container

```@docs
GameProblem
GameMetadata
CouplingGraph
PlayerSpec
```

## Constructors

```@docs
LQGameProblem
LTVLQGameProblem
PDGNEProblem
validate_game_problem
```

## Property Queries

```@docs
num_players
n_steps
state_dim
control_dim
has_separable_dynamics
is_lq_game
is_pd_gnep
is_potential_game
has_shared_constraints
is_unconstrained
is_separable
get_objective
```

## Player / DifferentialGame API

```@docs
Player
DifferentialGame
remake
```
