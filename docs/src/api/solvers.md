# Solver Interface

This package defines the abstract interface that all solver packages implement.
Concrete solvers live in [DifferentialGamesBaseSolvers.jl](https://github.com/JuliaDifferentialGames/DifferentialGamesBaseSolvers.jl).

## Core Interface

```@docs
GameSolver
solve
_solve
WarmstartData
solver_capabilities
```

## Information Structures

```@docs
AbstractInformationStructure
PerfectStateInformation
OpenLoopInformation
PrivateObservation
SharedObservation
AsymmetricInformation
requires_belief_state
is_feedback_compatible
is_open_loop
```
