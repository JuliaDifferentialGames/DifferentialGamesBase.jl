# DifferentialGamesBase.jl

[![CI](https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://JuliaDifferentialGames.github.io/DifferentialGames.jl/dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

Core problem-specification and interface library for the [DifferentialGames.jl](https://github.com/JuliaDifferentialGames/DifferentialGames.jl) ecosystem.

> ⚠️ **Work in progress** — API may change before v1.0.0.

## What's in this package

- **Abstract type hierarchy** — `AbstractGameProblem`, `AbstractPlayerDynamics`, `AbstractStrategy`, …
- **Problem containers** — `GameProblem`, `LQGameProblem`, `PDGNEProblem`, `InverseGameProblem`
- **Dynamics** — `LinearDynamics`, `SeparableDynamics`, `CoupledNonlinearDynamics`; discretization via ZOH, matrix exponential, or OrdinaryDiffEq
- **Cost functions** — `LQStageCost`, `LQTerminalCost`, `NonlinearStageCost`; diagonal convenience constructors; composable cost-term DSL
- **Constraints** — `ControlBounds`, `StateBounds`, `ProximityConstraint`, `SharedInequality`, …
- **Trajectory expansion** — linearization and quadraticization for iterative LQ solvers
- **Solution types** — `GNEPSolution`, `Trajectory`, feedback and open-loop strategies
- **Solver interface** — `GameSolver`, `solve`, `WarmstartData`, `solver_capabilities`
- **Inverse games** — `InverseGameProblem`, `ForwardSolverWrapper`, `ObservationModel`, `InverseGameSolution`

## Installation

Most users should install the umbrella package:

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGames.jl")
```

To use this package directly:

```julia
] add DifferentialGamesBase
```

## Quick Example

```julia
using DifferentialGamesBase, LinearAlgebra

# Two-player double-integrator game
dynamics1 = (xi, ui, p, t) -> [xi[3]; xi[4]; ui[1]; ui[2]]
dynamics2 = (xi, ui, p, t) -> [xi[3]; xi[4]; ui[1]; ui[2]]

player1 = PlayerSpec(1, 4, 2, [1.0, 0.0, 0.0, 0.0], dynamics1,
    PlayerObjective(1,
        DiagonalLQStageCost([1.0, 1.0, 0.1, 0.1], [0.1, 0.1]),
        DiagonalLQTerminalCost([10.0, 10.0, 1.0, 1.0])))

player2 = PlayerSpec(2, 4, 2, [-1.0, 0.0, 0.0, 0.0], dynamics2,
    PlayerObjective(2,
        DiagonalLQStageCost([1.0, 1.0, 0.1, 0.1], [0.1, 0.1]),
        DiagonalLQTerminalCost([10.0, 10.0, 1.0, 1.0])))

game = PDGNEProblem([player1, player2], 3.0, 0.1)

# Solve with any solver from DifferentialGamesBaseSolvers.jl
# sol = solve(game, iLQGames())
```

## License

MIT License — see LICENSE file for details.

## Disclosure of Generative AI Usage

Generative AI (Claude Sonnet 4.5/4.6) was used in the creation of this library as a programming aid including guided code generation, assistance with performance optimization, and documentation. All code and documentation has been reviewed by the author(s) for accuracy.
