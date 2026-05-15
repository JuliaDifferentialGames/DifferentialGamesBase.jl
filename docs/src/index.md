# DifferentialGamesBase.jl

**DifferentialGamesBase.jl** is the core problem-specification and interface library for the DifferentialGames.jl ecosystem. It provides abstract types, concrete game problem containers, dynamics models, cost functions, constraints, and the solution/solver interfaces that all solver packages build on.

> **Note:** This package is under active development. The API may change between minor versions until v1.0.0.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/JuliaDifferentialGames/DifferentialGamesBase.jl")
```

## Package Structure

```
DifferentialGamesBase.jl
├── Abstract hierarchy       (AbstractGameProblem, AbstractPlayerDynamics, …)
├── Problem types            (GameProblem, LQGameProblem, PDGNEProblem, …)
├── Dynamics                 (LinearDynamics, SeparableDynamics, CoupledNonlinearDynamics)
├── Cost functions           (LQStageCost, LQTerminalCost, NonlinearStageCost, …)
├── Constraints              (ControlBounds, ProximityConstraint, SharedInequality, …)
├── Solutions                (GNEPSolution, Trajectory)
├── Solver interface         (GameSolver, solve, solver_capabilities)
└── Inverse games            (InverseGameProblem, InversePDGNEProblem, …)
```

## Quick Start

### LQ Game (shared-state, fully coupled)

```julia
using DifferentialGamesBase, LinearAlgebra

n, m = 4, 2          # state dim, per-player control dim
A  = 0.9 * I(n)
B  = [Matrix(I(n))[:, 1:1], Matrix(I(n))[:, 3:3]]   # one column per player
Q  = [diagm(ones(n)), diagm(ones(n))]
R  = [fill(0.1, 1, 1), fill(0.1, 1, 1)]
Qf = Q
x0 = ones(n)

game = LQGameProblem(A, B, Q, R, Qf, x0, 2.0; dt=0.1)
```

### Partially-Decoupled GNEP (separable dynamics)

```julia
using DifferentialGamesBase

# Build player 1
dynamics1 = (xi, ui, p, t) -> [xi[3]; xi[4]; ui[1]; ui[2]]
stage1    = DiagonalLQStageCost([1.0, 1.0, 0.1, 0.1], [0.1, 0.1])
terminal1 = DiagonalLQTerminalCost([10.0, 10.0, 1.0, 1.0])
player1   = PlayerSpec(1, 4, 2, [1.0, 0.0, 0.0, 0.0], dynamics1,
                       PlayerObjective(1, stage1, terminal1))

# Build player 2 (similarly)
dynamics2 = (xi, ui, p, t) -> [xi[2]; ui[1]; ui[2]]
stage2    = DiagonalLQStageCost([1.0, 0.1, 0.1], [0.1, 0.1])
terminal2 = DiagonalLQTerminalCost([10.0, 1.0, 1.0])
player2   = PlayerSpec(2, 3, 2, zeros(3), dynamics2,
                       PlayerObjective(2, stage2, terminal2))

# Collision avoidance constraint
col = SharedInequality([1, 2];
    func = (x, u, p, t) -> [2.0^2 - sum((x[1:2] - x[5:6]).^2)],
    dim  = 1)

game = PDGNEProblem([player1, player2], [col], 5.0, 0.1)
```

### Inverse Game

```julia
using DifferentialGamesBase

# Define a minimal forward solver wrapper
struct MyWrapper <: ForwardSolverWrapper end

function DifferentialGamesBase.predict_next_state(
    ::MyWrapper, prob::GameProblem{T}, x0::AbstractVector{T}
) where {T}
    return copy(x0)   # placeholder — replace with a real forward solve
end

prob = InversePDGNEProblem(
    [player1, player2],
    [KnownObjective(PlayerObjective(1, stage1, terminal1)), UnknownObjective()],
    FullStateObservation(7),
    MyWrapper(),
    5.0, 0.1
)
```

## Design Principles

- **Immutable problems.** `GameProblem` and `InverseGameProblem` are pure specifications. All mutable solver state lives in the solver itself, mirroring the DifferentialEquations.jl pattern.
- **Separable by default.** The `PDGNEProblem` constructor builds games where each player has its own dynamics and state — the common case in multi-agent control.
- **Composable costs.** Use the cost-term DSL (`minimize`, `track_goal`, `avoid_proximity`, …) or raw `LQStageCost` for maximum flexibility.
- **Type-stable numerics.** All core types are parameterized by the numeric type `T` (typically `Float64`).
