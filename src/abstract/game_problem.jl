# ============================================================================
# AbstractGameProblem hierarchy
#
# Sits above GameProblem{T}. Solvers dispatch on these abstract types so
# that stochastic, partially observable, and inverse variants slot in without
# touching existing solver code.
#
# Containment rule: StochasticGameProblem, POGameProblem, InverseGameProblem
# each *wrap* a GameProblem via a named field — they do not extend it. This
# keeps FNELQ / FALCON / iLQGames decoupled from extended problem classes.
#
# Hierarchy is intentionally flat: no AbstractFeedbackGame intermediate type.
# The open-loop vs feedback distinction belongs to solver dispatch, not to
# the problem specification.
# ============================================================================

"""
    AbstractGameProblem{T}

Root abstract type for all game problem specifications.

# Type Parameter
- `T` : Numeric type (Float64, etc.)

# Subtype Tree
```
AbstractGameProblem{T}
├── AbstractDeterministicGame{T}
│   └── GameProblem{T}                  ← existing concrete type; flat hierarchy
├── AbstractStochasticGame{T}
│   ├── StochasticGameProblem{T}        ← covariance steering (EAGLE/OSPREY)
│   └── RobustGameProblem{T}            ← distributionally robust / min-max
├── AbstractPartiallyObservableGame{T}
│   └── POGameProblem{T}                ← POMDP-game hybrid (PACK)
└── AbstractInverseGameProblem{T}
    └── InverseGameProblem{T}           ← cost identification (MONGOOSE)
```

# Solver Dispatch Convention
Solvers declare methods on the narrowest abstract type they support:
```julia
# FNELQ handles any deterministic game
_solve(game::AbstractDeterministicGame{T}, solver::FNELQ, ...) where {T}

# MONGOOSE handles inverse problems with any forward model
_solve(game::AbstractInverseGameProblem{T}, solver::MONGOOSE, ...) where {T}
```
"""
abstract type AbstractGameProblem{T} end

"""
    AbstractDeterministicGame{T} <: AbstractGameProblem{T}

Games with fully determined dynamics and well-defined state observations.
`GameProblem{T}` is the concrete subtype for all GNEP variants (LQ, nonlinear,
PD-GNEP, constrained, unconstrained).

Equilibrium concept: Nash equilibrium in feedback or open-loop strategies,
depending on the information structure of the players.
"""
abstract type AbstractDeterministicGame{T} <: AbstractGameProblem{T} end

"""
    AbstractStochasticGame{T} <: AbstractGameProblem{T}

Games with stochastic dynamics or parametric uncertainty. Wraps a deterministic
`GameProblem` via a `nominal_game` field — the stochastic layer augments the
mean dynamics without replacing them.

Equilibrium concepts include expected-cost Nash (risk-neutral),
risk-sensitive Nash, and distributionally robust Nash.
"""
abstract type AbstractStochasticGame{T} <: AbstractGameProblem{T} end

"""
    AbstractPartiallyObservableGame{T} <: AbstractGameProblem{T}

Games in which players do not observe the full state. Strategies map belief
distributions over states to controls. Wraps a `GameProblem` via
`underlying_game` and adds per-player observation models and belief dynamics.
"""
abstract type AbstractPartiallyObservableGame{T} <: AbstractGameProblem{T} end

"""
    AbstractInverseGameProblem{T} <: AbstractGameProblem{T}

Inference problems where cost function parameters are latent and must be
recovered from observed trajectories or strategies. Contains a forward game
model as a field.

This is categorically not a game being *solved* — it is a game being
*identified*. Solvers (MONGOOSE) dispatch on this type.
"""
abstract type AbstractInverseGameProblem{T} <: AbstractGameProblem{T} end

# ============================================================================
# Interface — required methods on AbstractGameProblem
# ============================================================================

"""
    n_players(game::AbstractGameProblem) -> Int

Number of players. Must be implemented by every concrete subtype.
"""
function n_players end

"""
    time_horizon(game::AbstractGameProblem) -> TimeHorizon

Time horizon of the game.
"""
function time_horizon end

# ============================================================================
# Trait queries
# ============================================================================

is_deterministic(::AbstractDeterministicGame) = true
is_deterministic(::AbstractGameProblem)        = false

is_stochastic(::AbstractStochasticGame) = true
is_stochastic(::AbstractGameProblem)    = false

is_partially_observable(::AbstractPartiallyObservableGame) = true
is_partially_observable(::AbstractGameProblem)              = false

is_inverse(::AbstractInverseGameProblem) = true
is_inverse(::AbstractGameProblem)        = false