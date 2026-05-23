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

"""
    AbstractPotentialGame{T} <: AbstractDeterministicGame{T}

A deterministic game that admits a potential function Φ such that any
minimizer of Φ over the joint decision space is a variational Nash
equilibrium (v-NE) of the game.

This is used by FALCON's inner solve: the convexified subproblem (Problem 3)
has a strongly-convex potential Φ_k (Eq. 17), whose unique minimizer is the
v-NE of the convexified game (Lemma V.3).  Solvers dispatch on this type to
exploit the joint-minimization structure.

Subtypes must implement the standard `AbstractGameProblem` interface.
`ConvexifiedGame{T}` in the FALCON package subtypes this.
"""
abstract type AbstractPotentialGame{T} <: AbstractDeterministicGame{T} end

is_potential_game(::AbstractPotentialGame) = true
is_potential_game(::AbstractGameProblem)   = false

"""
    AbstractStatePotentialGame{T} <: AbstractPotentialGame{T}

State-based potential game (Marden 2012, Def 3.2). Extends potential games to include
an underlying finite state space X with Markovian transition P: A×X → Δ(X).

The potential function φ: A×X → ℝ satisfies for all i, a'ᵢ, a, x:
  (i)  Uᵢ(a'ᵢ, a₋ᵢ, x) − Uᵢ(a, x) = φ(a'ᵢ, a₋ᵢ, x) − φ(a, x)
  (ii) For every x' in support of P(a, x): φ(a, x') ≥ φ(a, x)

A recurrent state equilibrium is guaranteed to exist at any maximizer of φ.
"""
abstract type AbstractStatePotentialGame{T} <: AbstractPotentialGame{T} end

is_state_potential(::AbstractStatePotentialGame) = true
is_state_potential(::AbstractGameProblem)        = false

"""
    AbstractOrdinalPotentialGame{T} <: AbstractDeterministicGame{T}

Ordinal state-based potential game (Marden 2012, Section 3.3). A relaxation of
`AbstractStatePotentialGame` where the potential function only needs to *preserve
the sign* of unilateral utility changes:

    Uᵢ(a'ᵢ, a₋ᵢ, x) − Uᵢ(a, x) > 0  ⟹  φ(a'ᵢ, a₋ᵢ, x) − φ(a, x) > 0

A recurrent state equilibrium is guaranteed to exist (Lemma 3.1 in Marden 2012).
Note: this is strictly more general than `AbstractStatePotentialGame`; exact
potential games satisfy the ordinal condition but not vice-versa.
"""
abstract type AbstractOrdinalPotentialGame{T} <: AbstractDeterministicGame{T} end

is_ordinal_potential(::AbstractOrdinalPotentialGame) = true
is_ordinal_potential(::AbstractGameProblem)          = false

"""
    AbstractLexicographicGame{T} <: AbstractOrdinalPotentialGame{T}

Lexicographic general sum game (Miller & Mitra 2022, Def 1). Each agent i has a
two-component cost function Jᵢ: Z → ℝ² ordered lexicographically (≼):

    Jᵢ(z) = (Jᵢᶜᵒˡ(z), Jᵢᵖᵉʳ(z))

where Jᵢᶜᵒˡ is a shared pairwise collision cost and Jᵢᵖᵉʳ is an individual personal
cost. Any LG is an ordinal potential game (Proposition 1, Miller & Mitra 2022) with
ordinal potential P(z) = ⟨½ Σⱼ Jⱼᶜᵒˡ(z), Σⱼ gⱼ(zⱼ)⟩.
"""
abstract type AbstractLexicographicGame{T} <: AbstractOrdinalPotentialGame{T} end

is_lexicographic(::AbstractLexicographicGame) = true
is_lexicographic(::AbstractGameProblem)       = false

"""
    AbstractConvexGame{T} <: AbstractDeterministicGame{T}

An N-player deterministic game in which:
  (i)  Each player i's objective Jᵢ(xᵢ; x₋ᵢ) is convex in xᵢ for every fixed x₋ᵢ.
  (ii) Every player's feasible set (private and shared constraints) is convex.

These two conditions together guarantee:
- **Existence** of a Nash equilibrium via Kakutani's fixed-point theorem.
- A **variational inequality** (VI) reformulation: x* is a NE iff
      F(x*)ᵀ(x − x*) ≥ 0  for all x ∈ X,
  where F(x) = (∇ₓ₁J₁(x), …, ∇ₓₙJₙ(x)) is the pseudo-gradient.
- **Uniqueness** of the NE when `is_strictly_convex_game` holds
  (Rosen 1965, diagonal strict convexity).

Note: convexity and potential-game structure are independent — a game may have
both, either, or neither property.

# References
Rosen, J.B. (1965). Existence and uniqueness of equilibrium points for concave
N-person games. *Econometrica* 33(3), 520–534.
"""
abstract type AbstractConvexGame{T} <: AbstractDeterministicGame{T} end

is_convex_game(::AbstractConvexGame)          = true
is_convex_game(::AbstractGameProblem)         = false
is_strictly_convex_game(::AbstractGameProblem) = false

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