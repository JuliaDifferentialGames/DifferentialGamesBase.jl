using LinearAlgebra

# ============================================================================
# AbstractInformationStructure hierarchy
#
# What each player observes and when. Carried as a field on Player{T}.
# The DifferentialGame constructor inspects information structures across
# all players to determine the appropriate AbstractGameProblem subtype.
#
# Mathematical grounding (Başar & Olsder, Ch. 5):
#   Perfect state:  ηᵢ(t) = x(t)               — enables feedback Nash
#   Open loop:      ηᵢ(t) = x(0)               — open-loop Nash
#   Private obs:    ηᵢ(t) = {yᵢ(s) : s ≤ t}   — POMDP / belief-feedback
#   Shared obs:     ηᵢ(t) = {y(s)  : s ≤ t}   — common knowledge observation
#   Asymmetric:     leader commits first        — Stackelberg
#
# Design note:
#   Observation models are stored as fields (not type parameters) so that
#   players with structurally similar but distinct observation models share
#   the same concrete type. This simplifies dispatch in POGameProblem and
#   avoids an explosion of specializations at game-construction time.
# ============================================================================

"""
    AbstractInformationStructure

Abstract type representing what a single player observes. Carried as a field
on `Player{T}`. Determines the strategy space and which `AbstractGameProblem`
subtype `DifferentialGame` constructs.

# Concrete Subtypes
- `PerfectStateInformation` — player observes full joint state x(t)
- `OpenLoopInformation`     — player observes only initial state x(0)
- `PrivateObservation`      — player observes yᵢ(t) = hᵢ(x, vᵢ, p, t)
- `SharedObservation`       — all players share a common observation y(t)
- `AsymmetricInformation`   — Stackelberg leader–follower structure
"""
abstract type AbstractInformationStructure end

"""
    PerfectStateInformation <: AbstractInformationStructure

Player observes the full joint state `x(t)` at each time. Enables feedback
(closed-loop) strategies `γᵢ : t × ℝⁿ → ℝᵐⁱ`.

Default for all deterministic GNEPs. Both FALCON (open-loop output, feedback
structure) and iLQGames (feedback Nash) operate under this information structure.
"""
struct PerfectStateInformation <: AbstractInformationStructure end

"""
    OpenLoopInformation <: AbstractInformationStructure

Player observes only `x(0)` and commits to a control trajectory at time 0.
Strategies are functions of time only: `γᵢ : [0,T] → ℝᵐⁱ`.

FALCON's open-loop Nash output corresponds to this information structure even
though FALCON internally uses feedback during warm-starting.
"""
struct OpenLoopInformation <: AbstractInformationStructure end

"""
    PrivateObservation <: AbstractInformationStructure

Player observes `yᵢ(t) = hᵢ(x(t), vᵢ(t), p, t)` where `vᵢ` is private
observation noise. Strategies map belief histories to controls.

Triggers construction of `POGameProblem` in `DifferentialGame`.

# Fields
- `observation_model::Function` : `hᵢ(x, v, p, t) -> yᵢ`; must be
  ForwardDiff-compatible for filter Jacobian computation in belief updates.
- `obs_dim::Int` : Dimension of the observation vector `yᵢ`
- `noise_cov::Matrix{Float64}` : Observation noise covariance (obs_dim × obs_dim),
  symmetric positive definite.

# ForwardDiff Compatibility
Avoid type restrictions in `observation_model`:
```julia
# Bad  — breaks AD through the filter
h(x::Vector{Float64}, v, p, t) = ...
# Good — AD-compatible
h(x::AbstractVector, v, p, t) = ...
```
"""
struct PrivateObservation <: AbstractInformationStructure
    observation_model::Function
    obs_dim::Int
    noise_cov::Matrix{Float64}

    function PrivateObservation(
        model::Function,
        obs_dim::Int,
        noise_cov::Matrix{Float64}
    )
        @assert obs_dim > 0 "Observation dimension must be positive"
        @assert size(noise_cov) == (obs_dim, obs_dim) "noise_cov must be (obs_dim × obs_dim)"
        @assert issymmetric(noise_cov) "noise_cov must be symmetric"
        @assert isposdef(noise_cov) "noise_cov must be positive definite"
        new(model, obs_dim, noise_cov)
    end
end

"""
    SharedObservation <: AbstractInformationStructure

All players observe the same signal `y(t) = h(x(t), v(t), p, t)`. The
observation is common knowledge — each player knows what every other player
observed. Intermediate between `PerfectStateInformation` and `PrivateObservation`.

# Fields
- `observation_model::Function` : `h(x, v, p, t) -> y`; ForwardDiff-compatible
- `obs_dim::Int` : Dimension of the shared observation `y`
- `noise_cov::Matrix{Float64}` : Observation noise covariance (obs_dim × obs_dim)
"""
struct SharedObservation <: AbstractInformationStructure
    observation_model::Function
    obs_dim::Int
    noise_cov::Matrix{Float64}

    function SharedObservation(
        model::Function,
        obs_dim::Int,
        noise_cov::Matrix{Float64}
    )
        @assert obs_dim > 0 "Observation dimension must be positive"
        @assert size(noise_cov) == (obs_dim, obs_dim) "noise_cov must be (obs_dim × obs_dim)"
        @assert issymmetric(noise_cov) "noise_cov must be symmetric"
        @assert isposdef(noise_cov) "noise_cov must be positive definite"
        new(model, obs_dim, noise_cov)
    end
end

"""
    AsymmetricInformation <: AbstractInformationStructure

Stackelberg leader–follower information structure. The leader commits to a
strategy first; followers observe the committed strategy before choosing their
own responses.

Currently a placeholder — full Stackelberg solver support is future work.

# Field
- `leader_id::Int` : Player ID of the Stackelberg leader (1-based)
"""
struct AsymmetricInformation <: AbstractInformationStructure
    leader_id::Int

    function AsymmetricInformation(leader_id::Int)
        @assert leader_id > 0 "Leader player ID must be positive"
        new(leader_id)
    end
end

# ============================================================================
# Trait queries
# ============================================================================

"""
    requires_belief_state(info::AbstractInformationStructure) -> Bool

True if strategies must map beliefs to controls rather than states directly.
"""
requires_belief_state(::PerfectStateInformation) = false
requires_belief_state(::OpenLoopInformation)     = false
requires_belief_state(::PrivateObservation)      = true
requires_belief_state(::SharedObservation)       = false
requires_belief_state(::AsymmetricInformation)   = false

"""
    is_feedback_compatible(info::AbstractInformationStructure) -> Bool

True if feedback Nash strategies are well-defined under this information structure.
"""
is_feedback_compatible(::PerfectStateInformation) = true
is_feedback_compatible(::OpenLoopInformation)     = false
is_feedback_compatible(::PrivateObservation)      = true   # Belief-feedback
is_feedback_compatible(::SharedObservation)       = true
is_feedback_compatible(::AsymmetricInformation)   = false  # Stackelberg

"""
    is_open_loop(info::AbstractInformationStructure) -> Bool
"""
is_open_loop(::OpenLoopInformation)          = true
is_open_loop(::AbstractInformationStructure) = false

# ============================================================================
# Game class inference — used by DifferentialGame constructor
# ============================================================================

"""
    _infer_game_class(infos) -> Symbol

Given the information structures of all players, return a symbol indicating
which `AbstractGameProblem` subtype `DifferentialGame` should construct.

Called internally by `DifferentialGame`; not part of the public API.

Stackelberg takes priority over partial observability when both appear.

# Returns
- `:deterministic`        — all `PerfectStateInformation` or `OpenLoopInformation`
- `:partially_observable` — any `PrivateObservation` or `SharedObservation`
- `:stackelberg`          — any `AsymmetricInformation`
"""
function _infer_game_class(infos::AbstractVector{<:AbstractInformationStructure})
    if any(i -> i isa AsymmetricInformation, infos)
        return :stackelberg
    elseif any(i -> i isa Union{PrivateObservation, SharedObservation}, infos)
        return :partially_observable
    else
        return :deterministic
    end
end