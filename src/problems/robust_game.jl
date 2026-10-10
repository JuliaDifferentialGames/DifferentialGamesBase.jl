# ============================================================================
# problems/robust_game.jl
#
# Concrete AbstractStochasticGame subtype (per the abstract hierarchy's
# documented intent — StochasticGameProblem and RobustGameProblem share the
# same abstract parent): bounded, non-probabilistic disturbance on top of a
# deterministic nominal game. Wraps rather than duplicates.
# ============================================================================

"""
    DisturbanceSet{T}

Box bound on an additive disturbance: `w ∈ [w_min, w_max]` (component-wise),
worst-case rather than distributional. This is what tube-based/min-max solvers
(e.g. `DynamicTubeMPC` in DifferentialGamesBaseSolvers.jl) need — a set, not a
covariance.

# Fields
- `w_min::Vector{T}`
- `w_max::Vector{T}`
"""
struct DisturbanceSet{T}
    w_min::Vector{T}
    w_max::Vector{T}

    function DisturbanceSet{T}(w_min::Vector{T}, w_max::Vector{T}) where {T}
        @assert length(w_min) == length(w_max) "w_min/w_max length mismatch"
        @assert all(w_min .<= w_max) "w_min must be componentwise ≤ w_max"
        new{T}(w_min, w_max)
    end
end

DisturbanceSet(w_min::Vector{T}, w_max::Vector{T}) where {T} = DisturbanceSet{T}(w_min, w_max)

Base.length(d::DisturbanceSet) = length(d.w_min)

"""
    RobustGameProblem{T} <: AbstractStochasticGame{T}

A deterministic `nominal_game` subjected to a bounded (non-probabilistic)
additive disturbance on the joint dynamics.

# Fields
- `nominal_game::AbstractDeterministicGame{T}`
- `disturbance::DisturbanceSet{T}`
"""
struct RobustGameProblem{T} <: AbstractStochasticGame{T}
    nominal_game::AbstractDeterministicGame{T}
    disturbance::DisturbanceSet{T}
end

n_players(g::RobustGameProblem) = n_players(g.nominal_game)
time_horizon(g::RobustGameProblem) = time_horizon(g.nominal_game)

function Base.show(io::IO, g::RobustGameProblem{T}) where {T}
    print(io, "RobustGameProblem{$T} wrapping ", g.nominal_game,
          " [disturbance dim=", length(g.disturbance), "]")
end
