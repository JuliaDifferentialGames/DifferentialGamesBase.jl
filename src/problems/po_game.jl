# ============================================================================
# problems/po_game.jl
#
# Concrete AbstractPartiallyObservableGame subtype. Wraps an
# `underlying_game` and reuses the existing `PrivateObservation`/
# `SharedObservation` information-structure types directly — no duplicate
# observation-model type.
# ============================================================================

"""
    POGameProblem{T} <: AbstractPartiallyObservableGame{T}

A deterministic `underlying_game` in which each player observes according to
its own `PrivateObservation` or `SharedObservation` (reused from
`information_structure.jl`, not duplicated).

# Fields
- `underlying_game::AbstractDeterministicGame{T}`
- `observation_models::Vector{<:Union{PrivateObservation, SharedObservation}}` :
  one entry per player, in player-index order.
"""
struct POGameProblem{T} <: AbstractPartiallyObservableGame{T}
    underlying_game::AbstractDeterministicGame{T}
    observation_models::Vector{<:Union{PrivateObservation, SharedObservation}}

    function POGameProblem{T}(
        underlying_game::AbstractDeterministicGame{T},
        observation_models::Vector{<:Union{PrivateObservation, SharedObservation}}
    ) where {T}
        @assert(length(observation_models) == n_players(underlying_game),
            "Need one observation model per player, got $(length(observation_models)) " *
            "for $(n_players(underlying_game)) players")
        new{T}(underlying_game, observation_models)
    end
end

POGameProblem(
    underlying_game::AbstractDeterministicGame{T},
    observation_models::Vector{<:Union{PrivateObservation, SharedObservation}}
) where {T} = POGameProblem{T}(underlying_game, observation_models)

n_players(g::POGameProblem) = n_players(g.underlying_game)
time_horizon(g::POGameProblem) = time_horizon(g.underlying_game)

function Base.show(io::IO, g::POGameProblem{T}) where {T}
    print(io, "POGameProblem{$T} wrapping ", g.underlying_game,
          " [", n_players(g), " observation models]")
end
