using LinearAlgebra

# ============================================================================
# gnep_solutions.jl
#
# Defines Trajectory and GNEPSolution.
#
# Include order constraint: must come AFTER abstract/strategy.jl in
# DifferentialGamesBase.jl because GNEPSolution has a field of type
# AbstractStrategy{T}. The current module include order satisfies this:
#   ...strategy.jl (position 12) before gnep_solutions.jl (position 17).
#
# Trajectory is defined here (not in a separate file) because:
#   - It is only meaningful in the context of a game solution
#   - Moving it earlier creates no benefit; GNEPSolution is the only consumer
# ============================================================================

"""
    AbstractSolution{T}

Root abstract type for all game solution objects.
"""
abstract type AbstractSolution{T} end

# ============================================================================
# Trajectory
# ============================================================================

"""
    Trajectory{T}

Single player's state and control trajectory over time.

# Fields
- `player_id::Int`       : Player identifier
- `states::Matrix{T}`    : (n_x × N+1); `states[:, k]` at `times[k]`
- `controls::Matrix{T}`  : (n_u × N); `controls[:, k]` over `[times[k], times[k+1])`
- `times::Vector{T}`     : Length N+1
- `cost::T`              : Total cost for this player

# Notes
For shared-state games (LQ, iLQGames), `states` is the full joint state.
For PD-GNEPs, `states` is player i's private state of dimension nᵢ.
"""
struct Trajectory{T}
    player_id::Int
    states::Matrix{T}
    controls::Matrix{T}
    times::Vector{T}
    cost::T

    function Trajectory{T}(
        player_id::Int,
        states::Matrix{T},
        controls::Matrix{T},
        times::Vector{T},
        cost::T
    ) where {T}
        N = length(times) - 1
        @assert size(states, 2)   == N + 1 "states must have N+1 columns"
        @assert size(controls, 2) == N     "controls must have N columns"
        @assert player_id > 0              "player_id must be positive"
        new{T}(player_id, states, controls, times, cost)
    end
end

Trajectory(player_id::Int, states::Matrix{T}, controls::Matrix{T},
           times::Vector{T}, cost::T) where {T} =
    Trajectory{T}(player_id, states, controls, times, cost)

# ============================================================================
# GNEPSolution
# ============================================================================

"""
    GNEPSolution{T} <: AbstractSolution{T}

Solution to a generalized Nash equilibrium problem (GNEP).

Covers all variants: unconstrained Nash (iLQGames, FNELQ), PD-GNEP (FALCON),
open-loop or feedback equilibria, LQ or nonlinear games.

# Fields
- `game`             : The `GameProblem{T}` that was solved
- `trajectories`     : Per-player `Trajectory{T}` objects
- `state_trajectory` : Shared state matrix (n × N+1) for shared-state games;
  `nothing` for PD-GNEPs where each player has private state
- `strategy`         : `OpenLoopStrategy` (FALCON), `FeedbackStrategy`
  (FNELQ/iLQGames), or `nothing` if the solver did not compute gains
- `equilibrium_type` : One of `:FeedbackNash`, `:OpenLoopNash`,
  `:GeneralizedNash`, `:Approximate`, `:Unknown`
- `converged`        : Whether the solver converged
- `iterations`       : Number of outer iterations taken
- `solve_time`       : Wall-clock seconds
- `solver_info`      : Solver-specific diagnostics
"""
struct GNEPSolution{T} <: AbstractSolution{T}
    game::GameProblem{T}
    trajectories::Vector{Trajectory{T}}
    state_trajectory::Union{Nothing, Matrix{T}}
    strategy::Union{Nothing, AbstractStrategy{T}}
    equilibrium_type::Symbol
    converged::Bool
    iterations::Int
    solve_time::Float64
    solver_info::Dict{Symbol, Any}

    function GNEPSolution{T}(
        game::GameProblem{T},
        trajectories::Vector{Trajectory{T}},
        state_trajectory::Union{Nothing, Matrix{T}},
        strategy::Union{Nothing, AbstractStrategy{T}},
        equilibrium_type::Symbol,
        converged::Bool,
        iterations::Int,
        solve_time::Float64,
        solver_info::Dict{Symbol, Any}
    ) where {T}
        valid = (:FeedbackNash, :OpenLoopNash, :GeneralizedNash, :Approximate, :Unknown)
        @assert(equilibrium_type in valid,
            "equilibrium_type must be one of $valid, got :$equilibrium_type")
        @assert(length(trajectories) == num_players(game),
            "Must have one trajectory per player")
        @assert(allunique(t.player_id for t in trajectories),
            "Duplicate player IDs in trajectories")

        traj_ids = Set(t.player_id for t in trajectories)
        game_ids = Set(obj.player_id for obj in game.objectives)
        @assert(traj_ids == game_ids,
            "Trajectory player IDs must match game objectives")

        if strategy !== nothing
            @assert(n_players(strategy) == num_players(game),
                "Strategy n_players must match game n_players")
        end

        new{T}(game, trajectories, state_trajectory, strategy,
               equilibrium_type, converged, iterations, solve_time, solver_info)
    end
end

"""
    GNEPSolution(game, trajectories; kwargs...) -> GNEPSolution{T}

Keyword constructor. All keyword arguments have defaults so existing call
sites that don't use strategy/state_trajectory are unaffected.
"""
function GNEPSolution(
    game::GameProblem{T},
    trajectories::Vector{Trajectory{T}};
    state_trajectory::Union{Nothing, Matrix{T}}   = nothing,
    strategy::Union{Nothing, AbstractStrategy{T}} = nothing,
    equilibrium_type::Symbol                      = :Unknown,
    converged::Bool                               = true,
    iterations::Int                               = 0,
    solve_time::Float64                           = 0.0,
    solver_info::Dict{Symbol, Any}                = Dict{Symbol, Any}()
) where {T}
    GNEPSolution{T}(
        game, trajectories, state_trajectory, strategy,
        equilibrium_type, converged, iterations, solve_time, solver_info
    )
end

# ============================================================================
# Trait queries
# ============================================================================

has_strategy(sol::GNEPSolution)       = sol.strategy !== nothing
has_shared_state(sol::GNEPSolution)   = sol.state_trajectory !== nothing
is_feedback(sol::GNEPSolution)        = sol.equilibrium_type == :FeedbackNash
is_open_loop_solution(sol::GNEPSolution) = sol.equilibrium_type == :OpenLoopNash

# ============================================================================
# Accessors
# ============================================================================

function get_trajectory(sol::GNEPSolution, player_id::Int)
    idx = findfirst(t -> t.player_id == player_id, sol.trajectories)
    isnothing(idx) && error("No trajectory for player $player_id")
    return sol.trajectories[idx]
end

get_cost(sol::GNEPSolution, player_id::Int) = get_trajectory(sol, player_id).cost

function get_costs(sol::GNEPSolution{T}) where {T}
    sorted = sort(sol.trajectories, by = t -> t.player_id)
    return T[t.cost for t in sorted]
end

function get_strategy(sol::GNEPSolution)
    isnothing(sol.strategy) && error(
        "No strategy in solution. Check has_strategy(sol) before calling get_strategy."
    )
    return sol.strategy
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, sol::GNEPSolution{T}) where {T}
    status = sol.converged ? "CONVERGED" : "NOT CONVERGED"
    print(io, "GNEPSolution{$T} [$(num_players(sol.game)) players, ",
          ":$(sol.equilibrium_type), $status]")
end

function Base.show(io::IO, ::MIME"text/plain", sol::GNEPSolution{T}) where {T}
    println(io, "GNEPSolution{$T}")
    println(io, "  Players          : ", num_players(sol.game))
    println(io, "  Equilibrium      : ", sol.equilibrium_type)
    println(io, "  Converged        : ", sol.converged)
    println(io, "  Iterations       : ", sol.iterations)
    println(io, "  Solve time       : ", round(sol.solve_time, digits=4), "s")
    println(io, "  Has strategy     : ", has_strategy(sol))
    println(io, "  Strategy type    : ",
            has_strategy(sol) ? typeof(sol.strategy) : "—")
    println(io, "  Has shared state : ", has_shared_state(sol))
    println(io, "  Player costs:")
    for traj in sort(sol.trajectories, by = t -> t.player_id)
        println(io, "    Player $(traj.player_id): ", round(traj.cost, digits=6))
    end
end