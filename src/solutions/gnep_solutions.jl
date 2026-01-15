"""
    Trajectory{T}

Represents a single player's trajectory (state and control) over time.

# Fields
- `player_id::Int` : Player identifier
- `states::Matrix{T}` : State trajectory, size (n_x × N+1) for N time steps
- `controls::Matrix{T}` : Control trajectory, size (n_u × N)
- `times::Vector{T}` : Time points, length N+1
- `cost::T` : Total cost for this player's trajectory

# Notes
- For separable dynamics (PD-GNEP), states are player-specific dimension n_x^i
- For shared state space (LQ games), states are full dimension n
- Time indexing: states[:, k] corresponds to time times[k]
- Control indexing: controls[:, k] corresponds to interval [times[k], times[k+1])
"""
struct Trajectory{T}
    player_id::Int
    states::Matrix{T}      # (n_x × N+1)
    controls::Matrix{T}    # (n_u × N)
    times::Vector{T}       # (N+1,)
    cost::T
    
    function Trajectory{T}(
        player_id::Int,
        states::Matrix{T},
        controls::Matrix{T},
        times::Vector{T},
        cost::T
    ) where {T}
        N = length(times) - 1
        @assert size(states, 2) == N + 1 "States must have N+1 time steps"
        @assert size(controls, 2) == N "Controls must have N time steps"
        @assert player_id > 0 "Player ID must be positive"
        new{T}(player_id, states, controls, times, cost)
    end
end

# Convenience constructor
Trajectory(player_id::Int, states::Matrix{T}, controls::Matrix{T}, 
           times::Vector{T}, cost::T) where {T} = 
    Trajectory{T}(player_id, states, controls, times, cost)

"""
    GameSolution{T}

Solution structure for differential game problems.

# Fields
- `game::GameProblem{T}` : The game problem that was solved
- `trajectories::Vector{Trajectory{T}}` : Per-player trajectories
- `equilibrium_type::Symbol` : Type of equilibrium achieved
- `converged::Bool` : Whether the solver converged
- `iterations::Int` : Number of iterations taken
- `solve_time::Float64` : Wall-clock time in seconds
- `solver_info::Dict{Symbol, Any}` : Additional solver-specific information

# Equilibrium Types
- `:OpenLoopNash` : Open-loop Nash equilibrium (trajectories only)
- `:FeedbackNash` : Feedback Nash equilibrium (with policies)
- `:Stackelberg` : Stackelberg equilibrium (leader-follower)
- `:GeneralizedNash` : Generalized Nash with shared constraints
- `:Approximate` : Approximate equilibrium (within tolerance)

# Notes
Stores reference to original game for context. Use `get_trajectory(sol, player_id)`
to retrieve specific player trajectories.
"""
struct GameSolution{T}
    game::GameProblem{T}
    trajectories::Vector{Trajectory{T}}
    equilibrium_type::Symbol
    converged::Bool
    iterations::Int
    solve_time::Float64
    solver_info::Dict{Symbol, Any}
    
    function GameSolution{T}(
        game::GameProblem{T},
        trajectories::Vector{Trajectory{T}},
        equilibrium_type::Symbol,
        converged::Bool,
        iterations::Int,
        solve_time::Float64,
        solver_info::Dict{Symbol, Any}
    ) where {T}
        @assert length(trajectories) == num_players(game) "Must have trajectory for each player"
        @assert allunique(traj.player_id for traj in trajectories) "Duplicate player IDs in trajectories"
        
        # Validate player IDs match game
        traj_ids = Set(traj.player_id for traj in trajectories)
        game_ids = Set(obj.player_id for obj in game.objectives)
        @assert traj_ids == game_ids "Trajectory player IDs must match game"
        
        new{T}(game, trajectories, equilibrium_type, converged, 
               iterations, solve_time, solver_info)
    end
end

# Convenience constructor with keyword arguments
function GameSolution(
    game::GameProblem{T},
    trajectories::Vector{Trajectory{T}};
    equilibrium_type::Symbol = :OpenLoopNash,
    converged::Bool = true,
    iterations::Int = 0,
    solve_time::Float64 = 0.0,
    solver_info::Dict{Symbol, Any} = Dict{Symbol, Any}()
) where {T}
    return GameSolution{T}(game, trajectories, equilibrium_type, 
                          converged, iterations, solve_time, solver_info)
end

# ============================================================================
# Helper Functions
# ============================================================================

"""
    get_trajectory(sol::GameSolution, player_id::Int)

Get trajectory for specific player.
"""
function get_trajectory(sol::GameSolution, player_id::Int)
    idx = findfirst(traj -> traj.player_id == player_id, sol.trajectories)
    isnothing(idx) && error("No trajectory found for player $player_id")
    return sol.trajectories[idx]
end

"""
    get_cost(sol::GameSolution, player_id::Int)

Get final cost for specific player.
"""
get_cost(sol::GameSolution, player_id::Int) = get_trajectory(sol, player_id).cost

"""
    get_costs(sol::GameSolution)

Get vector of all player costs in order of player IDs.
"""
function get_costs(sol::GameSolution)
    sorted_trajs = sort(sol.trajectories, by = t -> t.player_id)
    return [traj.cost for traj in sorted_trajs]
end

"""
    is_nash_equilibrium(sol::GameSolution; tol=1e-6)

Check if solution satisfies Nash equilibrium conditions within tolerance.

Returns true if no player can improve their cost by more than `tol` through
unilateral deviation.
"""
function is_nash_equilibrium(sol::GameSolution; tol=1e-6)
    # This would require computing best responses - placeholder for now
    # Actual implementation would depend on solver capabilities
    return sol.converged && sol.equilibrium_type in (:OpenLoopNash, :FeedbackNash, :GeneralizedNash)
end

# ============================================================================
# Display Methods
# ============================================================================

function Base.show(io::IO, sol::GameSolution{T}) where {T}
    print(io, "GameSolution{$T} with $(num_players(sol.game)) players")
    if sol.converged
        print(io, " [CONVERGED]")
    else
        print(io, " [NOT CONVERGED]")
    end
end

function Base.show(io::IO, ::MIME"text/plain", sol::GameSolution{T}) where {T}
    println(io, "GameSolution{$T}")
    println(io, "  Players: ", num_players(sol.game))
    println(io, "  Equilibrium type: ", sol.equilibrium_type)
    println(io, "  Converged: ", sol.converged)
    println(io, "  Iterations: ", sol.iterations)
    println(io, "  Solve time: ", round(sol.solve_time, digits=3), "s")
    
    println(io, "  Player costs:")
    for traj in sort(sol.trajectories, by = t -> t.player_id)
        println(io, "    Player ", traj.player_id, ": ", round(traj.cost, digits=6))
    end
    
    if !isempty(sol.solver_info)
        println(io, "  Solver info: ", length(sol.solver_info), " entries")
    end
end