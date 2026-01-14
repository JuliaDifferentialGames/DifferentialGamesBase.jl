using LinearAlgebra

# ============================================================================
# ABSTRACT TYPE INTERFACE
# ============================================================================

"""
    num_players(game::AbstractGNEP)

Return the number of players in the game.

# Implementations required for subtypes
Must be implemented for each concrete GNEP type.
"""
function num_players end

"""
    state_dim(game::AbstractGNEP)

Return the total state dimension of the game.
"""
function state_dim end

"""
    state_dim(game::AbstractGNEP, player::Int)

Return the state dimension for a specific player.
"""
function state_dim(game::AbstractGNEP, player::Int)
    error("state_dim(game, player) not implemented for $(typeof(game))")
end

"""
    control_dim(game::AbstractGNEP)

Return the total control dimension of the game.
"""
function control_dim end

"""
    control_dim(game::AbstractGNEP, player::Int)

Return the control dimension for a specific player.
"""
function control_dim(game::AbstractGNEP, player::Int)
    error("control_dim(game, player) not implemented for $(typeof(game))")
end

"""
    time_horizon(game::AbstractGNEP)

Return the time horizon of the game.
"""
time_horizon(game::AbstractGNEP) = game.tf

"""
    initial_state(game::AbstractGNEP)

Return the initial state of the game (concatenated across all players).
"""
initial_state(game::AbstractGNEP) = game.x0

# ============================================================================
# DYNAMICS EVALUATION
# ============================================================================

"""
    evaluate_dynamics!(ẋ, game::AbstractGNEP, x, u, t)

Evaluate the dynamics of the game at state x, control u, and time t.
Modifies ẋ in-place.

# Arguments
- `ẋ::AbstractVector` : Output vector for state derivative (length n)
- `game::AbstractGNEP` : Game problem
- `x::AbstractVector` : Current state (length n)
- `u::AbstractVector` : Current control (length m)
- `t::Real` : Current time

# Notes
For coupled dynamics (standard GNEP): Evaluates single dynamics function
For separable dynamics (PD-GNEP): Evaluates per-player dynamics and concatenates
"""
function evaluate_dynamics! end

"""
    evaluate_dynamics(game::AbstractGNEP, x, u, t)

Non-mutating version of evaluate_dynamics!. Returns ẋ.
"""
function evaluate_dynamics(game::AbstractGNEP, x::AbstractVector, u::AbstractVector, t::Real)
    ẋ = similar(x)
    evaluate_dynamics!(ẋ, game, x, u, t)
    return ẋ
end

# Implementation for GNEProblem (coupled dynamics)
function evaluate_dynamics!(
    ẋ::AbstractVector, 
    game::GNEProblem, 
    x::AbstractVector, 
    u::AbstractVector, 
    t::Real
)
    ẋ .= game.dynamics(t, x, u)
    return ẋ
end

# Implementation for PDGNEProblem (separable dynamics)
function evaluate_dynamics!(
    ẋ::AbstractVector, 
    game::PDGNEProblem{T, NP}, 
    x::AbstractVector, 
    u::AbstractVector, 
    t::Real
) where {T, NP}
    X = split_state(game, x)
    U = split_control(game, u)
    
    for i in 1:NP
        player = game.players[i]
        indices = state_indices(game, i)
        ẋ[indices] = player.dynamics(X[i], U[i], player.parameters, t)
    end
    
    return ẋ
end

# Implementation for LQGameProblem (linear dynamics)
function evaluate_dynamics!(
    ẋ::AbstractVector, 
    game::LQGameProblem, 
    x::AbstractVector, 
    u::AbstractVector, 
    t::Real
)
    # ẋ = Ax + Σᵢ Bᵢuᵢ
    mul!(ẋ, game.A, x)
    
    offset = 0
    for i in 1:num_players(game)
        m_i = game.control_dims[i]
        u_i = view(u, offset+1:offset+m_i)
        mul!(ẋ, game.B[i], u_i, 1.0, 1.0)  # ẋ += Bᵢuᵢ
        offset += m_i
    end
    
    return ẋ
end

# ============================================================================
# COST EVALUATION
# ============================================================================

"""
    evaluate_running_cost(game::AbstractGNEP, player::Int, x, u, t)

Evaluate the running cost for a specific player at state x, control u, time t.

# Arguments
- `game::AbstractGNEP` : Game problem
- `player::Int` : Player index
- `x::AbstractVector` : Current state (full state for all players)
- `u::AbstractVector` : Current control (full control for all players)
- `t::Real` : Current time

# Returns
- Scalar running cost Lⁱ(x, u, t) for player i
"""
function evaluate_running_cost end

"""
    evaluate_terminal_cost(game::AbstractGNEP, player::Int, x)

Evaluate the terminal cost for a specific player at final state x.

# Arguments
- `game::AbstractGNEP` : Game problem
- `player::Int` : Player index
- `x::AbstractVector` : Terminal state

# Returns
- Scalar terminal cost Φⁱ(x) for player i
"""
function evaluate_terminal_cost end

# Implementation for GNEProblem
function evaluate_running_cost(
    game::GNEProblem, 
    player::Int, 
    x::AbstractVector, 
    u::AbstractVector, 
    t::Real
)
    return game.running_costs[player](t, x, u)
end

function evaluate_terminal_cost(
    game::GNEProblem, 
    player::Int, 
    x::AbstractVector
)
    return game.terminal_costs[player](x)
end

# Implementation for PDGNEProblem
function evaluate_running_cost(
    game::PDGNEProblem, 
    player::Int, 
    x::AbstractVector, 
    u::AbstractVector, 
    t::Real
)
    X = split_state(game, x)
    U = split_control(game, u)
    p = game.players[player]
    return p.running_cost(X, U[player], p.parameters, t)
end

function evaluate_terminal_cost(
    game::PDGNEProblem, 
    player::Int, 
    x::AbstractVector
)
    X = split_state(game, x)
    p = game.players[player]
    return p.terminal_cost(X)
end

# Implementation for LQGameProblem
function evaluate_running_cost(
    game::LQGameProblem, 
    player::Int, 
    x::AbstractVector, 
    u::AbstractVector, 
    t::Real
)
    # Lⁱ = xᵀQᵢx + uᵢᵀRᵢuᵢ
    offset = sum(game.control_dims[1:player-1]; init=0)
    m_i = game.control_dims[player]
    u_i = view(u, offset+1:offset+m_i)
    
    cost = dot(x, game.Q[player], x)
    cost += dot(u_i, game.R[player], u_i)
    return cost
end

function evaluate_terminal_cost(
    game::LQGameProblem, 
    player::Int, 
    x::AbstractVector
)
    # Φⁱ = xᵀQfᵢx
    return dot(x, game.Qf[player], x)
end

# Implementation for PotentialGameProblem
function evaluate_running_cost(
    game::PotentialGameProblem, 
    player::Int, 
    x::AbstractVector, 
    u::AbstractVector, 
    t::Real
)
    # In potential games, all players share the potential function
    # This returns the potential (same for all players)
    return game.running_potential(t, x, u)
end

function evaluate_terminal_cost(
    game::PotentialGameProblem, 
    player::Int, 
    x::AbstractVector
)
    return game.terminal_potential(x)
end

"""
    evaluate_total_cost(game::AbstractGNEP, player::Int, trajectory::Trajectory)

Evaluate the total cost for a player given a trajectory.

# Arguments
- `game::AbstractGNEP` : Game problem
- `player::Int` : Player index
- `trajectory::Trajectory` : State and control trajectory

# Returns
- Total cost Jⁱ = ∫ Lⁱ(x,u,t)dt + Φⁱ(x(tf)) for player i
"""
function evaluate_total_cost(
    game::AbstractGNEP, 
    player::Int, 
    trajectory::Trajectory{T}
) where T
    n_steps = length(trajectory.times)
    dt = trajectory.times[2] - trajectory.times[1]  # Assume uniform spacing
    
    # Integral cost (trapezoidal rule)
    running_cost = zero(T)
    for k in 1:n_steps
        x_k = trajectory.states[:, k]
        u_k = trajectory.controls[:, k]
        t_k = trajectory.times[k]
        L_k = evaluate_running_cost(game, player, x_k, u_k, t_k)
        
        # Trapezoidal weights
        weight = (k == 1 || k == n_steps) ? 0.5 : 1.0
        running_cost += weight * L_k * dt
    end
    
    # Terminal cost
    x_f = trajectory.states[:, end]
    terminal_cost = evaluate_terminal_cost(game, player, x_f)
    
    return running_cost + terminal_cost
end

# ============================================================================
# CONSTRAINT EVALUATION
# ============================================================================

"""
    evaluate_constraints(game::AbstractGNEP, x, u, t)

Evaluate all constraints at state x, control u, time t.

# Returns
- `Vector` : Concatenated constraint violations [private..., shared...]
"""
function evaluate_constraints end

"""
    has_constraints(game::AbstractGNEP)

Check if the game has any constraints (private or shared).
"""
function has_constraints(game::AbstractGNEP)
    return !isempty(game.constraints)
end

function has_constraints(game::PDGNEProblem)
    has_private = any(p -> !isempty(p.private_constraints), game.players)
    has_shared = !isempty(game.shared_constraints)
    return has_private || has_shared
end

"""
    count_constraints(game::AbstractGNEP)

Count the total number of constraint dimensions.

# Returns
- `Int` : Total constraint dimension
"""
function count_constraints(game::GNEProblem)
    return isempty(game.constraints) ? 0 : sum(length(c) for c in game.constraints)
end

function count_constraints(game::PDGNEProblem)
    constraints = count_total_constraints(game)
    return constraints.total
end

function count_constraints(game::LQGameProblem)
    return 0  # LQ games typically don't have explicit constraints (handled via KKT)
end

# ============================================================================
# TRAJECTORY UTILITIES
# ============================================================================

"""
    create_zero_trajectory(game::AbstractGNEP{T}, n_steps::Int) where T

Create a zero-initialized trajectory structure.

# Arguments
- `game::AbstractGNEP` : Game problem
- `n_steps::Int` : Number of time steps

# Returns
- `Trajectory{T}` : Zero-initialized trajectory
"""
function create_zero_trajectory(game::AbstractGNEP{T}, n_steps::Int) where T
    n = state_dim(game)
    m = control_dim(game)
    
    states = zeros(T, n, n_steps)
    controls = zeros(T, m, n_steps)
    times = range(zero(T), time_horizon(game), length=n_steps)
    
    return Trajectory{T}(states, controls, collect(times), zero(T))
end

"""
    create_trajectory_from_vectors(
        game::AbstractGNEP{T},
        X::Vector{Vector{T}},
        U::Vector{Vector{T}},
        times::Vector{T}
    ) where T

Create a trajectory from vectors of states and controls.

# Arguments
- `game::AbstractGNEP` : Game problem
- `X::Vector{Vector}` : Vector of state vectors at each time step
- `U::Vector{Vector}` : Vector of control vectors at each time step
- `times::Vector` : Time points

# Returns
- `Trajectory{T}` : Trajectory structure
"""
function create_trajectory_from_vectors(
    game::AbstractGNEP{T},
    X::Vector{<:AbstractVector{T}},
    U::Vector{<:AbstractVector{T}},
    times::Vector{T}
) where T
    n_steps = length(times)
    @assert length(X) == n_steps "State vector length mismatch"
    @assert length(U) == n_steps "Control vector length mismatch"
    
    n = state_dim(game)
    m = control_dim(game)
    
    states = zeros(T, n, n_steps)
    controls = zeros(T, m, n_steps)
    
    for k in 1:n_steps
        states[:, k] = X[k]
        controls[:, k] = U[k]
    end
    
    return Trajectory{T}(states, controls, times, zero(T))
end

"""
    extract_player_trajectory(
        game::AbstractGNEP,
        trajectory::Trajectory,
        player::Int
    )

Extract a specific player's trajectory from a full trajectory.

# Arguments
- `game::AbstractGNEP` : Game problem
- `trajectory::Trajectory` : Full trajectory
- `player::Int` : Player index

# Returns
- `Trajectory` : Player-specific trajectory (only their states and controls)
"""
function extract_player_trajectory(
    game::AbstractGNEP{T},
    trajectory::Trajectory{T},
    player::Int
) where T
    # This is most relevant for PD-GNEP with separable dynamics
    error("extract_player_trajectory not implemented for $(typeof(game))")
end

function extract_player_trajectory(
    game::PDGNEProblem{T, NP},
    trajectory::Trajectory{T},
    player::Int
) where {T, NP}
    state_idx = state_indices(game, player)
    control_idx = control_indices(game, player)
    
    n_i = state_dim(game, player)
    m_i = control_dim(game, player)
    n_steps = length(trajectory.times)
    
    states = zeros(T, n_i, n_steps)
    controls = zeros(T, m_i, n_steps)
    
    for k in 1:n_steps
        states[:, k] = trajectory.states[state_idx, k]
        controls[:, k] = trajectory.controls[control_idx, k]
    end
    
    return Trajectory{T}(states, controls, trajectory.times, zero(T))
end

"""
    interpolate_trajectory(trajectory::Trajectory{T}, t::Real) where T

Interpolate state and control at time t using linear interpolation.

# Arguments
- `trajectory::Trajectory` : Trajectory to interpolate
- `t::Real` : Time point for interpolation

# Returns
- `(x, u)` : Interpolated state and control at time t
"""
function interpolate_trajectory(trajectory::Trajectory{T}, t::Real) where T
    times = trajectory.times
    
    # Check bounds
    if t <= times[1]
        return (trajectory.states[:, 1], trajectory.controls[:, 1])
    elseif t >= times[end]
        return (trajectory.states[:, end], trajectory.controls[:, end])
    end
    
    # Find surrounding time indices
    idx = searchsortedfirst(times, t)
    if times[idx] == t
        return (trajectory.states[:, idx], trajectory.controls[:, idx])
    end
    
    # Linear interpolation
    t_lo = times[idx-1]
    t_hi = times[idx]
    α = (t - t_lo) / (t_hi - t_lo)
    
    x = (1 - α) * trajectory.states[:, idx-1] + α * trajectory.states[:, idx]
    u = (1 - α) * trajectory.controls[:, idx-1] + α * trajectory.controls[:, idx]
    
    return (x, u)
end

# ============================================================================
# SOLUTION UTILITIES
# ============================================================================

"""
    compute_nash_residual(game::AbstractGNEP, solution::GNEPSolution)

Compute a measure of Nash equilibrium violation (how much each player could improve).

# Arguments
- `game::AbstractGNEP` : Game problem
- `solution::GNEPSolution` : Candidate solution

# Returns
- `Vector{T}` : Residual for each player (0 = perfect equilibrium)

# Notes
This requires solving each player's best response problem, which may be expensive.
Implementation depends on specific solver algorithms.
"""
function compute_nash_residual(game::AbstractGNEP, solution::GNEPSolution)
    error("compute_nash_residual not implemented for $(typeof(game))")
end

"""
    is_nash_equilibrium(
        game::AbstractGNEP, 
        solution::GNEPSolution; 
        tol::Real=1e-6
    )

Check if a solution satisfies Nash equilibrium conditions within tolerance.

# Arguments
- `game::AbstractGNEP` : Game problem
- `solution::GNEPSolution` : Candidate solution
- `tol::Real` : Tolerance for equilibrium check

# Returns
- `Bool` : true if solution is a Nash equilibrium within tolerance
"""
function is_nash_equilibrium(
    game::AbstractGNEP, 
    solution::GNEPSolution; 
    tol::Real=1e-6
)
    residuals = compute_nash_residual(game, solution)
    return all(r -> r < tol, residuals)
end

"""
    compare_solutions(sol1::GNEPSolution, sol2::GNEPSolution)

Compare two solutions and compute differences in costs and trajectories.

# Returns
- `Dict` with comparison metrics
"""
function compare_solutions(sol1::GNEPSolution{T}, sol2::GNEPSolution{T}) where T
    @assert num_players(sol1) == num_players(sol2) "Solutions have different number of players"
    
    NP = num_players(sol1)
    
    # Cost differences
    cost_diffs = sol1.costs .- sol2.costs
    cost_relative_errors = abs.(cost_diffs) ./ (abs.(sol1.costs) .+ 1e-10)
    
    # Trajectory differences (L2 norm)
    traj_diffs = zeros(T, NP)
    for i in 1:NP
        state_diff = norm(sol1.trajectories[i].states - sol2.trajectories[i].states)
        control_diff = norm(sol1.trajectories[i].controls - sol2.trajectories[i].controls)
        traj_diffs[i] = state_diff + control_diff
    end
    
    return Dict(
        :cost_differences => cost_diffs,
        :cost_relative_errors => cost_relative_errors,
        :max_cost_error => maximum(abs, cost_diffs),
        :trajectory_differences => traj_diffs,
        :max_trajectory_difference => maximum(traj_diffs)
    )
end

# ============================================================================
# GAME STRUCTURE QUERIES
# ============================================================================

"""
    is_zero_sum(game::AbstractGNEP)

Check if the game is zero-sum (costs sum to zero for all outcomes).

# Notes
For general GNEPs, this is difficult to check analytically.
Returns false by default; can be overridden for specific game types.
"""
is_zero_sum(game::AbstractGNEP) = false

"""
    is_potential_game(game::AbstractGNEP)

Check if the game is a potential game.
"""
is_potential_game(::AbstractGNEP) = false
is_potential_game(::AbstractPotentialGame) = true

"""
    is_lq_game(game::AbstractGNEP)

Check if the game is linear-quadratic.
"""
is_lq_game(::AbstractGNEP) = false
is_lq_game(::AbstractLQGame) = true

"""
    has_separable_dynamics(game::AbstractGNEP)

Check if the game has separable (per-player) dynamics.
"""
has_separable_dynamics(::AbstractGNEP) = false
has_separable_dynamics(::PDGNEProblem) = true

"""
    game_structure_summary(game::AbstractGNEP)

Generate a summary of the game's mathematical structure.

# Returns
- `Dict{Symbol, Any}` with structural properties
"""
function game_structure_summary(game::AbstractGNEP)
    return Dict(
        :type => typeof(game),
        :num_players => num_players(game),
        :state_dim => state_dim(game),
        :control_dim => control_dim(game),
        :time_horizon => time_horizon(game),
        :has_constraints => has_constraints(game),
        :num_constraints => count_constraints(game),
        :is_potential_game => is_potential_game(game),
        :is_lq_game => is_lq_game(game),
        :separable_dynamics => has_separable_dynamics(game)
    )
end

# ============================================================================
# NUMERICAL UTILITIES
# ============================================================================

"""
    estimate_lipschitz_constant(
        game::AbstractGNEP,
        player::Int,
        x_samples::Vector,
        u_samples::Vector,
        t_samples::Vector
    )

Estimate the Lipschitz constant of player i's cost gradient.

# Arguments
- `game::AbstractGNEP` : Game problem
- `player::Int` : Player index
- `x_samples::Vector` : Sample states
- `u_samples::Vector` : Sample controls
- `t_samples::Vector` : Sample times

# Returns
- Estimated Lipschitz constant (useful for solver parameter tuning)

# Notes
This is a numerical estimate based on finite differences at sample points.
"""
function estimate_lipschitz_constant(
    game::AbstractGNEP{T},
    player::Int,
    x_samples::Vector,
    u_samples::Vector,
    t_samples::Vector;
    ε::Real=1e-5
) where T
    # Compute numerical gradients at sample points and estimate max gradient difference
    # This is a simplified placeholder - full implementation would use ForwardDiff
    @warn "estimate_lipschitz_constant is a placeholder - needs ForwardDiff implementation"
    return one(T)
end

"""
    check_controllability(game::LQGameProblem, player::Int)

Check controllability of player i's subsystem in an LQ game.

# Arguments
- `game::LQGameProblem` : LQ game problem
- `player::Int` : Player index

# Returns
- `Bool` : true if subsystem is controllable

# Notes
For LQ games, checks if rank([Bᵢ, ABᵢ, A²Bᵢ, ...]) = n
"""
function check_controllability(game::LQGameProblem, player::Int)
    A = game.A
    B_i = game.B[player]
    n = size(A, 1)
    m_i = size(B_i, 2)
    
    # Build controllability matrix [B, AB, A²B, ..., Aⁿ⁻¹B]
    C = zeros(eltype(A), n, n * m_i)
    C[:, 1:m_i] = B_i
    
    AB = B_i
    for i in 1:(n-1)
        AB = A * AB
        C[:, i*m_i+1:(i+1)*m_i] = AB
    end
    
    return rank(C) == n
end

"""
    compute_condition_number(game::LQGameProblem)

Compute condition numbers for LQ game matrices (A, Bᵢ, Qᵢ, Rᵢ).

# Returns
- `Dict` with condition numbers for each matrix
"""
function compute_condition_number(game::LQGameProblem)
    cond_nums = Dict{Symbol, Any}()
    cond_nums[:A] = cond(game.A)
    cond_nums[:B] = [cond(B_i) for B_i in game.B]
    cond_nums[:Q] = [cond(Q_i) for Q_i in game.Q]
    cond_nums[:R] = [cond(R_i) for R_i in game.R]
    cond_nums[:Qf] = [cond(Qf_i) for Qf_i in game.Qf]
    
    return cond_nums
end

# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

"""
    validate_trajectory(game::AbstractGNEP, trajectory::Trajectory; tol::Real=1e-6)

Validate that a trajectory satisfies the dynamics within tolerance.

# Arguments
- `game::AbstractGNEP` : Game problem
- `trajectory::Trajectory` : Trajectory to validate
- `tol::Real` : Tolerance for dynamics violation

# Returns
- `(valid::Bool, max_violation::Real)` : Validation result and maximum violation
"""
function validate_trajectory(
    game::AbstractGNEP{T},
    trajectory::Trajectory{T};
    tol::Real=1e-6
) where T
    n_steps = length(trajectory.times)
    max_violation = zero(T)
    
    for k in 1:(n_steps-1)
        x_k = trajectory.states[:, k]
        x_kp1 = trajectory.states[:, k+1]
        u_k = trajectory.controls[:, k]
        t_k = trajectory.times[k]
        dt = trajectory.times[k+1] - t_k
        
        # Compute dynamics
        ẋ_k = evaluate_dynamics(game, x_k, u_k, t_k)
        
        # Forward Euler estimate
        x_kp1_est = x_k + dt * ẋ_k
        
        # Compute violation
        violation = norm(x_kp1 - x_kp1_est)
        max_violation = max(max_violation, violation)
    end
    
    return (max_violation < tol, max_violation)
end

"""
    validate_solution(game::AbstractGNEP, solution::GNEPSolution; tol::Real=1e-6)

Comprehensive validation of a solution.

# Returns
- `Dict` with validation results for each check
"""
function validate_solution(
    game::AbstractGNEP,
    solution::GNEPSolution;
    tol::Real=1e-6
)
    results = Dict{Symbol, Any}()
    
    # Check number of players matches
    results[:player_count_match] = (num_players(solution) == num_players(game))
    
    # Validate each trajectory
    for i in 1:num_players(solution)
        traj = solution.trajectories[i]
        (valid, max_viol) = validate_trajectory(game, traj; tol=tol)
        results[Symbol("player_$(i)_dynamics_valid")] = valid
        results[Symbol("player_$(i)_max_violation")] = max_viol
    end
    
    # Check if costs match computed costs
    for i in 1:num_players(solution)
        computed_cost = evaluate_total_cost(game, i, solution.trajectories[i])
        reported_cost = solution.costs[i]
        cost_error = abs(computed_cost - reported_cost)
        results[Symbol("player_$(i)_cost_error")] = cost_error
        results[Symbol("player_$(i)_cost_valid")] = (cost_error < tol)
    end
    
    return results
end
