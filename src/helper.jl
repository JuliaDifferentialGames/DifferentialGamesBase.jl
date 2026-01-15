# ============================================================================
# Property Query Functions
# ============================================================================

"""
    has_separable_dynamics(game::GameProblem)

Check if game has separable (decoupled) dynamics structure.

Returns true for PD-GNEPs where ẋᵢ = fᵢ(xᵢ, uᵢ).
"""
has_separable_dynamics(game::GameProblem) = game.dynamics isa SeparableDynamics

"""
    is_lq_game(game::GameProblem)

Check if all objectives are linear-quadratic and dynamics are linear.

Returns true if this is a classical LQ game solvable via Riccati equations.
"""
function is_lq_game(game::GameProblem)
    # Check dynamics are linear
    if !(game.dynamics isa LinearDynamics)
        return false
    end
    
    # Check all objectives have LQ stage and terminal costs
    for obj in game.objectives
        if !(obj.stage_cost isa Union{LQStageCost, DiagonalLQStageCost})
            return false
        end
        if !(obj.terminal_cost isa Union{LQTerminalCost, DiagonalLQTerminalCost})
            return false
        end
    end
    
    return true
end

"""
    is_potential_game(game::GameProblem)

Check if game has potential function structure.

Returns true if stored in metadata (must be computed/verified at construction).
"""
is_potential_game(game::GameProblem) = game.metadata.is_potential

"""
    has_shared_constraints(game::GameProblem)

Check if game has constraints coupling multiple players.
"""
has_shared_constraints(game::GameProblem) = !isempty(game.shared_constraints)

"""
    is_unconstrained(game::GameProblem)

Check if game has no constraints of any kind.
"""
is_unconstrained(game::GameProblem) = 
    isempty(game.private_constraints) && isempty(game.shared_constraints)

"""
    is_pd_gnep(game::GameProblem)

Check if game is a Partially-Decoupled GNEP (separable dynamics).
"""
is_pd_gnep(game::GameProblem) = has_separable_dynamics(game)

"""
    is_lq_pd_gnep(game::GameProblem)

Check if game is an LQ game with separable dynamics.
"""
is_lq_pd_gnep(game::GameProblem) = is_lq_game(game) && is_pd_gnep(game)

# ============================================================================
# Helper Functions
# ============================================================================

"""
    num_players(game::GameProblem)

Get number of players in game.
"""
num_players(game::GameProblem) = game.n_players

"""
    control_dim(game::GameProblem)

Get total control dimension (sum across all players).
"""
control_dim(game::GameProblem) = sum(game.metadata.control_dims)

"""
    state_dim(game::GameProblem)

Get total state dimension (sum across all players for separable, or single dimension for shared).
"""
function state_dim(game::GameProblem)
    if has_separable_dynamics(game)
        return sum(game.metadata.state_dims)
    else
        # Shared state space
        return game.metadata.state_dims[1]
    end
end

"""
    state_dim(game::GameProblem, player::Int)

Get state dimension for specific player.
"""
state_dim(game::GameProblem, player::Int) = game.metadata.state_dims[player]

"""
    control_dim(game::GameProblem, player::Int)

Get control dimension for specific player.
"""
control_dim(game::GameProblem, player::Int) = game.metadata.control_dims[player]

"""
    get_objective(game::GameProblem, player::Int)

Get objective function for specific player.
"""
function get_objective(game::GameProblem, player::Int)
    idx = findfirst(obj -> obj.player_id == player, game.objectives)
    isnothing(idx) && error("No objective found for player $player")
    return game.objectives[idx]
end