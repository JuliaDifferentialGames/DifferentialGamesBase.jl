abstract type GameSolver end

"""
    solver_capabilities(::Type{<:GameSolver})

Return vector of symbols describing what problem types this solver can handle.

# Common Capabilities
- `:LQGame` - Linear-quadratic games
- `:NonlinearGame` - General nonlinear dynamics and costs
- `:GNEP` - Generalized Nash equilibrium problems (shared constraints)
- `:ConstrainedGame` - Games with private constraints
- `:SeparableDynamics` - Decoupled player dynamics
- `:SharedConstraints` - Any shared constraints between players
- `:PotentialGame` - Games with potential function structure
- `:FeedbackPolicies` - Can compute feedback policies (not just trajectories)

# Example
```julia
solver_capabilities(::Type{LQRiccati}) = [:LQGame, :SeparableDynamics]
solver_capabilities(::Type{ALGAMES}) = [:GNEP, :ConstrainedGame, :LQGame, :SharedConstraints]
```
"""
solver_capabilities(::Type{<:GameSolver}) = Symbol[]


"""
    required_capabilities(game::GameProblem)

Extract required solver capabilities from game structure and metadata.
Used for compatibility checking.
"""
function required_capabilities(game::GameProblem)
    caps = Symbol[]
    
    # Check game structure
    is_lq_game(game) && push!(caps, :LQGame)
    !is_lq_game(game) && push!(caps, :NonlinearGame)
    is_potential_game(game) && push!(caps, :PotentialGame)
    
    # Check constraint structure
    !isempty(game.shared_constraints) && push!(caps, :GNEP)
    !isempty(game.private_constraints) && push!(caps, :ConstrainedGame)
    has_shared_constraints(game) && push!(caps, :SharedConstraints)
    
    # Check dynamics properties
    game.metadata.separable_dynamics && push!(caps, :SeparableDynamics)
    
    return unique(caps)
end

# Solver implementations must override this
function _solve(
    game::GameProblem{T},
    solver::GameSolver,
    warmstart::Union{Nothing, GameSolution, WarmstartData},
    verbose::Bool
) where {T}
    error("solve not implemented for $(typeof(solver))")
end