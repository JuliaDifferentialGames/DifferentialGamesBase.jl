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

"""
    WarmstartData{T}

Flexible container for solver initialization data.
Solvers extract what they need and gracefully ignore the rest.

# Fields
- `trajectories::Union{Nothing, Vector{Trajectory{T}}}`: State/control trajectories
- `dual_variables::Union{Nothing, Dict{Symbol, Any}}`: Constraint multipliers
- `solver_specific::Dict{Symbol, Any}`: Solver-specific data

# Notes
Different solvers use different warmstart information:
- Trajectory-based: Use `.trajectories`
- Dual methods: Use `.dual_variables`
- Custom: Store in `.solver_specific`

Solvers should handle missing data gracefully by using default initialization.
"""
struct WarmstartData{T}
    trajectories::Union{Nothing, Vector{Trajectory{T}}}
    dual_variables::Union{Nothing, Dict{Symbol, Any}}
    solver_specific::Dict{Symbol, Any}
    
    function WarmstartData{T}(
        trajectories::Union{Nothing, Vector{Trajectory{T}}} = nothing,
        dual_variables::Union{Nothing, Dict{Symbol, Any}} = nothing,
        solver_specific::Dict{Symbol, Any} = Dict{Symbol, Any}()
    ) where {T}
        new{T}(trajectories, dual_variables, solver_specific)
    end
end

# Convenience constructor from GameSolution
function WarmstartData(sol::GameSolution{T}) where {T}
    WarmstartData{T}(
        sol.trajectories,
        get(sol.solver_info, :dual_variables, nothing),
        sol.solver_info
    )
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

# Allow GameSolution to be passed directly as warmstart
_solve(game::GameProblem{T}, solver::GameSolver, warmstart::GameSolution{T}, verbose::Bool) where {T} = 
    _solve(game, solver, WarmstartData(warmstart), verbose)


"""
    solve(game::GameProblem, solver::GameSolver; kwargs...)

Solve a differential game to find Nash equilibrium.

# Arguments
- `game::GameProblem{T}`: The game problem to solve
- `solver::GameSolver`: Configured solver instance

# Keyword Arguments
- `warmstart::Union{Nothing, GameSolution, WarmstartData}=nothing`: Initial guess
- `verbose::Bool=false`: Print iteration information
- `check_compatibility::Bool=true`: Verify solver capabilities match game requirements

# Returns
- `GameSolution{T}`: Solution containing trajectories and convergence info

# Notes
Compatibility checking issues warnings by default, never blocks execution.
Solvers handle warmstart data gracefully - extract what's useful, ignore the rest.

# Example
```julia
solver = iLQGames(max_iter=100, abs_tol=1e-6, rel_tol=1e-4)
sol = solve(game, solver; warmstart=previous_sol, verbose=true)

# Check convergence
sol.converged || @warn "Solver did not converge"

# Access solver-specific diagnostics
residuals = get(sol.solver_info, :residual_history, nothing)
```
"""
function solve(
    game::GameProblem{T}, 
    solver::GameSolver; 
    warmstart::Union{Nothing, GameSolution, WarmstartData} = nothing,
    verbose::Bool = false,
    check_compatibility::Bool = true
) where {T}
    
    # Compatibility check - warning only
    if check_compatibility
        required = required_capabilities(game)
        provided = solver_capabilities(typeof(solver))
        missing = setdiff(required, provided)
        
        if !isempty(missing)
            @warn """
            Solver $(typeof(solver)) may not handle game properties: $(missing)
            Proceeding anyway - disable this warning with check_compatibility=false
            """
        end
    end
    
    # Delegate to solver-specific implementation
    return _solve(game, solver, warmstart, verbose)
end
