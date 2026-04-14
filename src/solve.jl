# ============================================================================
# solve.jl
#
# Solver dispatch infrastructure. Updated from original:
#   - solve() returns AbstractSolution{T} (not GameSolution{T})
#   - WarmstartData.from_solution accepts GNEPSolution
#   - _solve default signature updated to return AbstractSolution
#   - GameSolution removed (replaced by GNEPSolution in gnep_solutions.jl)
# ============================================================================

abstract type GameSolver end

"""
    solver_capabilities(::Type{<:GameSolver}) -> Vector{Symbol}

Return symbols describing what problem types this solver handles.

# Common Capabilities
- `:LQGame`           — linear-quadratic games
- `:NonlinearGame`    — general nonlinear dynamics and costs
- `:GNEP`             — generalized Nash (shared constraints)
- `:ConstrainedGame`  — private constraints
- `:SeparableDynamics` — decoupled player dynamics
- `:FeedbackPolicies` — computes feedback gains (not just trajectories)
- `:DiscreteTime`     — requires discrete-time formulation
"""
solver_capabilities(::Type{<:GameSolver}) = Symbol[]

"""
    required_capabilities(game::GameProblem) -> Vector{Symbol}

Extract required solver capabilities from game structure.
"""
function required_capabilities(game::GameProblem)
    caps = Symbol[]
    is_lq_game(game)            && push!(caps, :LQGame)
    !is_lq_game(game)           && push!(caps, :NonlinearGame)
    is_potential_game(game)     && push!(caps, :PotentialGame)
    !isempty(game.shared_constraints) && push!(caps, :GNEP)
    !isempty(game.private_constraints) && push!(caps, :ConstrainedGame)
    has_shared_constraints(game) && push!(caps, :SharedConstraints)
    has_separable_dynamics(game) && push!(caps, :SeparableDynamics)
    game.time_horizon isa DiscreteTime && push!(caps, :DiscreteTime)
    return unique(caps)
end

# ============================================================================
# WarmstartData
# ============================================================================

"""
    WarmstartData{T}

Flexible warmstart container. Solvers extract what they need and ignore the rest.

# Fields
- `trajectories`    : Per-player trajectories or nothing
- `strategy`        : Typed AbstractStrategy or nothing (replaces raw dict entry)
- `dual_variables`  : Constraint multipliers or nothing
- `solver_specific` : Solver-specific Dict
"""
struct WarmstartData{T}
    trajectories::Union{Nothing, Vector{Trajectory{T}}}
    strategy::Union{Nothing, AbstractStrategy{T}}
    dual_variables::Union{Nothing, Dict{Symbol, Any}}
    solver_specific::Dict{Symbol, Any}

    function WarmstartData{T}(
        trajectories   = nothing,
        strategy       = nothing,
        dual_variables = nothing,
        solver_specific = Dict{Symbol, Any}()
    ) where {T}
        new{T}(trajectories, strategy, dual_variables, solver_specific)
    end
end

"""
    WarmstartData(sol::GNEPSolution{T}) -> WarmstartData{T}

Construct warmstart from a previous solution. Extracts typed strategy directly.
"""
function WarmstartData(sol::GNEPSolution{T}) where {T}
    WarmstartData{T}(
        sol.trajectories,
        sol.strategy,
        get(sol.solver_info, :dual_variables, nothing),
        sol.solver_info
    )
end

# ============================================================================
# _solve — solver implementations override this
# ============================================================================

"""
    _solve(game, solver, warmstart, verbose) -> AbstractSolution{T}

Internal dispatch target. Solver packages implement this method.
"""
function _solve(
    game::GameProblem{T},
    solver::GameSolver,
    warmstart::Union{Nothing, GNEPSolution, WarmstartData},
    verbose::Bool
) where {T}
    error("_solve not implemented for $(typeof(solver))")
end

# Allow GNEPSolution to be passed directly as warmstart
function _solve(
    game::GameProblem{T},
    solver::GameSolver,
    warmstart::GNEPSolution{T},
    verbose::Bool
) where {T}
    _solve(game, solver, WarmstartData(warmstart), verbose)
end

# ============================================================================
# solve — public API
# ============================================================================

"""
    solve(game::GameProblem, solver::GameSolver; kwargs...) -> AbstractSolution{T}

Solve a differential game for Nash equilibrium.

# Keyword Arguments
- `warmstart = nothing`        : Previous solution or WarmstartData
- `verbose::Bool = false`      : Print iteration info
- `check_compatibility = true` : Warn if solver may not support game type

# Returns
`AbstractSolution{T}` — typically `GNEPSolution{T}` for forward games.
"""
function solve(
    game::GameProblem{T},
    solver::GameSolver;
    warmstart::Union{Nothing, GNEPSolution, WarmstartData} = nothing,
    verbose::Bool = false,
    check_compatibility::Bool = true
) where {T}
    if check_compatibility
        required = required_capabilities(game)
        provided = solver_capabilities(typeof(solver))
        missing_caps = setdiff(required, provided)
        if !isempty(missing_caps)
            @warn "Solver $(typeof(solver)) may not handle: $missing_caps" *
                  " — disable with check_compatibility=false"
        end
    end
    return _solve(game, solver, warmstart, verbose)
end