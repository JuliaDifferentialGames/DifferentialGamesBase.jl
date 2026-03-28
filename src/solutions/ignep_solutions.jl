# ============================================================================
# Inverse Solution Type
# ============================================================================

"""
    InverseGameSolution{T}

Solution container for inverse game solvers. Analogous to `GameSolution`
for forward solvers.

# Fields
- `problem::InverseGameProblem{T}` : The problem that was solved
- `weights::Dict{Int, Vector{T}}` : Final recovered weight vector per unknown player
- `weight_history::Dict{Int, Matrix{T}}` : Full weight trajectory; columns are timesteps
- `ensemble_history::Dict{Int, Array{T,3}}` : Ensemble over time; size (k, N_e, T)
- `forward_solution::Union{Nothing, GameSolution{T}}` : Nash solution under recovered objectives
- `converged::Bool` : Whether STLS weights stabilized
- `solve_time::Float64`
- `solver_info::Dict{Symbol, Any}` : Solver diagnostics (residuals, STLS triggers, etc.)
"""
struct InverseGameSolution{T}
    problem::InverseGameProblem{T}
    weights::Dict{Int, Vector{T}}
    weight_history::Dict{Int, Matrix{T}}
    ensemble_history::Dict{Int, Array{T, 3}}
    forward_solution::Union{Nothing, GameSolution{T}}
    converged::Bool
    solve_time::Float64
    solver_info::Dict{Symbol, Any}
end

# Accessor mirrors GameSolution interface where applicable
"""Return the recovered weight vector for unknown player i."""
get_weights(sol::InverseGameSolution, i::Int) = sol.weights[i]

"""Return the weight trajectory matrix (k × T) for unknown player i."""
get_weight_history(sol::InverseGameSolution, i::Int) = sol.weight_history[i]
