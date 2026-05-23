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
    forward_solution::Union{Nothing, GNEPSolution{T}}
    converged::Bool
    solve_time::Float64
    solver_info::Dict{Symbol, Any}
end

# Accessor mirrors GameSolution interface where applicable
"""Return the recovered weight vector for unknown player i."""
get_weights(sol::InverseGameSolution, i::Int) = sol.weights[i]

"""Return the weight trajectory matrix (k × T) for unknown player i."""
get_weight_history(sol::InverseGameSolution, i::Int) = sol.weight_history[i]

# ============================================================================
# InverseLQGameSolution — result of the Inga et al. (2019) algebraic solver
# ============================================================================

"""
    InverseLQGameSolution{T}

Solution of an inverse LQ differential game (Inga et al. 2019).

The canonical parameter set for player i is the null space of Mᵢ:
    Θ = ∩ᵢ ker(Mᵢ)   with Rᵢᵢ ≻ 0 boundaries.

# Fields
- `problem`        : the `InverseLQGame{T}` that was solved
- `K_estimated`    : feedback matrices used (= K_star if given exactly;
                     least-squares estimate otherwise)
- `F`              : closed-loop matrix A − Σᵢ BᵢKᵢ  (n × n)
- `M`              : per-player constraint matrices Mᵢ ∈ ℝ^{npᵢ × L}
- `kernels`        : per-player null-space basis matrices (columns = basis vectors);
                     every θᵢ in the basis satisfies Mᵢθᵢ = 0
- `kernel_dims`    : dimension of each player's null space
- `theta_min_norm` : minimum-norm element of ker(Mᵢ) per player
                     (a representative cost-parameter vector)
- `residuals`      : ‖Mᵢθᵢ^(min-norm)‖₂ per player (≈ 0 for exact K*)
- `converged`      : `true` if all residuals ≤ `solver.tol`
- `solver_info`    : diagnostics: `:singular_values` (per player),
                     `:F_eigenvalues` (closed-loop spectrum),
                     `:K_estimated_residuals` (LS residuals if trajectory mode)

# Recovering cost matrices from `theta_min_norm`
```julia
sol = solve(prob, InverseLQGames())
n, ps = prob.state_dim, prob.control_dims
for i in 1:prob.n_players
    θ = sol.theta_min_norm[i]
    Q_i  = reshape(θ[1:n^2], n, n)
    off  = n^2
    R_ij = [reshape(θ[off + sum(ps[1:j-1].^2) + 1 : off + sum(ps[1:j].^2)], ps[j], ps[j])
            for j in 1:prob.n_players]
end
```
"""
struct InverseLQGameSolution{T}
    problem        ::InverseLQGame{T}
    K_estimated    ::Vector{Matrix{T}}
    F              ::Matrix{T}
    M              ::Vector{Matrix{T}}
    kernels        ::Vector{Matrix{T}}
    kernel_dims    ::Vector{Int}
    theta_min_norm ::Vector{Vector{T}}
    residuals      ::Vector{T}
    converged      ::Bool
    solver_info    ::Dict{Symbol, Any}
end

"""
    get_kernel(sol::InverseLQGameSolution, i::Int) -> Matrix{T}

Return the null-space basis for player i's cost parameters.
Columns of the result are orthonormal basis vectors; any valid parameter
vector is a linear combination of these columns.
"""
get_kernel(sol::InverseLQGameSolution, i::Int) = sol.kernels[i]

"""
    get_M(sol::InverseLQGameSolution, i::Int) -> Matrix{T}

Return the constraint matrix Mᵢ for player i (npᵢ × L).
"""
get_M(sol::InverseLQGameSolution, i::Int) = sol.M[i]

"""
    extract_cost_matrices(sol::InverseLQGameSolution, i::Int, θ=nothing)
    -> (Q_i, [R_{i1}, …, R_{iN}])

Reshape a parameter vector `θ` for player `i` into cost matrices.
If `θ` is not provided, uses `sol.theta_min_norm[i]`.

Returns `(Q_i, R_vec)` where `Q_i` is n×n and `R_vec[j]` is pⱼ×pⱼ.
"""
function extract_cost_matrices(
    sol::InverseLQGameSolution{T},
    i  ::Int,
    θ  ::Union{Nothing, AbstractVector{T}} = nothing
) where {T}
    prob = sol.problem
    n    = prob.state_dim
    ps   = prob.control_dims
    np   = prob.n_players
    θ_i  = θ === nothing ? sol.theta_min_norm[i] : θ

    Q_i = reshape(θ_i[1:n^2], n, n)
    R_vec = Vector{Matrix{T}}(undef, np)
    off = n^2
    for j in 1:np
        len = ps[j]^2
        R_vec[j] = reshape(θ_i[off+1:off+len], ps[j], ps[j])
        off += len
    end
    return Q_i, R_vec
end

function Base.show(io::IO, sol::InverseLQGameSolution{T}) where {T}
    status = sol.converged ? "converged" : "not converged"
    print(io, "InverseLQGameSolution{$T} ($status, max residual=",
          round(maximum(sol.residuals), sigdigits=3), ")")
end

function Base.show(io::IO, ::MIME"text/plain", sol::InverseLQGameSolution{T}) where {T}
    println(io, "InverseLQGameSolution{$T}")
    println(io, "  Status       : ", sol.converged ? "converged" : "not converged")
    println(io, "  Players      : ", sol.problem.n_players)
    for i in 1:sol.problem.n_players
        println(io, "  Player $i:")
        println(io, "    ker(M_$i) dim : ", sol.kernel_dims[i])
        println(io, "    residual     : ", round(sol.residuals[i], sigdigits=4))
        if !isempty(sol.solver_info[:singular_values][i])
            sv = sol.solver_info[:singular_values][i]
            println(io, "    σ_min(M_$i)  : ", round(minimum(sv), sigdigits=4))
        end
    end
end
