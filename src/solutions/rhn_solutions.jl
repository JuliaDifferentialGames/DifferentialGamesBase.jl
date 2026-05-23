# ============================================================================
# solutions/rhn_solutions.jl
#
# Defines RecedingHorizonNashSolution{T} — the closed-loop trajectory
# and accumulated cost from solving a RecedingHorizonNashProblem.
#
# Include order: must come after:
#   - problems/RHN.jl        (needs RecedingHorizonNashProblem{T})
#   - solutions/gnep_solutions.jl (needs AbstractSolution{T})
# ============================================================================

"""
    RecedingHorizonNashSolution{T} <: AbstractSolution{T}

Closed-loop trajectory resulting from solving a `RecedingHorizonNashProblem`.

Each simulation step t applies the first optimal control from the
receding-horizon sub-game solved at state X[:,t], propagates the
dynamics, and accumulates costs.

# Fields
- `problem`     : The `RecedingHorizonNashProblem{T}` that was solved.
- `X`           : Closed-loop state trajectory, shape (n × n_sim_steps+1).
  `X[:,1] = x0`; `X[:,t+1] = f(X[:,t], u_joint_t)`.
- `U`           : Per-player closed-loop controls.
  `U[i]` has shape (mᵢ × n_sim_steps); `U[i][:,t]` is player i's
  control applied at step t.
- `costs`       : Total accumulated cost per player (length n_players).
  Computed as Σ_t stage_costᵢ(X[:,t], U[i][:,t]) + terminal_costᵢ(X[:,end]).
- `converged`   : True iff ALL n_sim_steps inner sub-game solves converged.
- `solve_time`  : Total wall-clock seconds spent inside inner solves.
- `solver_info` : Diagnostics Dict. Standard entries:
  - `:n_steps_simulated`  — Int, equals n_sim_steps
  - `:inner_solver_type`  — Type, typeof(inner_solver)
  - `:n_inner_converged`  — Int, count of converged sub-solves

# Accessors
- `get_state_trajectory(sol)`        → X (n × N_sim+1)
- `get_control_trajectory(sol, i)`   → U[i] (mᵢ × N_sim)
- `get_total_cost(sol, i)`           → scalar accumulated cost for player i
"""
struct RecedingHorizonNashSolution{T} <: AbstractSolution{T}
    problem    ::RecedingHorizonNashProblem{T}
    X          ::Matrix{T}
    U          ::Vector{Matrix{T}}
    costs      ::Vector{T}
    converged  ::Bool
    solve_time ::Float64
    solver_info::Dict{Symbol, Any}

    function RecedingHorizonNashSolution{T}(
        problem    ::RecedingHorizonNashProblem{T},
        X          ::Matrix{T},
        U          ::Vector{Matrix{T}},
        costs      ::Vector{T},
        converged  ::Bool,
        solve_time ::Float64,
        solver_info::Dict{Symbol, Any}
    ) where {T}
        np    = n_players(problem)
        N_sim = problem.n_sim_steps
        n     = state_dim(problem)

        @assert size(X, 1) == n        "X must have $n rows (state dim)"
        @assert size(X, 2) == N_sim+1  "X must have $(N_sim+1) columns, got $(size(X,2))"
        @assert length(U)  == np       "U must have one entry per player ($np), got $(length(U))"
        @assert length(costs) == np    "costs must have one entry per player ($np)"
        for i in 1:np
            @assert size(U[i], 2) == N_sim "U[$i] must have $N_sim columns, got $(size(U[i],2))"
        end

        new{T}(problem, X, U, costs, converged, solve_time, solver_info)
    end
end

# ─── Outer keyword constructor ────────────────────────────────────────────────
function RecedingHorizonNashSolution(
    problem    ::RecedingHorizonNashProblem{T},
    X          ::Matrix{T},
    U          ::Vector{Matrix{T}},
    costs      ::Vector{T};
    converged  ::Bool              = true,
    solve_time ::Float64           = 0.0,
    solver_info::Dict{Symbol, Any} = Dict{Symbol, Any}()
) where {T}
    RecedingHorizonNashSolution{T}(problem, X, U, costs, converged, solve_time, solver_info)
end

# ============================================================================
# Accessors
# ============================================================================

"""
    get_state_trajectory(sol::RecedingHorizonNashSolution) -> Matrix{T}

Return the full closed-loop state trajectory (n × n_sim_steps+1).
"""
get_state_trajectory(sol::RecedingHorizonNashSolution) = sol.X

"""
    get_control_trajectory(sol::RecedingHorizonNashSolution, i::Int) -> Matrix{T}

Return player i's closed-loop control sequence (mᵢ × n_sim_steps).
"""
function get_control_trajectory(sol::RecedingHorizonNashSolution, i::Int)
    np = n_players(sol.problem)
    1 ≤ i ≤ np || error("Player $i out of range [1, $np]")
    return sol.U[i]
end

"""
    get_total_cost(sol::RecedingHorizonNashSolution, i::Int) -> Real

Return the total accumulated cost for player i over the full simulation.
"""
function get_total_cost(sol::RecedingHorizonNashSolution, i::Int)
    np = n_players(sol.problem)
    1 ≤ i ≤ np || error("Player $i out of range [1, $np]")
    return sol.costs[i]
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, sol::RecedingHorizonNashSolution{T}) where {T}
    status = sol.converged ? "CONVERGED" : "NOT CONVERGED"
    print(io, "RecedingHorizonNashSolution{$T} [$(n_players(sol.problem)) players, ",
          "N_sim=$(sol.problem.n_sim_steps), $status]")
end

function Base.show(io::IO, ::MIME"text/plain", sol::RecedingHorizonNashSolution{T}) where {T}
    np = n_players(sol.problem)
    println(io, "RecedingHorizonNashSolution{$T}")
    println(io, "  Players          : ", np)
    println(io, "  Simulation steps : ", sol.problem.n_sim_steps)
    println(io, "  Window horizon N : ", n_steps(sol.problem.horizon_game))
    println(io, "  Converged        : ", sol.converged)
    println(io, "  Solve time       : ", round(sol.solve_time, digits=4), " s")
    println(io, "  Player costs:")
    for i in 1:np
        println(io, "    Player $i: ", round(sol.costs[i], digits=6))
    end
    if haskey(sol.solver_info, :n_inner_converged)
        nc = sol.solver_info[:n_inner_converged]
        println(io, "  Inner convergence: $nc / $(sol.problem.n_sim_steps)")
    end
end
