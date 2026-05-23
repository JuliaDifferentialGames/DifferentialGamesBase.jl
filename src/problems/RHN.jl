# ============================================================================
# problems/RHN.jl
#
# Defines:
#   - RecedingHorizonNashProblem{T}   — specification for a RH Nash game
#   - with_initial_state(game, x0)   — O(1) subproblem rebuild helper
#
# Concept:
#   At each simulation step t, the outer loop:
#     1. Calls with_initial_state(horizon_game, X[:,t]) to patch the
#        sub-problem initial condition in O(1) (no deep copy).
#     2. Delegates to any GameSolver via solve(subgame, inner_solver).
#     3. Applies only the first control û*(1; x_t).
#     4. Advances via _rollout_step.
#     5. Shifts the previous solution to warm-start the next solve.
#
# Include order: must come after problems/GNEP.jl (needs GameProblem{T}).
# ============================================================================

# ============================================================================
# RecedingHorizonNashProblem
# ============================================================================

"""
    RecedingHorizonNashProblem{T} <: AbstractDeterministicGame{T}

Specification for a receding-horizon feedback Nash game.

The solver repeatedly solves a finite-horizon Nash sub-game, applies
only the first control action, advances the state, and shifts the
previous solution to warm-start the next sub-problem solve.

# Fields
- `horizon_game`  : Template `GameProblem{T}` for the receding window.
  Its dynamics, objectives, constraints, and time horizon (lookahead)
  are reused at every step. Its `initial_state` is ignored — it is
  patched at each step via `with_initial_state`.
- `x0`            : Initial state for the full closed-loop simulation.
- `n_sim_steps`   : Total number of real-time steps to simulate.

# Construction
```julia
prob = RecedingHorizonNashProblem(horizon_game, x0, n_sim_steps)
```

# Notes
- `horizon_game.time_horizon.tf` sets the prediction horizon length.
- Any `GameSolver` that handles `GameProblem{T}` can serve as the
  inner solver — FNELQ, iLQGames, ALGAMES, etc.
- The closed-loop cost accumulated is over `n_sim_steps` real steps,
  using the stage/terminal costs in `horizon_game.objectives`.

# References
- Mattingley, Wang, Boyd (2011) — receding horizon control with online
  convex optimization; shift-and-warm-start strategy.
- Laine et al. (2023) — GFNE: generalized feedback Nash equilibria;
  sequential LQ game approach for constrained receding-horizon games.
"""
struct RecedingHorizonNashProblem{T} <: AbstractDeterministicGame{T}
    horizon_game ::GameProblem{T}
    x0           ::Vector{T}
    n_sim_steps  ::Int

    function RecedingHorizonNashProblem{T}(
        horizon_game::GameProblem{T},
        x0::Vector{T},
        n_sim_steps::Int
    ) where {T}
        @assert n_sim_steps > 0 "n_sim_steps must be positive, got $n_sim_steps"
        n = state_dim(horizon_game)
        @assert length(x0) == n "x0 length $(length(x0)) ≠ state dim $n"
        new{T}(horizon_game, x0, n_sim_steps)
    end
end

# ─── Outer constructor: converts x0 to Vector{T} automatically ────────────────
function RecedingHorizonNashProblem(
    horizon_game::GameProblem{T},
    x0::AbstractVector{<:Real},
    n_sim_steps::Int
) where {T}
    RecedingHorizonNashProblem{T}(horizon_game, convert(Vector{T}, x0), n_sim_steps)
end

# ============================================================================
# Accessors
# ============================================================================

n_players(p::RecedingHorizonNashProblem) = n_players(p.horizon_game)
state_dim(p::RecedingHorizonNashProblem) = state_dim(p.horizon_game)

# ============================================================================
# with_initial_state — O(1) sub-problem construction
# ============================================================================

"""
    with_initial_state(game::GameProblem{T}, x0::AbstractVector) -> GameProblem{T}

Return a `GameProblem{T}` identical to `game` but with `initial_state`
replaced by `x0`. All other fields (dynamics, objectives, constraints,
time_horizon, metadata) are shared by reference — no deep copy is performed.

This is the core primitive of the receding-horizon loop: it patches the
sub-problem initial condition in O(1) time at each simulation step.

# Example
```julia
subgame = with_initial_state(horizon_game, X[:, t])
sol     = solve(subgame, inner_solver; warmstart=ws)
```
"""
function with_initial_state(game::GameProblem{T}, x0::AbstractVector) where {T}
    GameProblem{T}(
        game.n_players,
        game.objectives,
        game.dynamics,
        convert(Vector{T}, x0),
        game.private_constraints,
        game.shared_constraints,
        game.time_horizon,
        game.metadata
    )
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, p::RecedingHorizonNashProblem{T}) where {T}
    print(io, "RecedingHorizonNashProblem{$T} [$(n_players(p)) players, ",
          "N_sim=$(p.n_sim_steps), N_win=$(n_steps(p.horizon_game))]")
end

function Base.show(io::IO, ::MIME"text/plain", p::RecedingHorizonNashProblem{T}) where {T}
    println(io, "RecedingHorizonNashProblem{$T}")
    println(io, "  Players          : ", n_players(p))
    println(io, "  State dim        : ", state_dim(p))
    println(io, "  Simulation steps : ", p.n_sim_steps)
    println(io, "  Window horizon N : ", n_steps(p.horizon_game))
    println(io, "  Window tf        : ", p.horizon_game.time_horizon.tf)
    println(io, "  Initial state x0 : ", p.x0)
end
