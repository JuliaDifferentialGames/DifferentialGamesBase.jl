# ============================================================================
# player_based_game.jl
#
# Phase 2 additions:
#   - Player{T}           public alias for PlayerSpec{T}
#   - DifferentialGame    smart constructor dispatching to the right problem type
#   - remake              shallow-copy GameProblem with field overrides
#
# Include order: must come after problems/GNEP.jl in DifferentialGamesBase.jl
# ============================================================================

# ============================================================================
# Player{T} — public alias for PlayerSpec{T}
#
# PlayerSpec{T} is the internal implementation type and remains unchanged.
# Player{T} is the documented public API. They are the same type at runtime;
# no conversion, wrapping, or overhead is incurred.
# ============================================================================

"""
    Player{T}

Public alias for `PlayerSpec{T}`. Specifies a single player in a differential game.

# Fields
- `id`          : Player index (1-based, must be unique within the game)
- `n`           : State dimension
- `m`           : Control dimension
- `x0`          : Initial state (length n)
- `dynamics`    : Dynamics function `fᵢ(xᵢ, uᵢ, p, t)` — for PD-GNEPs with
                  `SeparableDynamics`. For shared-state games built via
                  `DifferentialGame(dyn, ...)`, this field is unused.
- `objective`   : `PlayerObjective` — use `minimize(...)` from cost_terms.jl
- `constraints` : Player-private constraints (optional)

# Construction
```julia
# PD-GNEP player (separable dynamics)
p1 = Player(1, n=4, m=2, x0=zeros(4),
            dynamics=(x,u,p,t) -> f1(x,u,p,t),
            objective=minimize(track_goal(xg, Q) + regularize_input(R),
                               terminal=terminal_goal(xg, Qf),
                               player_id=1))

# Keyword constructor (recommended)
p1 = Player{Float64}(1, 4, 2, zeros(4), f1, obj1)
```

See also: `DifferentialGame`, `minimize`
"""
const Player{T} = PlayerSpec{T}

# ============================================================================
# DifferentialGame — smart constructor
#
# Dispatch logic:
#
#   DifferentialGame(players::Vector{Player{T}}, ...)
#     → All players have separable dynamics → PDGNEProblem
#
#   DifferentialGame(dyn::LinearDynamics{T}, players::Vector{Player{T}}, ...)
#     → Shared linear dynamics → LQGameProblem or LTVLQGameProblem
#       (requires LQStageCost and LQTerminalCost in all objectives)
#
#   DifferentialGame(dyn::CoupledNonlinearDynamics{T}, players::Vector{Player{T}}, ...)
#     → Shared nonlinear dynamics → GameProblem directly
#
# In all cases the user may override the inferred problem type by passing
# the `force_type` keyword.
# ============================================================================

"""
    DifferentialGame(players, tf, dt; kwargs...) -> GameProblem{T}
    DifferentialGame(dyn, players, tf, dt; kwargs...) -> GameProblem{T}

Smart constructor for differential games. Dispatches to the appropriate
`GameProblem` constructor based on argument types.

# Signatures

## PD-GNEP (separable per-player dynamics)
```julia
DifferentialGame(
    players::Vector{Player{T}},
    tf::T, dt::T;
    shared_constraints = [],
    time_horizon::Union{Nothing,TimeHorizon{T}} = nothing
)
```
Requires each `Player` to have a dynamics function. Calls `PDGNEProblem`.

## Shared dynamics
```julia
DifferentialGame(
    dyn::DynamicsSpec{T},
    players::Vector{Player{T}},
    tf::T, dt::T;
    shared_constraints = [],
    time_horizon::Union{Nothing,TimeHorizon{T}} = nothing
)
```
Constructs a `GameProblem{T}` with a shared `dyn` and per-player objectives.
For `LinearDynamics` with `LQStageCost` objectives, the result passes
`is_lq_game` and can be solved directly by FNELQ.

# Arguments
- `players`             : `Vector{Player{T}}` (or `Vector{PlayerSpec{T}}`),
                          ordered by player ID 1:N. IDs must be 1:N exactly.
- `tf`, `dt`            : Final time and time step. Ignored if `time_horizon`
                          is provided explicitly.
- `shared_constraints`  : Constraints coupling multiple players (default empty).
- `time_horizon`        : If provided, overrides `tf`/`dt`.

# Returns
`GameProblem{T}` — use `validate_game_problem` to check consistency.

# Examples
```julia
# Two-player hallway avoidance (PD-GNEP)
p1 = Player{Float64}(1, 2, 1, x1_0, f1, obj1)
p2 = Player{Float64}(2, 2, 1, x2_0, f2, obj2)
game = DifferentialGame([p1, p2], 5.0, 0.1; shared_constraints=[wall_con])

# Two-player LQ game (shared linear dynamics)
game = DifferentialGame(dyn, [p1, p2], 5.0, 0.1)
```
"""
function DifferentialGame(
    players::Vector{<:PlayerSpec{T}},
    tf::T,
    dt::T;
    shared_constraints::AbstractVector = [],
    time_horizon::Union{Nothing, TimeHorizon{T}} = nothing
) where {T}
    _check_players(players)
    th = _resolve_time_horizon(tf, dt, time_horizon)
    return PDGNEProblem(Vector{PlayerSpec{T}}(players), shared_constraints,
                        th.tf, th.dt)
end

# Shared dynamics dispatch
function DifferentialGame(
    dyn::DynamicsSpec{T},
    players::Vector{<:PlayerSpec{T}},
    tf::T,
    dt::T;
    shared_constraints::AbstractVector = [],
    time_horizon::Union{Nothing, TimeHorizon{T}} = nothing
) where {T}
    _check_players(players)
    th = _resolve_time_horizon(tf, dt, time_horizon)
    return _build_shared_dynamics_game(dyn, players, shared_constraints, th)
end

# Internal: shared dynamics → GameProblem
function _build_shared_dynamics_game(
    dyn::DynamicsSpec{T},
    players::Vector{<:PlayerSpec{T}},
    shared_constraints::AbstractVector,
    th::DiscreteTime{T}
) where {T}
    n_players = length(players)
    objectives = [p.objective for p in players]

    # Validate that all objectives have player IDs consistent with position
    for (i, obj) in enumerate(objectives)
        @assert(obj.player_id == players[i].id,
            "Player $(players[i].id) objective has player_id=$(obj.player_id)")
    end

    # Reconstruct initial_state from players if dynamics is separable,
    # otherwise require players to have consistent x0 sizes
    initial_state = if dyn isa SeparableDynamics
        vcat([p.x0 for p in players]...)
    else
        # For shared-state games, all players share the same initial state.
        # We use player 1's x0 as the joint x0 — they must be equal or caller
        # must set x0 directly via remake(game; initial_state=x0).
        players[1].x0
    end

    @assert(length(initial_state) == total_state_dim(dyn),
        "initial_state length $(length(initial_state)) ≠ dynamics state dim $(total_state_dim(dyn))")

    private_constraints = Vector{Any}(vcat([p.constraints for p in players]...))
    shared_constraints  = Vector{Any}(shared_constraints)

    state_dims   = dyn isa SeparableDynamics ? dyn.state_dims : [total_state_dim(dyn)]
    # For CoupledNonlinearDynamics the struct only stores the total control dim
    # (scalar); per-player breakdown comes from the players themselves.
    # For SeparableDynamics and LinearDynamics, control_dims is a Vector field.
    control_dims = if dyn isa CoupledNonlinearDynamics
        [p.m for p in players]
    else
        dyn.control_dims
    end
    state_offsets   = dyn isa SeparableDynamics ?
        [0; cumsum(state_dims)[1:end-1]] : [0]
    control_offsets = [0; cumsum(control_dims)[1:end-1]]

    cost_coupling = sparse(trues(n_players, n_players))
    coupling_graph = CouplingGraph(
        cost_coupling,
        Vector{Int}[c.players for c in shared_constraints if hasproperty(c, :players)],
        nothing
    )
    metadata = GameMetadata(state_dims, control_dims, state_offsets, control_offsets,
                            coupling_graph, false, nothing)

    return GameProblem{T}(
        n_players, objectives, dyn, initial_state,
        private_constraints, shared_constraints, th, metadata
    )
end

# ============================================================================
# remake — shallow-copy GameProblem with field overrides
#
# Design rationale:
#   GameProblem is immutable. For MPC, the hot path is:
#     new_game = remake(game; initial_state=x_new)
#   which allocates only the new struct + the new initial_state vector.
#   All other fields (dynamics, objectives, constraints, metadata) are shared
#   by reference — O(1) cost regardless of horizon length.
#
#   This matches DifferentialEquations.jl's remake() pattern exactly.
# ============================================================================

"""
    remake(game::GameProblem{T}; kwargs...) -> GameProblem{T}

Create a copy of `game` with specified fields replaced.

All non-specified fields are shared by reference — no deep copy.
Allocation cost is O(1): one new struct + any explicitly overridden fields.

# Supported Keyword Overrides
- `initial_state::Vector{T}`        — primary use case for MPC
- `time_horizon::TimeHorizon{T}`    — change horizon length or dt
- `objectives`                      — replace all player objectives
- `private_constraints`             — replace private constraint list
- `shared_constraints`              — replace shared constraint list
- `metadata::GameMetadata`          — rarely needed; for structural changes

# MPC Pattern
```julia
game0 = DifferentialGame(dyn, players, tf, dt)
for t in 1:T_sim
    sol   = solve(game0, solver; warmstart=prev_sol)
    x_new = simulate_step(x0, sol)                    # apply u*(t₀), advance
    game0 = remake(game0; initial_state=x_new)        # O(1) — shared refs
end
```

# Notes
Does not call `validate_game_problem` — the caller is responsible for
ensuring consistency when structural fields (metadata, dynamics) are changed.
For `initial_state`-only updates, consistency is guaranteed.
"""
function remake(
    game::GameProblem{T};
    initial_state    = game.initial_state,
    time_horizon     = game.time_horizon,
    objectives       = game.objectives,
    private_constraints = game.private_constraints,
    shared_constraints  = game.shared_constraints,
    metadata         = game.metadata
) where {T}
    GameProblem{T}(
        game.n_players,
        objectives,
        game.dynamics,
        initial_state,
        private_constraints,
        shared_constraints,
        time_horizon,
        metadata
    )
end

# ============================================================================
# Internal helpers
# ============================================================================

function _check_players(players::Vector{<:PlayerSpec{T}}) where {T}
    n = length(players)
    @assert n > 0 "Must provide at least one player"
    ids = [p.id for p in players]
    @assert(allunique(ids), "Player IDs must be unique, got $ids")
    @assert(sort(ids) == collect(1:n),
        "Player IDs must be exactly 1:$n, got $(sort(ids))")
end

function _resolve_time_horizon(
    tf::T,
    dt::T,
    override::Union{Nothing, TimeHorizon{T}}
) where {T}
    override !== nothing && return override
    return DiscreteTime(tf, dt)
end