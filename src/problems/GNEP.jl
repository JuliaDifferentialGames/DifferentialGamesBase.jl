# ============================================================================
# problems/GNEP.jl
#
# Defines:
#   - LQStageCost{T,...}        unified LTI/LTV quadratic stage cost
#   - LQTerminalCost{T}         quadratic terminal cost
#   - GameProblem{T}            universal game problem container
#   - LQGameProblem             LTI constructor
#   - LTVLQGameProblem          LTV constructor
#   - PDGNEProblem              PD-GNEP constructor
#   - validate_game_problem     consistency check (Phase 1e)
#   - num_players               accessor on GameProblem
#   - has_separable_dynamics    structural query
#   - is_lq_pd_gnep             combined query
#
# Include order constraint:
#   Must come after: objectives.jl, constraints.jl, metadata.jl,
#                    dynamics.jl, time_horizon.jl, player_spec.jl
#   Must come before: abstract/strategy.jl, dynamics/discretization.jl,
#                     solutions/gnep_solutions.jl
# ============================================================================

# ============================================================================
# LQStageCost
# ============================================================================

"""
    LQStageCost{T, QM, RM, MM, QVM, RVM} <: AbstractStageCost

Unified LTI/LTV quadratic stage cost.

# Mathematical Form
ℓᵢ(x, uᵢ, k) = ½ xᵀQ(k)x + ½ uᵢᵀR(k)uᵢ + xᵀM(k)uᵢ + q(k)ᵀx + r(k)ᵀuᵢ + c

Always use `get_Q(cost, k)`, `get_R(cost, k)`, etc. in solver hot loops.
"""
struct LQStageCost{T, QM, RM, MM, QVM, RVM} <: AbstractStageCost
    Q::QM
    R::RM
    M::MM
    q::QVM
    r::RVM
    c::T

    function LQStageCost{T, QM, RM, MM, QVM, RVM}(
        Q::QM, R::RM, M::MM, q::QVM, r::RVM, c::T
    ) where {T, QM, RM, MM, QVM, RVM}
        new{T, QM, RM, MM, QVM, RVM}(Q, R, M, q, r, c)
    end
end

# ─── LTI positional constructor ───────────────────────────────────────────────

function LQStageCost(
    Q::Matrix{T}, R::Matrix{T}, M::Matrix{T},
    q::Vector{T}, r::Vector{T}, c::T
) where {T}
    n_x, n_u = size(M)
    @assert size(Q) == (n_x, n_x) && issymmetric(Q)   "Q must be symmetric (n_x × n_x)"
    @assert size(R) == (n_u, n_u) && issymmetric(R)   "R must be symmetric (n_u × n_u)"
    @assert isposdef(R)                                "R must be positive definite"
    ev = eigvals(Symmetric(Q))
    @assert all(ev .>= -sqrt(eps(T)) * maximum(abs.(ev))) "Q must be positive semidefinite"
    @assert length(q) == n_x && length(r) == n_u      "q/r dimension mismatch"
    LQStageCost{T, Matrix{T}, Matrix{T}, Matrix{T}, Vector{T}, Vector{T}}(Q, R, M, q, r, c)
end

# ─── LTI keyword convenience (preserves existing call sites) ──────────────────

function LQStageCost(
    Q::Matrix{T}, R::Matrix{T};
    M::Union{Matrix{T}, Nothing} = nothing,
    q::Union{Vector{T}, Nothing} = nothing,
    r::Union{Vector{T}, Nothing} = nothing,
    c::T = zero(T)
) where {T}
    n_x = size(Q, 1); n_u = size(R, 1)
    LQStageCost(
        Q, R,
        M === nothing ? zeros(T, n_x, n_u) : M,
        q === nothing ? zeros(T, n_x)       : q,
        r === nothing ? zeros(T, n_u)       : r,
        c
    )
end

# ─── LTV constructors ─────────────────────────────────────────────────────────

function LQStageCost(
    Q_seq::Vector{Matrix{T}}, R_seq::Vector{Matrix{T}},
    M_seq::Vector{Matrix{T}}, q_seq::Vector{Vector{T}}, r_seq::Vector{Vector{T}}
) where {T}
    N = length(Q_seq)
    @assert N > 0 "Cost sequences must be non-empty"
    @assert all(length.([R_seq, M_seq, q_seq, r_seq]) .== N) "All sequences must have length $N"
    n_x = size(Q_seq[1], 1); n_u = size(R_seq[1], 1)
    for k in 1:N
        @assert size(Q_seq[k]) == (n_x, n_x) && issymmetric(Q_seq[k])
        @assert size(R_seq[k]) == (n_u, n_u) && issymmetric(R_seq[k]) && isposdef(R_seq[k])
    end
    LQStageCost{T, Vector{Matrix{T}}, Vector{Matrix{T}}, Vector{Matrix{T}},
                Vector{Vector{T}}, Vector{Vector{T}}}(
        Q_seq, R_seq, M_seq, q_seq, r_seq, zero(T)
    )
end

function LQStageCost(Q_seq::Vector{Matrix{T}}, R_seq::Vector{Matrix{T}}) where {T}
    N = length(Q_seq); n_x = size(Q_seq[1], 1); n_u = size(R_seq[1], 1)
    LQStageCost(Q_seq, R_seq,
        [zeros(T, n_x, n_u) for _ in 1:N],
        [zeros(T, n_x)      for _ in 1:N],
        [zeros(T, n_u)      for _ in 1:N])
end

# ─── Structural queries ───────────────────────────────────────────────────────

is_ltv(::LQStageCost{T, Matrix{T}}) where {T}         = false
is_ltv(::LQStageCost{T, Vector{Matrix{T}}}) where {T} = true
is_separable(::LQStageCost) = false

# ─── Accessors ────────────────────────────────────────────────────────────────

get_Q(c::LQStageCost{T, Matrix{T}}, k::Int) where {T}         = c.Q
get_Q(c::LQStageCost{T, Vector{Matrix{T}}}, k::Int) where {T} = c.Q[k]

get_R(c::LQStageCost{T, Matrix{T}}, k::Int) where {T}         = c.R
get_R(c::LQStageCost{T, Vector{Matrix{T}}}, k::Int) where {T} = c.R[k]

get_M(c::LQStageCost{T, Matrix{T}}, k::Int) where {T}         = c.M
get_M(c::LQStageCost{T, Vector{Matrix{T}}}, k::Int) where {T} = c.M[k]

get_q(c::LQStageCost{T, <:Any, <:Any, <:Any, Vector{T}, <:Any}, k::Int) where {T}         = c.q
get_q(c::LQStageCost{T, <:Any, <:Any, <:Any, Vector{Vector{T}}, <:Any}, k::Int) where {T} = c.q[k]

get_r(c::LQStageCost{T, <:Any, <:Any, <:Any, <:Any, Vector{T}}, k::Int) where {T}         = c.r
get_r(c::LQStageCost{T, <:Any, <:Any, <:Any, <:Any, Vector{Vector{T}}}, k::Int) where {T} = c.r[k]

# ─── Evaluation (defined here because it references the struct accessors) ─────

function evaluate_stage_cost(cost::LQStageCost, x, u, p, t)
    Q = get_Q(cost, t); R = get_R(cost, t); M = get_M(cost, t)
    q = get_q(cost, t); r = get_r(cost, t)
    return 0.5 * (x' * Q * x + u' * R * u + 2 * x' * M * u) + q' * x + r' * u + cost.c
end

function stage_cost_gradient(cost::LQStageCost, x, u, p, t)
    Q = get_Q(cost, t); R = get_R(cost, t); M = get_M(cost, t)
    q = get_q(cost, t); r = get_r(cost, t)
    return (Q * x + M * u + q, R * u + M' * x + r)
end

function stage_cost_hessian(cost::LQStageCost, x, u, p, t)
    return (get_Q(cost, t), get_R(cost, t), get_M(cost, t))
end

# ============================================================================
# LQTerminalCost
# ============================================================================

"""
    LQTerminalCost{T} <: AbstractTerminalCost

Quadratic terminal cost: Vᵢ(x(tf)) = ½ x(tf)ᵀQf x(tf) + qfᵀx(tf) + cf
"""
struct LQTerminalCost{T} <: AbstractTerminalCost
    Qf::Matrix{T}
    qf::Vector{T}
    cf::T

    function LQTerminalCost(Qf::Matrix{T}, qf::Vector{T}, cf::T) where {T}
        n = size(Qf, 1)
        @assert size(Qf) == (n, n) && issymmetric(Qf) "Qf must be symmetric square"
        @assert length(qf) == n                        "qf length must match Qf"
        ev = eigvals(Symmetric(Qf))
        @assert all(ev .>= -sqrt(eps(T)) * maximum(abs.(ev))) "Qf must be PSD"
        new{T}(Qf, qf, cf)
    end
end

LQTerminalCost(Qf::Matrix{T}; qf=nothing, cf=zero(T)) where {T} =
    LQTerminalCost(Qf, qf === nothing ? zeros(T, size(Qf,1)) : qf, cf)

LQTerminalCost(Qf::Matrix{T}, qf::Vector{T}) where {T} =
    LQTerminalCost(Qf, qf, zero(T))

"""
    DiagonalLQStageCost(q_diag, r_diag) -> LQStageCost

Convenience constructor: build an `LQStageCost` from diagonal weight vectors.
"""
DiagonalLQStageCost(q_diag::AbstractVector{T}, r_diag::AbstractVector{T}) where {T} =
    LQStageCost(diagm(q_diag), diagm(r_diag))

"""
    DiagonalLQTerminalCost(qf_diag) -> LQTerminalCost

Convenience constructor: build an `LQTerminalCost` from a diagonal weight vector.
"""
DiagonalLQTerminalCost(qf_diag::AbstractVector{T}) where {T} =
    LQTerminalCost(diagm(qf_diag))

function evaluate_terminal_cost(cost::LQTerminalCost, x, p)
    return 0.5 * x' * cost.Qf * x + cost.qf' * x + cost.cf
end

function terminal_cost_gradient(cost::LQTerminalCost, x, p)
    return cost.Qf * x + cost.qf
end

function terminal_cost_hessian(cost::LQTerminalCost, x, p)
    return cost.Qf
end

# ============================================================================
# GameProblem
# ============================================================================

"""
    GameProblem{T} <: AbstractDeterministicGame{T}

Universal game problem representation for all GNEP variants.
Subtypes AbstractDeterministicGame from the Phase 0a hierarchy.
"""
struct GameProblem{T} <: AbstractDeterministicGame{T}
    n_players::Int
    objectives::Vector{<:PlayerObjective}
    dynamics::DynamicsSpec{T}
    initial_state::Vector{T}
    private_constraints::AbstractVector
    shared_constraints::AbstractVector
    time_horizon::TimeHorizon{T}
    metadata::GameMetadata

    function GameProblem{T}(
        n_players::Int,
        objectives::Vector{<:PlayerObjective},
        dynamics::DynamicsSpec{T},
        initial_state::Vector{T},
        private_constraints::AbstractVector,
        shared_constraints::AbstractVector,
        time_horizon::TimeHorizon{T},
        metadata::GameMetadata
    ) where {T}
        @assert n_players > 0                              "Must have at least one player"
        @assert length(objectives) == n_players            "Must have objective for each player"
        @assert allunique(obj.player_id for obj in objectives) "Duplicate player IDs"
        @assert all(1 ≤ obj.player_id ≤ n_players for obj in objectives) "Invalid player IDs"
        all_ids = Set(1:n_players)
        for c in private_constraints
            @assert get_player(c) in all_ids "Private constraint references invalid player"
        end
        for c in shared_constraints
            @assert all(p in all_ids for p in get_players(c)) "Shared constraint references invalid player"
        end
        new{T}(n_players, objectives, dynamics, initial_state,
               private_constraints, shared_constraints, time_horizon, metadata)
    end
end

# ============================================================================
# AbstractGameProblem interface — implement n_players for GameProblem
# ============================================================================

"""
    n_players(game::GameProblem) -> Int

Return the number of players in the game.
"""
n_players(g::GameProblem) = g.n_players

# ============================================================================
# Structural queries on GameProblem
# ============================================================================

"""
    num_players(g::GameProblem) -> Int

Number of players. Alias for `n_players` — use whichever reads more clearly.
"""
num_players(g::GameProblem) = g.n_players

is_unconstrained(g::GameProblem) =
    isempty(g.private_constraints) && isempty(g.shared_constraints)

is_lq_game(g::GameProblem) =
    g.dynamics isa LinearDynamics &&
    all(obj.stage_cost isa LQStageCost for obj in g.objectives)

is_pd_gnep(g::GameProblem)             = g.dynamics isa SeparableDynamics
has_shared_constraints(g::GameProblem) = !isempty(g.shared_constraints)
is_potential_game(g::GameProblem)      = g.metadata.is_potential
has_separable_dynamics(g::GameProblem) = g.dynamics isa SeparableDynamics

is_lq_pd_gnep(g::GameProblem) = is_pd_gnep(g) && is_lq_game(g)

"""
    state_dim(game::GameProblem) -> Int
    state_dim(game::GameProblem, i::Int) -> Int

Total joint state dimension, or player `i`'s private state dimension for PD-GNEPs.
"""
state_dim(g::GameProblem)          = total_state_dim(g.dynamics)
state_dim(g::GameProblem, i::Int)  = g.metadata.state_dims[i]

"""
    control_dim(game::GameProblem) -> Int
    control_dim(game::GameProblem, i::Int) -> Int

Total joint control dimension, or player `i`'s control dimension.
"""
control_dim(g::GameProblem)        = total_control_dim(g.dynamics)
control_dim(g::GameProblem, i::Int) = g.metadata.control_dims[i]

"""
    n_steps(game::GameProblem) -> Int

Number of discrete time steps `N = round(tf / dt)`.
"""
function n_steps(g::GameProblem{T}) where {T}
    th = g.time_horizon
    @assert th isa DiscreteTime "n_steps requires a DiscreteTime horizon"
    return Int(round(th.tf / th.dt))
end

function get_objective(g::GameProblem, player_id::Int)
    idx = findfirst(obj -> obj.player_id == player_id, g.objectives)
    isnothing(idx) && error("No objective for player $player_id")
    return g.objectives[idx]
end

# ============================================================================
# validate_game_problem (Phase 1e) — lives here because it references GameProblem
# ============================================================================

"""
    validate_game_problem(game::GameProblem{T}) -> Nothing

Assert internal consistency of a `GameProblem`. Raises `AssertionError` on first failure.

# Checks
1. Objective count matches n_players
2. Player IDs are unique and cover 1:n_players
3. GameMetadata.control_dims matches dynamics.control_dims
4. GameMetadata.control_offsets are cumulative sums of control_dims
5. GameMetadata.state_dims sum matches total_state_dim(dynamics)
6. initial_state length matches state dim
7. LTV sequence length is consistent with time horizon (for DiscreteTime)
"""
function validate_game_problem(game::GameProblem{T}) where {T}
    np = game.n_players

    @assert(length(game.objectives) == np,
        "n_players=$np but $(length(game.objectives)) objectives")

    ids = [obj.player_id for obj in game.objectives]
    @assert allunique(ids) "Duplicate player IDs: $ids"
    @assert(sort(ids) == collect(1:np),
        "Player IDs must be exactly 1:$np, got $(sort(ids))")

    dyn_cd  = game.dynamics.control_dims
    meta_cd = game.metadata.control_dims
    @assert(dyn_cd == meta_cd,
        "control_dims mismatch: dynamics=$dyn_cd, metadata=$meta_cd")

    exp_offs = [0; cumsum(dyn_cd)[1:end-1]]
    @assert(game.metadata.control_offsets == exp_offs,
        "control_offsets inconsistent with control_dims")

    @assert(sum(game.metadata.state_dims) == total_state_dim(game.dynamics),
        "state_dims sum ≠ dynamics total state dim")

    @assert(length(game.initial_state) == total_state_dim(game.dynamics),
        "initial_state length $(length(game.initial_state)) ≠ state dim $(total_state_dim(game.dynamics))")

    if game.time_horizon isa DiscreteTime && is_ltv(game.dynamics)
        Nh = n_steps(game)
        Nd = game.dynamics.n_steps
        if Nd !== nothing
            @assert(Nd == Nh,
                "LTV dynamics has $Nd steps but time horizon implies $Nh")
        end
    end

    return nothing
end

# ============================================================================
# LQGameProblem — LTI constructor
# ============================================================================

"""
    LQGameProblem(A, B, Q, R, Qf, x0, tf; dt=0.01) -> GameProblem

Construct a finite-horizon, discrete-time, linear-quadratic (LQ) game with shared state.

All players observe and act on the same joint state vector `x ∈ ℝⁿ`. Dynamics are linear
time-invariant: `x_{k+1} = A x_k + Σᵢ Bᵢ uᵢᵏ`. Each player `i` minimizes a quadratic cost
`Σₖ (xₖᵀ Qᵢ xₖ + uᵢᵏᵀ Rᵢ uᵢᵏ) + xₙᵀ Qfᵢ xₙ`.

# Arguments
- `A`: `n×n` state transition matrix
- `B`: `Vector{Matrix}` — `B[i]` is the `n×mᵢ` input matrix for player `i`
- `Q`: `Vector{Matrix}` — `Q[i]` is player `i`'s `n×n` stage state cost
- `R`: `Vector{Matrix}` — `R[i]` is player `i`'s `mᵢ×mᵢ` control cost
- `Qf`: `Vector{Matrix}` — terminal cost matrices, one per player
- `x0`: initial state vector
- `tf`: final time; `dt` (keyword) is the time step
"""
function LQGameProblem(
    A::Matrix{T}, B::Vector{Matrix{T}},
    Q::Vector{Matrix{T}}, R::Vector{Matrix{T}}, Qf::Vector{Matrix{T}},
    x0::Vector{T}, tf::T;
    dt::T = T(0.01),
    M::Union{Vector{Matrix{T}}, Nothing} = nothing,
    q::Union{Vector{Vector{T}}, Nothing} = nothing,
    r::Union{Vector{Vector{T}}, Nothing} = nothing
) where {T}
    n = size(A, 1)
    n_players = length(B)
    @assert length(Q) == n_players && length(R) == n_players && length(Qf) == n_players
    @assert length(x0) == n

    dynamics     = LinearDynamics(A, B)
    control_dims = dynamics.control_dims

    objectives = map(1:n_players) do i
        mi = control_dims[i]
        stage_cost    = LQStageCost(
            Q[i], R[i],
            isnothing(M) ? zeros(T, n, mi) : M[i],
            isnothing(q) ? zeros(T, n)     : q[i],
            isnothing(r) ? zeros(T, mi)    : r[i],
            zero(T)
        )
        terminal_cost = LQTerminalCost(Qf[i])
        PlayerObjective(i, stage_cost, terminal_cost)
    end

    time_horizon    = DiscreteTime(tf, dt)
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    coupling_graph  = CouplingGraph(sparse(trues(n_players, n_players)), Vector{Int}[], nothing)
    metadata = GameMetadata([n], control_dims, [0], control_offsets, coupling_graph, false, nothing)

    return GameProblem{T}(
        n_players, objectives, dynamics, x0,
        AbstractPrivateConstraint[], AbstractSharedConstraint[], time_horizon, metadata
    )
end

# ============================================================================
# LTVLQGameProblem — LTV constructor
# ============================================================================

"""
    LTVLQGameProblem(A_seq, B_seq, Q_seq, R_seq, Qf, x0, tf; dt=0.01) -> GameProblem

Construct a finite-horizon, discrete-time, linear time-varying (LTV) LQ game.

Like `LQGameProblem` but with time-varying matrices. Indexing convention:
- `A_seq[k]` — state matrix at step `k`
- `B_seq[i][k]` — player `i`'s input matrix at step `k`
- `Q_seq[i][k]` — player `i`'s state cost at step `k`
- `R_seq[i][k]` — player `i`'s control cost at step `k`
- `Qf[i]` — player `i`'s terminal cost matrix
"""
function LTVLQGameProblem(
    A_seq::Vector{Matrix{T}}, B_seq::Vector{Vector{Matrix{T}}},
    Q_seq::Vector{Vector{Matrix{T}}}, R_seq::Vector{Vector{Matrix{T}}},
    Qf::Vector{Matrix{T}}, x0::Vector{T}, tf::T;
    dt::T = T(0.01),
    M_seq::Union{Vector{Vector{Matrix{T}}}, Nothing} = nothing,
    q_seq::Union{Vector{Vector{Vector{T}}}, Nothing} = nothing,
    r_seq::Union{Vector{Vector{Vector{T}}}, Nothing} = nothing
) where {T}
    N         = length(A_seq)
    N_from_dt = Int(round(tf / dt))
    @assert N == N_from_dt "A_seq length $N ≠ tf/dt=$N_from_dt"

    n_players = length(B_seq)
    @assert length(Q_seq) == n_players && length(R_seq) == n_players && length(Qf) == n_players

    n  = size(A_seq[1], 1)
    @assert length(x0) == n

    dynamics     = LinearDynamics(A_seq, B_seq)
    control_dims = dynamics.control_dims

    objectives = map(1:n_players) do i
        mi     = control_dims[i]
        stage_cost = LQStageCost(
            Q_seq[i], R_seq[i],
            isnothing(M_seq) ? [zeros(T, n, mi) for _ in 1:N] : M_seq[i],
            isnothing(q_seq) ? [zeros(T, n)     for _ in 1:N] : q_seq[i],
            isnothing(r_seq) ? [zeros(T, mi)    for _ in 1:N] : r_seq[i]
        )
        PlayerObjective(i, stage_cost, LQTerminalCost(Qf[i]))
    end

    time_horizon    = DiscreteTime(tf, dt)
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    coupling_graph  = CouplingGraph(sparse(trues(n_players, n_players)), Vector{Int}[], nothing)
    metadata = GameMetadata([n], control_dims, [0], control_offsets, coupling_graph, false, nothing)

    return GameProblem{T}(
        n_players, objectives, dynamics, x0,
        AbstractPrivateConstraint[], AbstractSharedConstraint[], time_horizon, metadata
    )
end

# ============================================================================
# PDGNEProblem — PD-GNEP constructor
# ============================================================================

"""
    PDGNEProblem(players, shared_constraints, tf, dt) -> GameProblem
    PDGNEProblem(players, tf, dt) -> GameProblem

Construct a Partially-Decoupled Generalized Nash Equilibrium Problem (PD-GNEP).

# Formal Definition
The PD-GNEP is the tuple

    𝔾 = (T, {Υⁱ}ᵢ, {Jⁱ}ᵢ, {fⁱ}ᵢ, {C(𝒳, Υ)}ᵢ)

where each player i ∈ {1, …, N} solves

    min_{𝒳ⁱ, Υⁱ}  Jⁱ(𝒳, Υ)
    subject to     Dⁱ(𝒳ⁱ, Υⁱ) = 0    [separable dynamics, private]
                   C(𝒳, Υ)    ≤ 0    [shared constraints, equality or inequality]

The "partial decoupling" refers to the dynamics: fⁱ depends only on player i's
own state 𝒳ⁱ and control Υⁱ. The objectives Jⁱ and shared constraints C(𝒳, Υ)
may depend on the full joint state 𝒳 and the full joint control Υ = (Υ¹, …, Υᴺ).

In the GNEP framing each player i optimises over Υⁱ while treating Υ⁻ⁱ as
fixed — the definition writes C(𝒳, Υ⁻ⁱ) to emphasise this, but the constraint
function itself is C(𝒳, Υ) and may depend on all controls.

# Argument-to-definition mapping
| Argument                    | Definition component                     |
|-----------------------------|------------------------------------------|
| `tf`, `dt`                  | T — time horizon                         |
| `players[i].dynamics`       | fⁱ — separable player dynamics           |
| `players[i].objective`      | Jⁱ — per-player cost functional          |
| `players[i].constraints`    | Υⁱ — strategy space (bounds, etc.)       |
| `shared_constraints`        | C(𝒳, Υ) — shared equality/inequality    |

The dynamics Dⁱ(𝒳ⁱ, Υⁱ) = 0 are enforced via forward simulation (shooting
method) rather than as explicit equality constraints, which is equivalent for
trajectory optimisation but differs from direct transcription formulations.

# Solution concept
A strategy profile Υ* is an *open-loop Nash equilibrium* if for every player i
and every feasible alternative strategy Υⁱ:

    Jⁱ(𝒳*, Υ*) ≤ Jⁱ(𝒳(Υⁱ, Υ⁻ⁱ*), {Υⁱ, Υ⁻ⁱ*})

# Arguments
- `players`: `Vector{PlayerSpec{T}}` — one entry per player; bundles fⁱ, Jⁱ,
  initial state x₀ⁱ, and private strategy-space constraints Υⁱ
- `shared_constraints`: constraints C(𝒳, Υ) ≤ 0 (or = 0) evaluated on the
  full joint state and control; may be equality or inequality
- `tf`: final time
- `dt`: time step
"""
function PDGNEProblem(
    players::Vector{PlayerSpec{T}},
    shared_constraints::AbstractVector,
    tf::T, dt::T
) where {T}
    n_players = length(players)
    @assert n_players > 0 && allunique(p.id for p in players)

    objectives   = [p.objective for p in players]
    state_dims   = [p.n for p in players]
    control_dims = [p.m for p in players]
    dynamics     = SeparableDynamics([p.dynamics for p in players], state_dims, control_dims)
    initial_state = vcat([p.x0 for p in players]...)

    private_constraints = Vector{Any}(vcat([p.constraints for p in players]...))
    shared_constraints  = Vector{Any}(shared_constraints)

    time_horizon    = DiscreteTime(tf, dt)
    state_offsets   = [0; cumsum(state_dims)[1:end-1]]
    control_offsets = [0; cumsum(control_dims)[1:end-1]]

    cost_coupling = sparse(trues(n_players, n_players))
    for (i, obj) in enumerate(objectives)
        if is_separable(obj.stage_cost)
            cost_coupling[i, :] .= false
            cost_coupling[i, i]  = true
        end
    end
    coupling_graph = CouplingGraph(
        cost_coupling,
        Vector{Int}[get_players(c) for c in shared_constraints],
        nothing
    )
    metadata = GameMetadata(state_dims, control_dims, state_offsets, control_offsets,
                            coupling_graph, false, nothing)

    return GameProblem{T}(
        n_players, objectives, dynamics, initial_state,
        private_constraints, shared_constraints, time_horizon, metadata
    )
end

PDGNEProblem(players::Vector{PlayerSpec{T}}, tf::T, dt::T) where {T} =
    PDGNEProblem(players, AbstractSharedConstraint[], tf, dt)

# ============================================================================
# Utility functions
# ============================================================================

function total_cost(obj::PlayerObjective, X::Vector, U::Vector, p)
    N = length(U)
    @assert length(X) == N + 1
    stage_sum = sum(evaluate_stage_cost(obj.stage_cost, X[t], U[t], p, t) for t in 1:N)
    return obj.scaling * (stage_sum + evaluate_terminal_cost(obj.terminal_cost, X[end], p))
end

function diagnose_scaling(obj::PlayerObjective, X::Vector, U::Vector, p)
    N           = length(U)
    stage_costs = [evaluate_stage_cost(obj.stage_cost, X[t], U[t], p, t) for t in 1:N]
    term_cost   = evaluate_terminal_cost(obj.terminal_cost, X[end], p)
    stage_sum   = sum(stage_costs)
    return (
        stage_costs = stage_costs, terminal_cost = term_cost,
        stage_mean = stage_sum / N, stage_std = std(stage_costs),
        stage_min = minimum(stage_costs), stage_max = maximum(stage_costs),
        stage_to_terminal_ratio = stage_sum / (term_cost + eps()),
        total_cost = obj.scaling * (stage_sum + term_cost)
    )
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, g::GameProblem{T}) where {T}
    tags = String[]
    is_lq_game(g)        && push!(tags, "LQ")
    is_ltv(g.dynamics)   && push!(tags, "LTV")
    is_pd_gnep(g)        && push!(tags, "PD-GNEP")
    is_potential_game(g) && push!(tags, "Potential")
    is_unconstrained(g)  && push!(tags, "Unconstrained")
    tag_str = isempty(tags) ? "" : " [$(join(tags, ", "))]"
    print(io, "GameProblem{$T} with $(g.n_players) players$tag_str")
end

function Base.show(io::IO, ::MIME"text/plain", g::GameProblem{T}) where {T}
    println(io, "GameProblem{$T}")
    println(io, "  Players         : ", g.n_players)
    println(io, "  State dim       : ", state_dim(g))
    println(io, "  Control dim     : ", control_dim(g))
    println(io, "  Dynamics        : ", g.dynamics)
    println(io, "  Time horizon    : ", g.time_horizon)
    println(io, "  Private constr. : ", length(g.private_constraints))
    println(io, "  Shared constr.  : ", length(g.shared_constraints))
    println(io, "  Properties:")
    println(io, "    LQ game       : ", is_lq_game(g))
    println(io, "    LTV           : ", is_ltv(g.dynamics))
    println(io, "    PD-GNEP       : ", is_pd_gnep(g))
    println(io, "    Potential     : ", is_potential_game(g))
    println(io, "    Unconstrained : ", is_unconstrained(g))
end

# ============================================================================
# StatePotentialGameProblem — state-based potential game (Marden 2012, Def 3.2)
# ============================================================================

"""
    StatePotentialGameProblem{T, S} <: AbstractStatePotentialGame{T}

State-based potential game with a finite state space X and Markovian transition.

# Fields
- `n_players`: number of agents N = {1, …, n}
- `state_space`: finite state space X represented as a `Vector{S}`
- `action_spaces`: per-player action sets; `action_spaces[i]` is Aᵢ
- `utility_functions`: `utility_functions[i](a, x)` returns Uᵢ(a, x) ∈ ℝ,
  where `a` is the joint action tuple/vector and `x ∈ state_space`
- `transition`: `transition(a, x)` returns either a state `x' ∈ state_space`
  (deterministic) or a `Dict{S, Float64}` probability distribution over X
- `potential`: `potential(a, x)` returns φ(a, x) ∈ ℝ, the exact potential
  satisfying conditions (i) and (ii) of Marden (2012) Def 3.2

# References
Marden, J.R. (2012). State based potential games. *Automatica* 48(12), 3075–3088.
"""
struct StatePotentialGameProblem{T, S} <: AbstractStatePotentialGame{T}
    n_players::Int
    state_space::Vector{S}
    action_spaces::Vector{<:AbstractVector}
    utility_functions::Vector{Function}
    transition::Function
    potential::Function
end

n_players(g::StatePotentialGameProblem) = g.n_players

"""
    StatePotentialGameProblem(state_space, action_spaces, utility_functions,
                              transition, potential; T=Float64)

Construct a state-based potential game (Marden 2012, Def 3.2).

# Arguments
- `state_space`: finite set X as a `Vector`; states can be any type `S`
- `action_spaces`: `Vector` of per-player action sets, one per player
- `utility_functions`: `Vector` of utility functions; `utility_functions[i](a, x)`
  where `a` is the joint action and `x ∈ state_space`
- `transition`: `(a, x) -> x'` or `(a, x) -> Dict(x' => prob, ...)`
- `potential`: `(a, x) -> φ` satisfying Def 3.2 of Marden (2012)
- `T`: numeric type (default `Float64`)
"""
function StatePotentialGameProblem(
    state_space::Vector{S},
    action_spaces::Vector{<:AbstractVector},
    utility_functions::Vector{Function},
    transition::Function,
    potential::Function;
    T::Type = Float64
) where {S}
    n = length(action_spaces)
    @assert length(utility_functions) == n "Need one utility function per player"
    @assert !isempty(state_space)           "State space must be non-empty"
    StatePotentialGameProblem{T, S}(n, state_space, action_spaces,
                                    utility_functions, transition, potential)
end

function Base.show(io::IO, g::StatePotentialGameProblem{T}) where {T}
    print(io, "StatePotentialGameProblem{$T} with $(g.n_players) players, ",
          "|X|=$(length(g.state_space))")
end

function Base.show(io::IO, ::MIME"text/plain", g::StatePotentialGameProblem{T}) where {T}
    println(io, "StatePotentialGameProblem{$T}")
    println(io, "  Players      : ", g.n_players)
    println(io, "  |X|          : ", length(g.state_space))
    println(io, "  |Aᵢ|         : ", join(length.(g.action_spaces), ", "))
end

# ============================================================================
# OrdinalPotentialGameProblem — ordinal state-based potential game (Marden 2012)
# ============================================================================

"""
    OrdinalPotentialGameProblem{T, S} <: AbstractOrdinalPotentialGame{T}

Ordinal state-based potential game with a finite state space X.

Unlike `StatePotentialGameProblem`, the potential only needs to satisfy the
*ordinal* (sign-preserving) condition, not the exact-equality condition.
A recurrent state equilibrium is guaranteed to exist by Lemma 3.1 of Marden (2012).

# Fields
- `n_players`: number of agents
- `state_space`: finite state space X
- `action_spaces`: per-player action sets {Aᵢ}
- `utility_functions`: `utility_functions[i](a, x)` returns Uᵢ(a, x) ∈ ℝ
- `transition`: `(a, x) -> x'` or `(a, x) -> Dict(x' => prob, ...)`
- `ordinal_potential`: `(a, x) -> P(a, x)` ∈ ℝ, ordinal potential satisfying
  Uᵢ(a'ᵢ, a₋ᵢ, x) − Uᵢ(a, x) > 0  ⟹  P(a'ᵢ, a₋ᵢ, x) − P(a, x) > 0

# References
Marden, J.R. (2012). State based potential games. *Automatica* 48(12), 3075–3088.
"""
struct OrdinalPotentialGameProblem{T, S} <: AbstractOrdinalPotentialGame{T}
    n_players::Int
    state_space::Vector{S}
    action_spaces::Vector{<:AbstractVector}
    utility_functions::Vector{Function}
    transition::Function
    ordinal_potential::Function
end

n_players(g::OrdinalPotentialGameProblem) = g.n_players

"""
    OrdinalPotentialGameProblem(state_space, action_spaces, utility_functions,
                                transition, ordinal_potential; T=Float64)

Construct an ordinal state-based potential game (Marden 2012, Section 3.3).

Same interface as `StatePotentialGameProblem`; `ordinal_potential` only needs to
preserve the sign of unilateral utility differences, not match their magnitude.
"""
function OrdinalPotentialGameProblem(
    state_space::Vector{S},
    action_spaces::Vector{<:AbstractVector},
    utility_functions::Vector{Function},
    transition::Function,
    ordinal_potential::Function;
    T::Type = Float64
) where {S}
    n = length(action_spaces)
    @assert length(utility_functions) == n "Need one utility function per player"
    @assert !isempty(state_space)           "State space must be non-empty"
    OrdinalPotentialGameProblem{T, S}(n, state_space, action_spaces,
                                      utility_functions, transition, ordinal_potential)
end

function Base.show(io::IO, g::OrdinalPotentialGameProblem{T}) where {T}
    print(io, "OrdinalPotentialGameProblem{$T} with $(g.n_players) players, ",
          "|X|=$(length(g.state_space))")
end

function Base.show(io::IO, ::MIME"text/plain", g::OrdinalPotentialGameProblem{T}) where {T}
    println(io, "OrdinalPotentialGameProblem{$T}")
    println(io, "  Players      : ", g.n_players)
    println(io, "  |X|          : ", length(g.state_space))
    println(io, "  |Aᵢ|         : ", join(length.(g.action_spaces), ", "))
end

# ============================================================================
# LexicographicGameProblem — lexicographic general sum game (Miller & Mitra 2022)
# ============================================================================

"""
    LexicographicGameProblem{T} <: AbstractLexicographicGame{T}

Lexicographic general sum game (LG). Each agent i has a two-part cost

    Jᵢ(z) = (Jᵢᶜᵒˡ(z), Jᵢᵖᵉʳ(z))  ∈  ℝ²

ordered lexicographically (≼): minimize collision first, personal cost second.

    Jᵢᶜᵒˡ(z) = Σⱼ≠ᵢ fᵢⱼ(zᵢ, zⱼ)            (pairwise collision cost, ≥ 0)
    Jᵢᵖᵉʳ(z) = gᵢ(zᵢ) − Σⱼ≠ᵢ gⱼ(zⱼ)        (personal cost with zero-sum term)

Any LG is an ordinal potential game with potential P(z) = ⟨½ Σⱼ Jⱼᶜᵒˡ(z), Σⱼ gⱼ(zⱼ)⟩
(Proposition 1, Miller & Mitra 2022), guaranteeing a pure-strategy Nash equilibrium.

# Fields
- `n_players`: number of agents N = {1, …, n}
- `forward_game`: underlying `GameProblem{T}` providing dynamics, time horizon,
  initial states, and structural constraints (built via `PDGNEProblem`)
- `collision_cost_pairs`: `n×n` matrix of pairwise cost functions;
  `collision_cost_pairs[i,j](z_i, z_j)` evaluates fᵢⱼ(zᵢ, zⱼ) ≥ 0;
  must be symmetric (fᵢⱼ = fⱼᵢ); diagonal entries are unused
- `personal_costs`: `personal_costs[i](z_i)` evaluates gᵢ(zᵢ) for player i;
  `z_i` is the trajectory of player i (e.g. a `Vector` of state vectors)

# References
Miller, K. & Mitra, S. (2022). Multi-agent motion planning using differential games
with lexicographic preferences. *IEEE CDC*, pp. 5751–5757.
"""
struct LexicographicGameProblem{T} <: AbstractLexicographicGame{T}
    n_players::Int
    forward_game::GameProblem{T}
    collision_cost_pairs::Matrix{Function}
    personal_costs::Vector{Function}
end

n_players(g::LexicographicGameProblem)       = g.n_players
state_dim(g::LexicographicGameProblem)        = state_dim(g.forward_game)
state_dim(g::LexicographicGameProblem, i)     = state_dim(g.forward_game, i)
control_dim(g::LexicographicGameProblem)      = control_dim(g.forward_game)
control_dim(g::LexicographicGameProblem, i)   = control_dim(g.forward_game, i)
n_steps(g::LexicographicGameProblem)          = n_steps(g.forward_game)

"""
    collision_cost(g, i, z_i, z_minus_i) -> T

Player i's collision cost Jᵢᶜᵒˡ(z) = Σⱼ≠ᵢ fᵢⱼ(zᵢ, zⱼ).
`z_minus_i[k]` is the trajectory of the k-th opponent (in ascending player-index order,
skipping i).
"""
function collision_cost(g::LexicographicGameProblem{T},
                        i::Int, z_i, z_minus_i) where {T}
    val = zero(T)
    k = 1
    for j in 1:g.n_players
        j == i && continue
        val += g.collision_cost_pairs[i, j](z_i, z_minus_i[k])
        k += 1
    end
    return val
end

"""
    personal_cost(g, i, z_i, z_minus_i) -> T

Player i's personal cost Jᵢᵖᵉʳ(z) = gᵢ(zᵢ) − Σⱼ≠ᵢ gⱼ(zⱼ).
The subtracted terms make the personal component zero-sum across agents.
"""
function personal_cost(g::LexicographicGameProblem{T},
                       i::Int, z_i, z_minus_i) where {T}
    val = g.personal_costs[i](z_i)
    k = 1
    for j in 1:g.n_players
        j == i && continue
        val -= g.personal_costs[j](z_minus_i[k])
        k += 1
    end
    return val
end

"""
    lexicographic_cost(g, i, z_i, z_minus_i) -> Tuple{T, T}

Full lexicographic cost (Jᵢᶜᵒˡ, Jᵢᵖᵉʳ) for player i.
"""
function lexicographic_cost(g::LexicographicGameProblem{T},
                             i::Int, z_i, z_minus_i) where {T}
    return (collision_cost(g, i, z_i, z_minus_i),
            personal_cost(g,  i, z_i, z_minus_i))
end

"""
    ordinal_potential(g, z_by_player) -> Tuple{T, T}

Evaluate the ordinal potential P(z) = ⟨½ Σⱼ Jⱼᶜᵒˡ(z), Σⱼ gⱼ(zⱼ)⟩.
`z_by_player[i]` is the trajectory for player i.
"""
function ordinal_potential(g::LexicographicGameProblem{T},
                           z_by_player::AbstractVector) where {T}
    @assert length(z_by_player) == g.n_players
    col = zero(T)
    per = zero(T)
    for i in 1:g.n_players
        z_i  = z_by_player[i]
        z_mi = [z_by_player[j] for j in 1:g.n_players if j != i]
        col += collision_cost(g, i, z_i, z_mi)
        per += g.personal_costs[i](z_i)
    end
    return (col / 2, per)
end

"""
    LexicographicGameProblem(players, collision_cost_pairs, personal_costs,
                             shared_constraints, tf, dt)
    LexicographicGameProblem(players, collision_cost_pairs, personal_costs, tf, dt)

Construct a lexicographic general sum game (Miller & Mitra 2022, Def 1).

# Arguments
- `players`: `Vector{PlayerSpec{T}}` — one per agent (dynamics, initial state, objective)
- `collision_cost_pairs`: `n×n` `Matrix{Function}`; `[i,j](z_i, z_j)` is fᵢⱼ ≥ 0,
  symmetric (fᵢⱼ = fⱼᵢ); diagonal entries are ignored
- `personal_costs`: `Vector{Function}`; `personal_costs[i](z_i)` is gᵢ(zᵢ)
- `shared_constraints`: inter-player constraints (optional, default `[]`)
- `tf`, `dt`: final time and time step
"""
function LexicographicGameProblem(
    players::Vector{PlayerSpec{T}},
    collision_cost_pairs::Matrix{Function},
    personal_costs::Vector{Function},
    shared_constraints::AbstractVector,
    tf::T, dt::T
) where {T}
    n = length(players)
    @assert size(collision_cost_pairs) == (n, n) "collision_cost_pairs must be $n×$n"
    @assert length(personal_costs) == n          "Need one personal cost per player"
    forward = PDGNEProblem(players, shared_constraints, tf, dt)
    return LexicographicGameProblem{T}(n, forward, collision_cost_pairs, personal_costs)
end

LexicographicGameProblem(
    players::Vector{PlayerSpec{T}},
    collision_cost_pairs::Matrix{Function},
    personal_costs::Vector{Function},
    tf::T, dt::T
) where {T} = LexicographicGameProblem(players, collision_cost_pairs, personal_costs,
                                        AbstractSharedConstraint[], tf, dt)

function Base.show(io::IO, g::LexicographicGameProblem{T}) where {T}
    print(io, "LexicographicGameProblem{$T} with $(g.n_players) players [Ordinal Potential]")
end

function Base.show(io::IO, ::MIME"text/plain", g::LexicographicGameProblem{T}) where {T}
    println(io, "LexicographicGameProblem{$T}")
    println(io, "  Players       : ", g.n_players)
    println(io, "  State dim     : ", state_dim(g))
    println(io, "  Control dim   : ", control_dim(g))
    println(io, "  Time horizon  : ", g.forward_game.time_horizon)
    println(io, "  Properties:")
    println(io, "    Ordinal pot.: true (Proposition 1, Miller & Mitra 2022)")
    println(io, "    Pure NE     : guaranteed (Proposition 2)")
end

# ============================================================================
# ConvexGameProblem — general convex game (Rosen 1965)
# ============================================================================

"""
    ConvexGameProblem{T} <: AbstractConvexGame{T}

General convex N-player game: all player objectives are convex in own decisions
and all constraint sets (private and shared) are convex.

# Convexity verification
- **LQ objectives** (`LQStageCost` + `LQTerminalCost`): verified automatically.
  Their constructors already enforce R ≻ 0, Q ≽ 0, Qf ≽ 0, which suffices for
  convexity in the control when dynamics are linear.
- **Constraints**: verified automatically by checking `is_convex(c)` on every
  constraint. `ControlBounds` and `StateBounds` are marked convex; all others
  default to `false` and require `assume_convex=true`.
- **Nonlinear objectives or non-convex-tagged constraints**: require the caller
  to pass `assume_convex=true` to suppress the check.

# Fields
- `n_players`: number of players
- `forward_game`: underlying `GameProblem{T}` (dynamics, objectives, constraints)
- `is_strictly_convex`: `true` if the pseudo-gradient F = (∇ₓᵢJᵢ)ᵢ satisfies
  Rosen's diagonal strict convexity condition, guaranteeing a *unique* NE

# Equilibrium guarantees
- **Existence**: always guaranteed (objectives convex + closed convex sets)
- **Uniqueness**: guaranteed when `is_strictly_convex == true` (Rosen 1965, Thm 2)

# References
Rosen, J.B. (1965). Existence and uniqueness of equilibrium points for concave
N-person games. *Econometrica* 33(3), 520–534.
"""
struct ConvexGameProblem{T} <: AbstractConvexGame{T}
    n_players::Int
    forward_game::GameProblem{T}
    is_strictly_convex::Bool
end

n_players(g::ConvexGameProblem)              = g.n_players
state_dim(g::ConvexGameProblem)              = state_dim(g.forward_game)
state_dim(g::ConvexGameProblem, i::Int)      = state_dim(g.forward_game, i)
control_dim(g::ConvexGameProblem)            = control_dim(g.forward_game)
control_dim(g::ConvexGameProblem, i::Int)    = control_dim(g.forward_game, i)
n_steps(g::ConvexGameProblem)               = n_steps(g.forward_game)
is_strictly_convex_game(g::ConvexGameProblem) = g.is_strictly_convex

# ─── Internal convexity check ─────────────────────────────────────────────────

# Returns nothing if the game is verifiably convex, throws with a descriptive
# message if any component cannot be verified.
function _verify_convexity(game::GameProblem)
    for obj in game.objectives
        id = obj.player_id
        if !(obj.stage_cost isa LQStageCost)
            error(
                "Player $id has a $(typeof(obj.stage_cost)) stage cost; " *
                "convexity cannot be verified automatically. " *
                "Pass `assume_convex=true` to assert it."
            )
        end
        if !(obj.terminal_cost isa LQTerminalCost)
            error(
                "Player $id has a $(typeof(obj.terminal_cost)) terminal cost; " *
                "convexity cannot be verified automatically. " *
                "Pass `assume_convex=true` to assert it."
            )
        end
    end
    for c in game.private_constraints
        is_convex(c) || error(
            "Private constraint $(typeof(c)) is not marked convex " *
            "(`is_convex` returns false). Pass `assume_convex=true` to assert it."
        )
    end
    for c in game.shared_constraints
        is_convex(c) || error(
            "Shared constraint $(typeof(c)) is not marked convex " *
            "(`is_convex` returns false). Pass `assume_convex=true` to assert it."
        )
    end
    return nothing
end

# ─── Constructors ─────────────────────────────────────────────────────────────

"""
    ConvexGameProblem(game; strict=false, assume_convex=false)

Wrap an existing `GameProblem{T}` as a convex game.

Automatically verifies convexity for LQ objectives and `is_convex`-tagged
constraints. Pass `assume_convex=true` to skip the check for nonlinear problems.
Set `strict=true` to assert Rosen's diagonal strict convexity (unique NE).
"""
function ConvexGameProblem(
    game::GameProblem{T};
    strict::Bool        = false,
    assume_convex::Bool = false
) where {T}
    assume_convex || _verify_convexity(game)
    return ConvexGameProblem{T}(n_players(game), game, strict)
end

"""
    ConvexGameProblem(players, shared_constraints, tf, dt; strict=false, assume_convex=false)
    ConvexGameProblem(players, tf, dt; strict=false, assume_convex=false)

Construct a convex game from `PlayerSpec` entries, building a `PDGNEProblem` internally.

# Arguments
- `players`: `Vector{PlayerSpec{T}}` — one per player
- `shared_constraints`: inter-player constraints (optional, default `[]`)
- `tf`, `dt`: final time and time step
- `strict`: assert diagonal strict convexity → unique NE (Rosen 1965, Thm 2)
- `assume_convex`: skip convexity check (required for nonlinear objectives)
"""
function ConvexGameProblem(
    players::Vector{PlayerSpec{T}},
    shared_constraints::AbstractVector,
    tf::T, dt::T;
    strict::Bool        = false,
    assume_convex::Bool = false
) where {T}
    forward = PDGNEProblem(players, shared_constraints, tf, dt)
    return ConvexGameProblem(forward; strict=strict, assume_convex=assume_convex)
end

ConvexGameProblem(
    players::Vector{PlayerSpec{T}},
    tf::T, dt::T;
    strict::Bool        = false,
    assume_convex::Bool = false
) where {T} = ConvexGameProblem(players, AbstractSharedConstraint[], tf, dt;
                                 strict=strict, assume_convex=assume_convex)

# ─── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, g::ConvexGameProblem{T}) where {T}
    tag = g.is_strictly_convex ? "Strictly Convex" : "Convex"
    print(io, "ConvexGameProblem{$T} with $(g.n_players) players [$tag]")
end

function Base.show(io::IO, ::MIME"text/plain", g::ConvexGameProblem{T}) where {T}
    println(io, "ConvexGameProblem{$T}")
    println(io, "  Players         : ", g.n_players)
    println(io, "  State dim       : ", state_dim(g))
    println(io, "  Control dim     : ", control_dim(g))
    println(io, "  Dynamics        : ", g.forward_game.dynamics)
    println(io, "  Time horizon    : ", g.forward_game.time_horizon)
    println(io, "  Private constr. : ", length(g.forward_game.private_constraints))
    println(io, "  Shared constr.  : ", length(g.forward_game.shared_constraints))
    println(io, "  Properties:")
    println(io, "    Strictly cvx  : ", g.is_strictly_convex)
    println(io, "    NE existence  : guaranteed (Kakutani)")
    println(io, "    NE uniqueness : ", g.is_strictly_convex ?
                "guaranteed (Rosen 1965, Thm 2)" : "not guaranteed")
end