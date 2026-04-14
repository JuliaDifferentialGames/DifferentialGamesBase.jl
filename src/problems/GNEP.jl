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

state_dim(g::GameProblem)   = total_state_dim(g.dynamics)
control_dim(g::GameProblem) = total_control_dim(g.dynamics)

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