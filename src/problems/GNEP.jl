# ============================================================================
# Cost Types — unified LTI/LTV via type parameters
# ============================================================================
#
# LQStageCost, LQTerminalCost, and PlayerObjective are defined here (not in
# objectives.jl) because they require parametric type parameters for LTV
# support. Evaluation methods for all three live in objectives.jl and are
# compiled after this file is loaded (include order: GNEP.jl before
# objectives.jl is therefore WRONG — see DifferentialGamesBase.jl include
# order which must be: objectives.jl abstract types first, then GNEP.jl).
#
# Include order in DifferentialGamesBase.jl must be:
#   include("objectives.jl")      ← abstract types + Diagonal/Nonlinear structs
#   include("problems/GNEP.jl")   ← LQStageCost, LQTerminalCost, PlayerObjective, GameProblem
#
# QM ∈ {Matrix{T}, Vector{Matrix{T}}} selects LTI vs LTV.
# Always use get_Q(cost, k), get_R(cost, k), etc. in solver hot loops.

# ============================================================================
# LQStageCost
# ============================================================================

"""
    LQStageCost{T, QM, RM, MM, QVM, RVM} <: AbstractStageCost

Unified LTI/LTV quadratic stage cost.

# Type Parameters
- `QM`  : `Matrix{T}` (LTI) or `Vector{Matrix{T}}` (LTV)
- `RM`  : `Matrix{T}` (LTI) or `Vector{Matrix{T}}` (LTV)
- `MM`  : `Matrix{T}` (LTI) or `Vector{Matrix{T}}` (LTV) — cross term
- `QVM` : `Vector{T}` (LTI) or `Vector{Vector{T}}` (LTV) — linear state cost
- `RVM` : `Vector{T}` (LTI) or `Vector{Vector{T}}` (LTV) — linear control cost

# Mathematical Form
ℓᵢ(x, uᵢ, k) = ½ xᵀQ(k)x + ½ uᵢᵀR(k)uᵢ + xᵀM(k)uᵢ + q(k)ᵀx + r(k)ᵀuᵢ + c

# Construction
```julia
# LTI — positional (new)
cost = LQStageCost(Q, R, M, q, r, c)
# LTI — keyword convenience (preserves existing call sites)
cost = LQStageCost(Q, R; M=nothing, q=nothing, r=nothing, c=0.0)
# LTV
cost = LQStageCost(Q_seq, R_seq)
cost = LQStageCost(Q_seq, R_seq, M_seq, q_seq, r_seq)
```

# Accessors
Always use `get_Q(cost, k)`, `get_R(cost, k)`, etc. — never access fields directly.
"""
struct LQStageCost{T, QM, RM, MM, QVM, RVM} <: AbstractStageCost
    Q::QM
    R::RM
    M::MM
    q::QVM
    r::RVM
    c::T   # Scalar constant offset. For LTV, treated as time-invariant.

    function LQStageCost{T, QM, RM, MM, QVM, RVM}(
        Q::QM, R::RM, M::MM, q::QVM, r::RVM, c::T
    ) where {T, QM, RM, MM, QVM, RVM}
        new{T, QM, RM, MM, QVM, RVM}(Q, R, M, q, r, c)
    end
end

# ─── LTI: positional constructor ─────────────────────────────────────────────

function LQStageCost(
    Q::Matrix{T},
    R::Matrix{T},
    M::Matrix{T},
    q::Vector{T},
    r::Vector{T},
    c::T
) where {T}
    n_x, n_u = size(M)
    @assert size(Q) == (n_x, n_x) "Q must be (n_x × n_x)"
    @assert issymmetric(Q) "Q must be symmetric"
    @assert size(R) == (n_u, n_u) "R must be (n_u × n_u)"
    @assert issymmetric(R) "R must be symmetric"
    @assert isposdef(R) "R must be positive definite (unique best response)"
    eigvals_Q = eigvals(Symmetric(Q))
    @assert all(eigvals_Q .>= -sqrt(eps(T)) * maximum(abs.(eigvals_Q))) "Q must be positive semidefinite"
    @assert length(q) == n_x "q must have length n_x"
    @assert length(r) == n_u "r must have length n_u"
    LQStageCost{T, Matrix{T}, Matrix{T}, Matrix{T}, Vector{T}, Vector{T}}(Q, R, M, q, r, c)
end

# ─── LTI: keyword convenience — preserves all existing call sites ─────────────

function LQStageCost(
    Q::Matrix{T},
    R::Matrix{T};
    M::Union{Matrix{T}, Nothing} = nothing,
    q::Union{Vector{T}, Nothing} = nothing,
    r::Union{Vector{T}, Nothing} = nothing,
    c::T = zero(T)
) where {T}
    n_x = size(Q, 1)
    n_u = size(R, 1)
    M_mat = M === nothing ? zeros(T, n_x, n_u) : M
    q_vec = q === nothing ? zeros(T, n_x)       : q
    r_vec = r === nothing ? zeros(T, n_u)       : r
    return LQStageCost(Q, R, M_mat, q_vec, r_vec, c)
end

# ─── LTV constructor ─────────────────────────────────────────────────────────

function LQStageCost(
    Q_seq::Vector{Matrix{T}},
    R_seq::Vector{Matrix{T}},
    M_seq::Vector{Matrix{T}},
    q_seq::Vector{Vector{T}},
    r_seq::Vector{Vector{T}}
) where {T}
    N = length(Q_seq)
    @assert N > 0 "Cost sequences must be non-empty"
    @assert length(R_seq) == N && length(M_seq) == N &&
            length(q_seq) == N && length(r_seq) == N "All sequences must have length $N"
    n_x = size(Q_seq[1], 1)
    n_u = size(R_seq[1], 1)
    for k in 1:N
        @assert size(Q_seq[k]) == (n_x, n_x) && issymmetric(Q_seq[k]) "Q_seq[$k] must be symmetric ($n_x × $n_x)"
        @assert size(R_seq[k]) == (n_u, n_u) && issymmetric(R_seq[k]) "R_seq[$k] must be symmetric ($n_u × $n_u)"
        @assert isposdef(R_seq[k]) "R_seq[$k] must be positive definite"
    end
    LQStageCost{T, Vector{Matrix{T}}, Vector{Matrix{T}}, Vector{Matrix{T}},
                Vector{Vector{T}}, Vector{Vector{T}}}(
        Q_seq, R_seq, M_seq, q_seq, r_seq, zero(T)
    )
end

function LQStageCost(Q_seq::Vector{Matrix{T}}, R_seq::Vector{Matrix{T}}) where {T}
    N = length(Q_seq)
    n_x = size(Q_seq[1], 1)
    n_u = size(R_seq[1], 1)
    LQStageCost(Q_seq, R_seq,
        [zeros(T, n_x, n_u) for _ in 1:N],
        [zeros(T, n_x)      for _ in 1:N],
        [zeros(T, n_u)      for _ in 1:N])
end

# ─── Structural query ─────────────────────────────────────────────────────────

is_ltv(::LQStageCost{T, Matrix{T}}) where {T}         = false
is_ltv(::LQStageCost{T, Vector{Matrix{T}}}) where {T} = true

# ─── Accessors ────────────────────────────────────────────────────────────────

get_Q(c::LQStageCost{T, Matrix{T}}, k::Int) where {T}         = c.Q
get_Q(c::LQStageCost{T, Vector{Matrix{T}}}, k::Int) where {T} = c.Q[k]

get_R(c::LQStageCost{T, Matrix{T}}, k::Int) where {T}         = c.R
get_R(c::LQStageCost{T, Vector{Matrix{T}}}, k::Int) where {T} = c.R[k]

get_M(c::LQStageCost{T, Matrix{T}}, k::Int) where {T}         = c.M
get_M(c::LQStageCost{T, Vector{Matrix{T}}}, k::Int) where {T} = c.M[k]

# q and r: dispatch on QVM/RVM (5th/6th type parameter)
get_q(c::LQStageCost{T, <:Any, <:Any, <:Any, Vector{T}, <:Any}, k::Int) where {T}         = c.q
get_q(c::LQStageCost{T, <:Any, <:Any, <:Any, Vector{Vector{T}}, <:Any}, k::Int) where {T} = c.q[k]

get_r(c::LQStageCost{T, <:Any, <:Any, <:Any, <:Any, Vector{T}}, k::Int) where {T}         = c.r
get_r(c::LQStageCost{T, <:Any, <:Any, <:Any, <:Any, Vector{Vector{T}}}, k::Int) where {T} = c.r[k]

# ============================================================================
# LQTerminalCost
# ============================================================================

"""
    LQTerminalCost{T} <: AbstractTerminalCost

Quadratic terminal cost: Vᵢ(x(tf)) = ½ x(tf)ᵀ Qf x(tf) + qfᵀ x(tf) + cf

# Fields
- `Qf::Matrix{T}` : Terminal cost matrix (n × n), symmetric, PSD
- `qf::Vector{T}` : Linear terminal cost (n,)
- `cf::T`         : Constant offset
"""
struct LQTerminalCost{T} <: AbstractTerminalCost
    Qf::Matrix{T}
    qf::Vector{T}
    cf::T   # Preserved from original — referenced by evaluate_terminal_cost

    function LQTerminalCost(Qf::Matrix{T}, qf::Vector{T}, cf::T) where {T}
        n = size(Qf, 1)
        @assert size(Qf) == (n, n) "Qf must be square"
        @assert issymmetric(Qf) "Qf must be symmetric"
        @assert length(qf) == n "qf must have length $n"
        eigvals_Qf = eigvals(Symmetric(Qf))
        @assert all(eigvals_Qf .>= -sqrt(eps(T)) * maximum(abs.(eigvals_Qf))) "Qf must be positive semidefinite"
        new{T}(Qf, qf, cf)
    end
end

# Convenience constructors — both forms preserved
LQTerminalCost(Qf::Matrix{T}; qf::Union{Vector{T}, Nothing} = nothing, cf::T = zero(T)) where {T} =
    LQTerminalCost(Qf, qf === nothing ? zeros(T, size(Qf, 1)) : qf, cf)

LQTerminalCost(Qf::Matrix{T}, qf::Vector{T}) where {T} =
    LQTerminalCost(Qf, qf, zero(T))

# ============================================================================
# Universal Game Problem Container
# ============================================================================

"""
    GameProblem{T}

Universal game problem representation.
"""
struct GameProblem{T}
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
        @assert n_players > 0 "Must have at least one player"
        @assert length(objectives) == n_players "Must have objective for each player"
        @assert allunique(obj.player_id for obj in objectives) "Duplicate player IDs"
        @assert all(1 ≤ obj.player_id ≤ n_players for obj in objectives) "Invalid player IDs"
        all_ids = Set(1:n_players)
        for c in private_constraints
            @assert c.player in all_ids "Private constraint references invalid player"
        end
        for c in shared_constraints
            @assert all(p in all_ids for p in c.players) "Shared constraint references invalid player"
        end
        new{T}(n_players, objectives, dynamics, initial_state,
               private_constraints, shared_constraints, time_horizon, metadata)
    end
end

# ============================================================================
# Structural queries
# ============================================================================

is_unconstrained(g::GameProblem) =
    isempty(g.private_constraints) && isempty(g.shared_constraints)

is_lq_game(g::GameProblem) =
    g.dynamics isa LinearDynamics &&
    all(obj.stage_cost isa LQStageCost for obj in g.objectives)

is_pd_gnep(g::GameProblem)          = g.dynamics isa SeparableDynamics
has_shared_constraints(g::GameProblem) = !isempty(g.shared_constraints)
is_potential_game(g::GameProblem)   = g.metadata.is_potential
state_dim(g::GameProblem)           = total_state_dim(g.dynamics)
control_dim(g::GameProblem)         = total_control_dim(g.dynamics)

function n_steps(g::GameProblem{T}) where {T}
    th = g.time_horizon
    @assert th isa DiscreteTime "n_steps requires a DiscreteTime horizon"
    return Int(round(th.tf / th.dt))
end

# ============================================================================
# LQGameProblem — LTI constructor (existing call signature preserved)
# ============================================================================

function LQGameProblem(
    A::Matrix{T},
    B::Vector{Matrix{T}},
    Q::Vector{Matrix{T}},
    R::Vector{Matrix{T}},
    Qf::Vector{Matrix{T}},
    x0::Vector{T},
    tf::T;
    dt::T = T(0.01),
    M::Union{Vector{Matrix{T}}, Nothing} = nothing,
    q::Union{Vector{Vector{T}}, Nothing} = nothing,
    r::Union{Vector{Vector{T}}, Nothing} = nothing
) where {T}
    n = size(A, 1)
    n_players = length(B)

    @assert length(Q) == n_players && length(R) == n_players && length(Qf) == n_players
    @assert length(x0) == n "x0 length must match state dimension"

    for i in 1:n_players
        @assert size(Q[i]) == (n, n) && issymmetric(Q[i]) "Q[$i] must be symmetric (n × n)"
        @assert issymmetric(R[i]) && isposdef(R[i]) "R[$i] must be symmetric positive definite"
        @assert size(Qf[i]) == (n, n) && issymmetric(Qf[i]) "Qf[$i] must be symmetric (n × n)"
    end

    dynamics     = LinearDynamics(A, B)
    control_dims = dynamics.control_dims

    objectives = map(1:n_players) do i
        mi = control_dims[i]
        Mi = isnothing(M) ? zeros(T, n, mi) : M[i]
        qi = isnothing(q) ? zeros(T, n)     : q[i]
        ri = isnothing(r) ? zeros(T, mi)    : r[i]
        stage_cost    = LQStageCost(Q[i], R[i], Mi, qi, ri, zero(T))
        terminal_cost = LQTerminalCost(Qf[i])
        PlayerObjective(i, stage_cost, terminal_cost)
    end

    time_horizon    = DiscreteTime(tf, dt)
    state_offsets   = [0]
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    cost_coupling   = sparse(trues(n_players, n_players))
    coupling_graph  = CouplingGraph(cost_coupling, Vector{Int}[], nothing)

    metadata = GameMetadata(
        [n], control_dims, state_offsets, control_offsets,
        coupling_graph, false, nothing
    )

    return GameProblem{T}(
        n_players, objectives, dynamics, x0,
        PrivateConstraint[], SharedConstraint[], time_horizon, metadata
    )
end

# ============================================================================
# LTVLQGameProblem — LTV constructor (new)
# ============================================================================

function LTVLQGameProblem(
    A_seq::Vector{Matrix{T}},
    B_seq::Vector{Vector{Matrix{T}}},
    Q_seq::Vector{Vector{Matrix{T}}},
    R_seq::Vector{Vector{Matrix{T}}},
    Qf::Vector{Matrix{T}},
    x0::Vector{T},
    tf::T;
    dt::T = T(0.01),
    M_seq::Union{Vector{Vector{Matrix{T}}}, Nothing} = nothing,
    q_seq::Union{Vector{Vector{Vector{T}}}, Nothing} = nothing,
    r_seq::Union{Vector{Vector{Vector{T}}}, Nothing} = nothing
) where {T}
    N         = length(A_seq)
    N_from_dt = Int(round(tf / dt))
    @assert N == N_from_dt "A_seq has length $N but tf/dt = $N_from_dt; must be consistent"

    n_players = length(B_seq)
    @assert length(Q_seq) == n_players && length(R_seq) == n_players && length(Qf) == n_players

    n = size(A_seq[1], 1)
    @assert length(x0) == n "x0 length must match state dimension"

    dynamics     = LinearDynamics(A_seq, B_seq)
    control_dims = dynamics.control_dims

    objectives = map(1:n_players) do i
        mi     = control_dims[i]
        Mi_seq = isnothing(M_seq) ? [zeros(T, n, mi) for _ in 1:N] : M_seq[i]
        qi_seq = isnothing(q_seq) ? [zeros(T, n)     for _ in 1:N] : q_seq[i]
        ri_seq = isnothing(r_seq) ? [zeros(T, mi)    for _ in 1:N] : r_seq[i]
        stage_cost    = LQStageCost(Q_seq[i], R_seq[i], Mi_seq, qi_seq, ri_seq)
        terminal_cost = LQTerminalCost(Qf[i])
        PlayerObjective(i, stage_cost, terminal_cost)
    end

    time_horizon    = DiscreteTime(tf, dt)
    state_offsets   = [0]
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    cost_coupling   = sparse(trues(n_players, n_players))
    coupling_graph  = CouplingGraph(cost_coupling, Vector{Int}[], nothing)

    metadata = GameMetadata(
        [n], control_dims, state_offsets, control_offsets,
        coupling_graph, false, nothing
    )

    return GameProblem{T}(
        n_players, objectives, dynamics, x0,
        PrivateConstraint[], SharedConstraint[], time_horizon, metadata
    )
end

# ============================================================================
# PDGNEProblem — unchanged
# ============================================================================

function PDGNEProblem(
    players::Vector{PlayerSpec{T}},
    shared_constraints::AbstractVector,
    tf::T,
    dt::T
) where {T}
    n_players = length(players)
    @assert n_players > 0
    @assert allunique(p.id for p in players)

    objectives   = [p.objective for p in players]
    state_dims   = [p.n for p in players]
    control_dims = [p.m for p in players]

    dynamics      = SeparableDynamics([p.dynamics for p in players], state_dims, control_dims)
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
            cost_coupling[i, i] = true
        end
    end
    constraint_coupling = Vector{Int}[c.players for c in shared_constraints]
    coupling_graph      = CouplingGraph(cost_coupling, constraint_coupling, nothing)

    metadata = GameMetadata(
        state_dims, control_dims, state_offsets, control_offsets,
        coupling_graph, false, nothing
    )

    return GameProblem{T}(
        n_players, objectives, dynamics, initial_state,
        private_constraints, shared_constraints, time_horizon, metadata
    )
end

PDGNEProblem(players::Vector{PlayerSpec{T}}, tf::T, dt::T) where {T} =
    PDGNEProblem(players, SharedConstraint[], tf, dt)

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
# LQStageCost evaluation methods
# (defined here, after the struct, because objectives.jl loads first)
# ============================================================================

function evaluate_stage_cost(cost::LQStageCost, x, u, p, t)
    Q = get_Q(cost, t); R = get_R(cost, t); M = get_M(cost, t)
    q = get_q(cost, t); r = get_r(cost, t)
    return 0.5 * (x' * Q * x + u' * R * u + 2 * x' * M * u) +
           q' * x + r' * u + cost.c
end

function stage_cost_gradient(cost::LQStageCost, x, u, p, t)
    Q = get_Q(cost, t); R = get_R(cost, t); M = get_M(cost, t)
    q = get_q(cost, t); r = get_r(cost, t)
    ∇x = Q * x + M * u + q
    ∇u = R * u + M' * x + r
    return (∇x, ∇u)
end

function stage_cost_hessian(cost::LQStageCost, x, u, p, t)
    return (get_Q(cost, t), get_R(cost, t), get_M(cost, t))
end

# ============================================================================
# LQTerminalCost evaluation methods
# ============================================================================

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
# is_separable — LQStageCost dispatch (must follow struct definition)
# ============================================================================

is_separable(::LQStageCost) = false  # General LQ can couple players via shared Q

# ============================================================================
# Utilities that reference PlayerObjective (must follow struct definition)
# ============================================================================

function total_cost(obj::PlayerObjective, X::Vector, U::Vector, p)
    N = length(U)
    @assert length(X) == N + 1 "X must have length N+1 for N control steps"
    # t is 1-based here; evaluate_stage_cost receives it as the timestep index
    # for LTV accessor dispatch. LTI costs ignore t entirely.
    stage_sum = sum(evaluate_stage_cost(obj.stage_cost, X[t], U[t], p, t) for t in 1:N)
    terminal  = evaluate_terminal_cost(obj.terminal_cost, X[end], p)
    return obj.scaling * (stage_sum + terminal)
end

function diagnose_scaling(obj::PlayerObjective, X::Vector, U::Vector, p)
    N           = length(U)
    stage_costs = [evaluate_stage_cost(obj.stage_cost, X[t], U[t], p, t) for t in 1:N]
    term_cost   = evaluate_terminal_cost(obj.terminal_cost, X[end], p)
    stage_sum   = sum(stage_costs)
    return (
        stage_costs             = stage_costs,
        terminal_cost           = term_cost,
        stage_mean              = stage_sum / N,
        stage_std               = std(stage_costs),
        stage_min               = minimum(stage_costs),
        stage_max               = maximum(stage_costs),
        stage_to_terminal_ratio = stage_sum / (term_cost + eps()),
        total_cost              = obj.scaling * (stage_sum + term_cost)
    )
end