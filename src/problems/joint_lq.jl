# ============================================================================
# problems/joint_lq.jl
#
# Full joint-control quadratic player costs for LQ games.
#
# `LQStageCost` (GNEP.jl) only lets player i's cost depend on its OWN control.
# `JointLQStageCost` lets it depend on every player's control through the
# stacked joint control u = (u¹,…,uᴺ) ∈ ℝᵐ:
#
#   ℓᵢ(x,u,k) = ½xᵀQx + xᵀMu + ½uᵀRu + qᵀx + rᵀu + c
#
# with Q: n×n, M: n×m, R: m×m (symmetric, not necessarily PD — only the
# own-control diagonal block must satisfy the second-order Nash condition,
# which solvers check), q: n, r: m.
#
# CONVENTION: unlike every other stage cost, `evaluate_stage_cost` for this
# type takes the JOINT control vector. It is consumed by FNELQ/SFNELQ; other
# solvers that pass only the player's own control do not support it.
# ============================================================================

_at(x::AbstractVector{<:AbstractArray}, k::Int) = x[k]
_at(x::AbstractArray, ::Int) = x

"""
    JointLQStageCost{T, QM, RM, MM, QVM, RVM} <: AbstractStageCost

Quadratic stage cost over the joint state and joint control. Each field is
either a single array (LTI) or a vector of arrays, one per time step (LTV).
"""
struct JointLQStageCost{T, QM, RM, MM, QVM, RVM} <: AbstractStageCost
    Q::QM
    R::RM
    M::MM
    q::QVM
    r::RVM
    c::T
end

function JointLQStageCost(Q::Matrix{T}, R::Matrix{T};
        M::Matrix{T} = zeros(T, size(Q, 1), size(R, 1)),
        q::Vector{T} = zeros(T, size(Q, 1)),
        r::Vector{T} = zeros(T, size(R, 1)),
        c::T = zero(T)) where {T}
    n, m = size(M)
    @assert size(Q) == (n, n) && issymmetric(Q) "Q must be symmetric n×n"
    @assert size(R) == (m, m) && issymmetric(R) "R must be symmetric m×m"
    @assert length(q) == n && length(r) == m "q/r dimension mismatch"
    JointLQStageCost{T, Matrix{T}, Matrix{T}, Matrix{T}, Vector{T}, Vector{T}}(Q, R, M, q, r, c)
end

function JointLQStageCost(Q::Vector{Matrix{T}}, R::Vector{Matrix{T}};
        M::Vector{Matrix{T}} = [zeros(T, size(Q[1], 1), size(R[1], 1)) for _ in Q],
        q::Vector{Vector{T}} = [zeros(T, size(Q[1], 1)) for _ in Q],
        r::Vector{Vector{T}} = [zeros(T, size(R[1], 1)) for _ in Q],
        c::T = zero(T)) where {T}
    N = length(Q)
    @assert all(length.([R, M, q, r]) .== N) "All sequences must have length $N"
    for k in 1:N
        @assert issymmetric(Q[k]) && issymmetric(R[k]) "Q/R must be symmetric"
    end
    JointLQStageCost{T, typeof(Q), typeof(R), typeof(M), typeof(q), typeof(r)}(Q, R, M, q, r, c)
end

get_Q(c::JointLQStageCost, k::Int) = _at(c.Q, k)
get_R(c::JointLQStageCost, k::Int) = _at(c.R, k)
get_M(c::JointLQStageCost, k::Int) = _at(c.M, k)
get_q(c::JointLQStageCost, k::Int) = _at(c.q, k)
get_r(c::JointLQStageCost, k::Int) = _at(c.r, k)
is_ltv(c::JointLQStageCost) = c.Q isa AbstractVector
is_separable(::JointLQStageCost) = false

"""
    evaluate_stage_cost(cost::JointLQStageCost, x, u_joint, p, t)

`u_joint` is the stacked control of ALL players.
"""
function evaluate_stage_cost(cost::JointLQStageCost, x, u, p, t)
    Q = get_Q(cost, t); R = get_R(cost, t); M = get_M(cost, t)
    return 0.5 * (x' * Q * x + u' * R * u) + x' * M * u +
           get_q(cost, t)' * x + get_r(cost, t)' * u + cost.c
end

"""
    JointLQGameProblem(A, B, Q, R, Qf, x0, tf; dt, M, q, r, qf) -> GameProblem

LTI LQ game whose player costs depend on the joint control. `R[i]` is the full
`m×m` control weight of player `i` (`m = Σ mⱼ`), `M[i]` is `n×m`, `r[i]` has
length `m`; `qf[i]` is an optional terminal linear term.
"""
function JointLQGameProblem(
    A::Matrix{T}, B::Vector{Matrix{T}},
    Q::Vector{Matrix{T}}, R::Vector{Matrix{T}}, Qf::Vector{Matrix{T}},
    x0::Vector{T}, tf::T;
    dt::T = T(0.01),
    M::Union{Vector{Matrix{T}}, Nothing} = nothing,
    q::Union{Vector{Vector{T}}, Nothing} = nothing,
    r::Union{Vector{Vector{T}}, Nothing} = nothing,
    qf::Union{Vector{Vector{T}}, Nothing} = nothing
) where {T<:Real}
    n = size(A, 1)
    np = length(B)
    @assert length(Q) == np && length(R) == np && length(Qf) == np
    @assert length(x0) == n
    dynamics = LinearDynamics(A, B)
    control_dims = dynamics.control_dims
    m = sum(control_dims)
    objectives = map(1:np) do i
        @assert size(R[i]) == (m, m) "R[$i] must be the full $m×$m joint-control weight"
        stage = JointLQStageCost(Q[i], R[i];
            M = isnothing(M) ? zeros(T, n, m) : M[i],
            q = isnothing(q) ? zeros(T, n) : q[i],
            r = isnothing(r) ? zeros(T, m) : r[i])
        term = LQTerminalCost(Qf[i], isnothing(qf) ? zeros(T, n) : qf[i], zero(T))
        PlayerObjective(i, stage, term)
    end
    th = DiscreteTime(tf, dt)
    control_offsets = [0; cumsum(control_dims)[1:end-1]]
    coupling_graph = CouplingGraph(sparse(trues(np, np)), Vector{Int}[], nothing)
    metadata = GameMetadata([n], control_dims, [0], control_offsets, coupling_graph, false, nothing)
    return GameProblem{T}(np, objectives, dynamics, x0,
        AbstractPrivateConstraint[], AbstractSharedConstraint[], th, metadata)
end
