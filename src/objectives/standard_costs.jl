# ============================================================================
# standard_costs.jl
#
# Concrete AbstractCostTerm and AbstractTerminalCostTerm implementations.
#
# All terms receive the FULL joint state and control vectors.
# Use player_slice(x, offset, dim) to extract per-player slices.
# All terms are ForwardDiff-compatible.
#
# Include order: must come after cost_terms.jl in DifferentialGamesBase.jl.
#
# Contents:
#   Stage terms:
#     QuadraticStateCost        ½ (x_i - x_ref)ᵀ Q (x_i - x_ref)
#     QuadraticControlCost      ½ uᵢᵀ R uᵢ
#     ProximityCost             softplus collision avoidance
#     CommunicationCost         softplus upper-bound proximity penalty
#     ControlBarrierCost        exponential CBF-like state constraint cost
#
#   Terminal terms:
#     QuadraticTerminalCost     ½ (x_i - x_ref)ᵀ Qf (x_i - x_ref)
#     ProximityTerminalCost     terminal collision avoidance
#
#   Constructors (ergonomic API):
#     track_goal(x_ref, Q; ...)            → QuadraticStateCost
#     regularize_input(R; ...)             → QuadraticControlCost
#     avoid_proximity(; ...)               → ProximityCost
#     maintain_proximity(; ...)            → CommunicationCost
#     terminal_goal(x_ref, Qf; ...)        → QuadraticTerminalCost
# ============================================================================

# ============================================================================
# QuadraticStateCost
# ============================================================================

"""
    QuadraticStateCost <: AbstractCostTerm

Stage cost: ½ (xᵢ - x_ref)ᵀ Q (xᵢ - x_ref)

Reads player i's state slice from the full joint state via `state_offset`
and `state_dim`. Cross-player reference is not supported — use the joint
state offset if you need to penalise a relative state.

# Fields
- `Q`            : (n_i × n_i) weight matrix, symmetric PSD
- `x_ref`        : (n_i,) reference state (can be time-varying if passed via `p`)
- `state_offset` : zero-based offset into full joint state for player i
- `state_dim`    : dimension of player i's state n_i

# Trait Queries
`is_quadratic` → true, `is_separable_term` → true
"""
struct QuadraticStateCost <: AbstractCostTerm
    Q::Matrix{Float64}
    x_ref::Vector{Float64}
    state_offset::Int
    state_dim::Int

    function QuadraticStateCost(
        Q::Matrix{Float64},
        x_ref::Vector{Float64},
        state_offset::Int,
        state_dim::Int
    )
        @assert size(Q) == (state_dim, state_dim) "Q must be ($state_dim × $state_dim)"
        @assert issymmetric(Q)                    "Q must be symmetric"
        @assert length(x_ref) == state_dim        "x_ref must have length $state_dim"
        @assert state_offset >= 0                 "state_offset must be non-negative"
        new(Q, x_ref, state_offset, state_dim)
    end
end

function evaluate_cost_term(t::QuadraticStateCost, x, u, p, ts)
    xi = player_slice(x, t.state_offset, t.state_dim)
    δ  = xi .- t.x_ref
    return 0.5 * dot(δ, t.Q * δ)
end

function cost_term_gradient(t::QuadraticStateCost, x, u, p, ts)
    xi  = player_slice(x, t.state_offset, t.state_dim)
    δ   = xi .- t.x_ref
    ∇xi = t.Q * δ
    ∇x  = zeros(eltype(x), length(x))
    ∇x[t.state_offset+1 : t.state_offset+t.state_dim] = ∇xi
    return (∇x, zeros(eltype(u), length(u)))
end

function cost_term_hessian(t::QuadraticStateCost, x, u, p, ts)
    n  = length(x); m = length(u)
    Hx = zeros(Float64, n, n)
    Hx[t.state_offset+1:t.state_offset+t.state_dim,
       t.state_offset+1:t.state_offset+t.state_dim] = t.Q
    return (Hx, zeros(Float64, m, m), zeros(Float64, n, m))
end

is_quadratic(::QuadraticStateCost)      = true
is_separable_term(::QuadraticStateCost) = true

# ============================================================================
# QuadraticControlCost
# ============================================================================

"""
    QuadraticControlCost <: AbstractCostTerm

Stage cost: ½ uᵢᵀ R uᵢ

Reads player i's control slice from the full joint control via `control_offset`
and `control_dim`.

# Fields
- `R`              : (m_i × m_i) weight matrix, symmetric PD
- `control_offset` : zero-based offset into full joint control for player i
- `control_dim`    : dimension of player i's control m_i
"""
struct QuadraticControlCost <: AbstractCostTerm
    R::Matrix{Float64}
    control_offset::Int
    control_dim::Int

    function QuadraticControlCost(
        R::Matrix{Float64},
        control_offset::Int,
        control_dim::Int
    )
        @assert size(R) == (control_dim, control_dim) "R must be ($control_dim × $control_dim)"
        @assert issymmetric(R) && isposdef(R)          "R must be symmetric positive definite"
        @assert control_offset >= 0                    "control_offset must be non-negative"
        new(R, control_offset, control_dim)
    end
end

function evaluate_cost_term(t::QuadraticControlCost, x, u, p, ts)
    ui = player_slice(u, t.control_offset, t.control_dim)
    return 0.5 * dot(ui, t.R * ui)
end

function cost_term_gradient(t::QuadraticControlCost, x, u, p, ts)
    ui  = player_slice(u, t.control_offset, t.control_dim)
    ∇ui = t.R * ui
    ∇u  = zeros(eltype(u), length(u))
    ∇u[t.control_offset+1 : t.control_offset+t.control_dim] = ∇ui
    return (zeros(eltype(x), length(x)), ∇u)
end

function cost_term_hessian(t::QuadraticControlCost, x, u, p, ts)
    n  = length(x); m = length(u)
    Hu = zeros(Float64, m, m)
    Hu[t.control_offset+1:t.control_offset+t.control_dim,
       t.control_offset+1:t.control_offset+t.control_dim] = t.R
    return (zeros(Float64, n, n), Hu, zeros(Float64, n, m))
end

is_quadratic(::QuadraticControlCost)      = true
is_separable_term(::QuadraticControlCost) = true

# ============================================================================
# ProximityCost
# ============================================================================

"""
    ProximityCost <: AbstractCostTerm

Stage cost penalising inter-agent proximity below a safe distance.

Uses the softplus approximation to avoid non-differentiability at the
constraint boundary. Reads position components of two players' state slices.

# Mathematical Form
ℓ(x) = weight · softplus(α · (d_min - d(xᵢ, xⱼ)))² / α²

where d(xᵢ, xⱼ) = ‖pos(xᵢ) - pos(xⱼ)‖₂ + ε (ε = 1e-6 for ForwardDiff safety)
and softplus(z) = log(1 + exp(z)) / α approximates max(z, 0).

The ε regularisation prevents division by zero in the Jacobian when agents
are co-located. The softplus sharpness α trades off smoothness vs. constraint
fidelity — α ∈ [5, 50] is typical.

# Fields
- `i_offset` : zero-based state offset for player i (position is i_offset+1:i_offset+pos_dim)
- `j_offset` : zero-based state offset for player j
- `pos_dim`  : dimensionality of the position subvector (typically 2 or 3)
- `d_min`    : minimum safe separation distance
- `weight`   : penalty weight
- `α`        : softplus sharpness (default 10.0)

# Notes
`pos_dim` must be ≤ the respective player's state dimension. The cost assumes
position occupies the FIRST `pos_dim` components of each player's state slice.
"""
struct ProximityCost <: AbstractCostTerm
    i_offset::Int
    j_offset::Int
    pos_dim::Int
    d_min::Float64
    weight::Float64
    α::Float64

    function ProximityCost(
        i_offset::Int, j_offset::Int, pos_dim::Int,
        d_min::Float64, weight::Float64;
        α::Float64 = 10.0
    )
        @assert pos_dim > 0   "pos_dim must be positive"
        @assert d_min > 0     "d_min must be positive"
        @assert weight >= 0   "weight must be non-negative"
        @assert α > 0         "softplus sharpness α must be positive"
        new(i_offset, j_offset, pos_dim, d_min, weight, α)
    end
end

function evaluate_cost_term(t::ProximityCost, x, u, p, ts)
    pi_ = view(x, t.i_offset+1 : t.i_offset+t.pos_dim)
    pj  = view(x, t.j_offset+1 : t.j_offset+t.pos_dim)
    Δ   = pi_ .- pj
    d   = sqrt(dot(Δ, Δ) + 1e-12)            # ε² = 1e-12 → ε = 1e-6 distance tolerance
    z   = t.α * (t.d_min - d)
    sp  = log(1 + exp(z)) / t.α              # softplus(z)/α → smooth max(d_min-d, 0)
    return t.weight * sp^2
end

is_quadratic(::ProximityCost)      = false
is_separable_term(::ProximityCost) = false   # reads two players' slices

# ============================================================================
# CommunicationCost
# ============================================================================

"""
    CommunicationCost <: AbstractCostTerm

Stage cost penalising inter-agent separation above a maximum range.
Dual of `ProximityCost`: enforces an upper bound on distance.

Used for communication-constrained or formation-keeping objectives.

# Mathematical Form
ℓ(x) = weight · softplus(α · (d(xᵢ, xⱼ) - d_max))²  / α²

# Fields
- `i_offset` : zero-based state offset for player i
- `j_offset` : zero-based state offset for player j
- `pos_dim`  : position dimensionality
- `d_max`    : maximum allowed separation
- `weight`   : penalty weight
- `α`        : softplus sharpness (default 10.0)
"""
struct CommunicationCost <: AbstractCostTerm
    i_offset::Int
    j_offset::Int
    pos_dim::Int
    d_max::Float64
    weight::Float64
    α::Float64

    function CommunicationCost(
        i_offset::Int, j_offset::Int, pos_dim::Int,
        d_max::Float64, weight::Float64;
        α::Float64 = 10.0
    )
        @assert pos_dim > 0  "pos_dim must be positive"
        @assert d_max > 0    "d_max must be positive"
        @assert weight >= 0  "weight must be non-negative"
        @assert α > 0        "α must be positive"
        new(i_offset, j_offset, pos_dim, d_max, weight, α)
    end
end

function evaluate_cost_term(t::CommunicationCost, x, u, p, ts)
    pi_ = view(x, t.i_offset+1 : t.i_offset+t.pos_dim)
    pj  = view(x, t.j_offset+1 : t.j_offset+t.pos_dim)
    Δ   = pi_ .- pj
    d   = sqrt(dot(Δ, Δ) + 1e-12)
    z   = t.α * (d - t.d_max)
    sp  = log(1 + exp(z)) / t.α
    return t.weight * sp^2
end

is_quadratic(::CommunicationCost)      = false
is_separable_term(::CommunicationCost) = false

# ============================================================================
# ControlBarrierCost
# ============================================================================

"""
    ControlBarrierCost <: AbstractCostTerm

Exponential penalty for violating a scalar state constraint h(xᵢ) ≤ 0.

# Mathematical Form
ℓ(x) = weight · exp(α · h(xᵢ))

This is the exponential CBF relaxation: inactive when h(xᵢ) << 0,
grows steeply as h(xᵢ) → 0, and is always finite (no hard barrier).
Suitable when the constraint function h is provided analytically.

For example, a box constraint xᵢ[k] ≤ x_max gives h(xᵢ) = xᵢ[k] - x_max.

# Fields
- `h`            : scalar constraint function h(xᵢ) where xᵢ is the FULL joint state
- `state_offset` : zero-based offset; used only for `is_separable_term` dispatch,
                   not slicing (h receives the full joint state)
- `weight`       : penalty weight (default 1.0)
- `α`            : barrier sharpness (default 5.0)

# Notes
If h depends on a single player's state, set `state_offset` appropriately and
wrap the slice extraction inside h itself for clarity.
"""
struct ControlBarrierCost <: AbstractCostTerm
    h::Function
    state_offset::Int
    weight::Float64
    α::Float64

    function ControlBarrierCost(
        h::Function;
        state_offset::Int = 0,
        weight::Float64 = 1.0,
        α::Float64 = 5.0
    )
        @assert weight >= 0  "weight must be non-negative"
        @assert α > 0        "α must be positive"
        new(h, state_offset, weight, α)
    end
end

function evaluate_cost_term(t::ControlBarrierCost, x, u, p, ts)
    return t.weight * exp(t.α * t.h(x))
end

is_quadratic(::ControlBarrierCost)      = false
is_separable_term(::ControlBarrierCost) = false  # conservative default

# ============================================================================
# QuadraticTerminalCost
# ============================================================================

"""
    QuadraticTerminalCost <: AbstractTerminalCostTerm

Terminal cost: ½ (xᵢ - x_ref)ᵀ Qf (xᵢ - x_ref)

# Fields
- `Qf`           : (n_i × n_i) terminal weight matrix, symmetric PSD
- `x_ref`        : (n_i,) reference terminal state
- `state_offset` : zero-based offset into full joint state for player i
- `state_dim`    : dimension of player i's state
"""
struct QuadraticTerminalCost <: AbstractTerminalCostTerm
    Qf::Matrix{Float64}
    x_ref::Vector{Float64}
    state_offset::Int
    state_dim::Int

    function QuadraticTerminalCost(
        Qf::Matrix{Float64},
        x_ref::Vector{Float64},
        state_offset::Int,
        state_dim::Int
    )
        @assert size(Qf) == (state_dim, state_dim) "Qf must be ($state_dim × $state_dim)"
        @assert issymmetric(Qf)                    "Qf must be symmetric"
        @assert length(x_ref) == state_dim         "x_ref must have length $state_dim"
        new(Qf, x_ref, state_offset, state_dim)
    end
end

function evaluate_cost_term(t::QuadraticTerminalCost, x, p)
    xi = player_slice(x, t.state_offset, t.state_dim)
    δ  = xi .- t.x_ref
    return 0.5 * dot(δ, t.Qf * δ)
end

function cost_term_gradient(t::QuadraticTerminalCost, x, p)
    ∇x = zeros(eltype(x), length(x))
    xi  = player_slice(x, t.state_offset, t.state_dim)
    δ   = xi .- t.x_ref
    ∇x[t.state_offset+1 : t.state_offset+t.state_dim] = t.Qf * δ
    return ∇x
end

function cost_term_hessian(t::QuadraticTerminalCost, x, p)
    n  = length(x)
    Hx = zeros(Float64, n, n)
    Hx[t.state_offset+1:t.state_offset+t.state_dim,
       t.state_offset+1:t.state_offset+t.state_dim] = t.Qf
    return Hx
end

is_quadratic(::QuadraticTerminalCost) = true

# ============================================================================
# ProximityTerminalCost
# ============================================================================

"""
    ProximityTerminalCost <: AbstractTerminalCostTerm

Terminal collision avoidance cost. Same formulation as `ProximityCost`
but applied at the terminal state only.
"""
struct ProximityTerminalCost <: AbstractTerminalCostTerm
    i_offset::Int
    j_offset::Int
    pos_dim::Int
    d_min::Float64
    weight::Float64
    α::Float64

    function ProximityTerminalCost(
        i_offset::Int, j_offset::Int, pos_dim::Int,
        d_min::Float64, weight::Float64;
        α::Float64 = 10.0
    )
        @assert pos_dim > 0 && d_min > 0 && weight >= 0 && α > 0
        new(i_offset, j_offset, pos_dim, d_min, weight, α)
    end
end

function evaluate_cost_term(t::ProximityTerminalCost, x, p)
    pi_ = view(x, t.i_offset+1 : t.i_offset+t.pos_dim)
    pj  = view(x, t.j_offset+1 : t.j_offset+t.pos_dim)
    Δ   = pi_ .- pj
    d   = sqrt(dot(Δ, Δ) + 1e-12)
    z   = t.α * (t.d_min - d)
    sp  = log(1 + exp(z)) / t.α
    return t.weight * sp^2
end

is_quadratic(::ProximityTerminalCost) = false

# ============================================================================
# Ergonomic constructors
# ============================================================================

"""
    track_goal(x_ref, Q; state_offset, state_dim) -> QuadraticStateCost

Penalise deviation from a reference state.
"""
function track_goal(
    x_ref::Vector{Float64},
    Q::Matrix{Float64};
    state_offset::Int = 0,
    state_dim::Int    = length(x_ref)
)
    QuadraticStateCost(Q, x_ref, state_offset, state_dim)
end

"""
    regularize_input(R; control_offset, control_dim) -> QuadraticControlCost

Penalise control effort.
"""
function regularize_input(
    R::Matrix{Float64};
    control_offset::Int = 0,
    control_dim::Int    = size(R, 1)
)
    QuadraticControlCost(R, control_offset, control_dim)
end

"""
    avoid_proximity(; i_offset, j_offset, pos_dim, d_min, weight, α) -> ProximityCost

Penalise inter-agent proximity below `d_min`.
"""
function avoid_proximity(;
    i_offset::Int,
    j_offset::Int,
    pos_dim::Int,
    d_min::Float64,
    weight::Float64,
    α::Float64 = 10.0
)
    ProximityCost(i_offset, j_offset, pos_dim, d_min, weight; α)
end

"""
    maintain_proximity(; i_offset, j_offset, pos_dim, d_max, weight, α) -> CommunicationCost

Penalise inter-agent separation above `d_max`.
"""
function maintain_proximity(;
    i_offset::Int,
    j_offset::Int,
    pos_dim::Int,
    d_max::Float64,
    weight::Float64,
    α::Float64 = 10.0
)
    CommunicationCost(i_offset, j_offset, pos_dim, d_max, weight; α)
end

"""
    terminal_goal(x_ref, Qf; state_offset, state_dim) -> QuadraticTerminalCost

Terminal cost penalising deviation from a reference state.
"""
function terminal_goal(
    x_ref::Vector{Float64},
    Qf::Matrix{Float64};
    state_offset::Int = 0,
    state_dim::Int    = length(x_ref)
)
    QuadraticTerminalCost(Qf, x_ref, state_offset, state_dim)
end