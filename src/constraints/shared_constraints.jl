# ============================================================================
# constraints/shared_constraints.jl
#
# Concrete shared constraint types coupling ≥2 players.
# Each carries `players::Vector{Int}` for coupling graph construction.
# Evaluated once per timestep — not once per player.
# ============================================================================

# ============================================================================
# ProximityConstraint — minimum separation between two agents
# ============================================================================

"""
    ProximityConstraint{T} <: AbstractSharedInequality

Hard separation constraint between two agents: `d_min - ‖pᵢ - pⱼ‖₂ ≤ 0`.

Uses softplus regularisation on the distance to avoid the non-smooth gradient
at co-location: `d(xᵢ, xⱼ) = sqrt(‖pᵢ - pⱼ‖² + ε²)`.

This makes the Jacobian well-defined everywhere. The regularisation parameter
`ε` (default 1e-6) is small enough to be negligible at any physically
meaningful separation.

# Fields
- `players`        : `[i, j]` — the two players involved, sorted ascending
- `i_offset`       : Zero-based state offset for player i
- `j_offset`       : Zero-based state offset for player j
- `pos_dim`        : Dimensionality of the position sub-vector (typically 2 or 3).
                     Position is assumed to occupy the FIRST `pos_dim` components
                     of each player's state slice.
- `d_min`          : Minimum safe separation distance
- `ε`              : Distance regularisation for ForwardDiff safety (default 1e-6)

# Constraint Value
Returns `[d_min - d(xᵢ, xⱼ)]` — negative means satisfied.

# Example
```julia
# Two quadrotors, 2D position in first 2 components of a 4D state
c = ProximityConstraint([1, 2],
    i_offset=0, j_offset=4, pos_dim=2, d_min=0.5)
```
"""
struct ProximityConstraint{T} <: AbstractSharedInequality
    players::Vector{Int}
    i_offset::Int
    j_offset::Int
    pos_dim::Int
    d_min::T
    ε::T

    function ProximityConstraint(
        players::Vector{Int},
        i_offset::Int,
        j_offset::Int,
        pos_dim::Int,
        d_min::T;
        ε::T = T(1e-6)
    ) where {T}
        @assert length(players) == 2 "ProximityConstraint requires exactly 2 players"
        @assert allunique(players)   "Player indices must be distinct"
        @assert pos_dim > 0
        @assert d_min > 0
        @assert ε > 0
        new{T}(sort(players), i_offset, j_offset, pos_dim, d_min, ε)
    end
end

function ProximityConstraint(
    players::Vector{Int};
    i_offset::Int,
    j_offset::Int,
    pos_dim::Int,
    d_min::T,
    ε::Real = 1e-6
) where {T}
    ProximityConstraint(players, i_offset, j_offset, pos_dim, d_min; ε = T(ε))
end

function evaluate_constraint(c::ProximityConstraint, x, u, p, t)
    pi_ = view(x, c.i_offset+1 : c.i_offset+c.pos_dim)
    pj  = view(x, c.j_offset+1 : c.j_offset+c.pos_dim)
    Δ   = pi_ .- pj
    d   = sqrt(dot(Δ, Δ) + c.ε^2)
    return [c.d_min - d]
end

# Analytical Jacobian — avoids dual-number instability near co-location
function constraint_jacobian(c::ProximityConstraint{T}, x, u, p, t) where {T}
    nx = length(x); nu = length(u)
    pi_ = view(x, c.i_offset+1 : c.i_offset+c.pos_dim)
    pj  = view(x, c.j_offset+1 : c.j_offset+c.pos_dim)
    Δ   = pi_ .- pj
    d   = sqrt(dot(Δ, Δ) + c.ε^2)

    # ∂(d_min - d)/∂x: -∂d/∂pᵢ = -Δ/d, -∂d/∂pⱼ = +Δ/d
    Jx = zeros(T, 1, nx)
    Jx[1, c.i_offset+1 : c.i_offset+c.pos_dim] = -Δ ./ d
    Jx[1, c.j_offset+1 : c.j_offset+c.pos_dim] =  Δ ./ d

    return (Jx, zeros(T, 1, nu))
end

# ============================================================================
# CommunicationConstraint — maximum separation (keep-in-range)
# ============================================================================

"""
    CommunicationConstraint{T} <: AbstractSharedInequality

Maximum separation constraint: `‖pᵢ - pⱼ‖₂ - d_max ≤ 0`.

Enforces that two agents stay within communication/formation range.

# Fields
- `players`  : `[i, j]` — sorted ascending
- `i_offset` : Zero-based state offset for player i
- `j_offset` : Zero-based state offset for player j
- `pos_dim`  : Position dimensionality
- `d_max`    : Maximum allowed separation
- `ε`        : Distance regularisation (default 1e-6)
"""
struct CommunicationConstraint{T} <: AbstractSharedInequality
    players::Vector{Int}
    i_offset::Int
    j_offset::Int
    pos_dim::Int
    d_max::T
    ε::T

    function CommunicationConstraint(
        players::Vector{Int},
        i_offset::Int,
        j_offset::Int,
        pos_dim::Int,
        d_max::T;
        ε::T = T(1e-6)
    ) where {T}
        @assert length(players) == 2 && allunique(players)
        @assert pos_dim > 0 && d_max > 0 && ε > 0
        new{T}(sort(players), i_offset, j_offset, pos_dim, d_max, ε)
    end
end

function CommunicationConstraint(
    players::Vector{Int};
    i_offset::Int,
    j_offset::Int,
    pos_dim::Int,
    d_max::T,
    ε::Real = 1e-6
) where {T}
    CommunicationConstraint(players, i_offset, j_offset, pos_dim, d_max; ε = T(ε))
end

function evaluate_constraint(c::CommunicationConstraint, x, u, p, t)
    pi_ = view(x, c.i_offset+1 : c.i_offset+c.pos_dim)
    pj  = view(x, c.j_offset+1 : c.j_offset+c.pos_dim)
    Δ   = pi_ .- pj
    d   = sqrt(dot(Δ, Δ) + c.ε^2)
    return [d - c.d_max]
end

function constraint_jacobian(c::CommunicationConstraint{T}, x, u, p, t) where {T}
    nx = length(x); nu = length(u)
    pi_ = view(x, c.i_offset+1 : c.i_offset+c.pos_dim)
    pj  = view(x, c.j_offset+1 : c.j_offset+c.pos_dim)
    Δ   = pi_ .- pj
    d   = sqrt(dot(Δ, Δ) + c.ε^2)

    Jx = zeros(T, 1, nx)
    Jx[1, c.i_offset+1 : c.i_offset+c.pos_dim] =  Δ ./ d
    Jx[1, c.j_offset+1 : c.j_offset+c.pos_dim] = -Δ ./ d

    return (Jx, zeros(T, 1, nu))
end

# ============================================================================
# LinearCoupling — linear inequality coupling players
# ============================================================================

"""
    LinearCoupling{T} <: AbstractSharedInequality

Linear shared inequality: `A * [x; u] ≤ b`.

For constraints that couple players linearly (e.g., total fuel budget,
shared resource limits, linear formation constraints).

# Fields
- `players` : Players involved (used for coupling graph only)
- `A`       : Constraint matrix (m × (nx + nu_total))
- `b`       : Constraint vector (m,)

# Example
```julia
# Two players must not exceed a shared control budget: u1 + u2 ≤ U_max
A = reshape([1.0, 1.0], 1, 2)
c = LinearCoupling([1, 2], A, [U_max])
```
"""
struct LinearCoupling{T} <: AbstractSharedInequality
    players::Vector{Int}
    A::Matrix{T}
    b::Vector{T}

    function LinearCoupling(players::Vector{Int}, A::Matrix{T}, b::Vector{T}) where {T}
        @assert !isempty(players) && allunique(players)
        @assert all(p > 0 for p in players)
        @assert size(A, 1) == length(b)
        new{T}(sort(players), A, b)
    end
end

function evaluate_constraint(c::LinearCoupling, x, u, p, t)
    return c.A * vcat(x, u) .- c.b
end

function constraint_jacobian(c::LinearCoupling{T}, x, u, p, t) where {T}
    nx = length(x); nu = length(u)
    return (c.A[:, 1:nx], c.A[:, nx+1:end])
end

# ============================================================================
# SharedNonlinear — general nonlinear shared constraint
# ============================================================================

"""
    SharedNonlinearInequality{F, J} <: AbstractSharedInequality

General nonlinear shared inequality: `func(x, u, p, t) ≤ 0`.

For coupling constraints that don't fit the structured types above.
`func` receives full joint `(x, u, p, t)` and returns a vector.
Use `player_slice` inside `func` for ergonomic slice extraction.

# Fields
- `players`   : All players involved (used for coupling graph)
- `func`      : `(x, u, p, t) → AbstractVector` — ForwardDiff-compatible
- `jacobian`  : `Nothing` or `(x, u, p, t) → (Jx, Ju)`
- `dim`       : Output dimension

# Example
```julia
# Nonlinear formation constraint on players 1 and 3
c = SharedInequality([1, 3],
    func = (x, u, p, t) -> [norm(x[1:2] - x[9:10])^2 - 25.0],
    dim  = 1
)
```
"""
struct SharedNonlinearInequality{F, J} <: AbstractSharedInequality
    players::Vector{Int}
    func::F
    jacobian::J
    dim::Int

    function SharedNonlinearInequality(
        players::Vector{Int}, func::F, jacobian::J, dim::Int
    ) where {F, J}
        @assert !isempty(players) && allunique(players) && all(p > 0 for p in players)
        @assert dim > 0
        new{F, J}(sort(players), func, jacobian, dim)
    end
end

struct SharedNonlinearEquality{F, J} <: AbstractSharedEquality
    players::Vector{Int}
    func::F
    jacobian::J
    dim::Int

    function SharedNonlinearEquality(
        players::Vector{Int}, func::F, jacobian::J, dim::Int
    ) where {F, J}
        @assert !isempty(players) && allunique(players) && all(p > 0 for p in players)
        @assert dim > 0
        new{F, J}(sort(players), func, jacobian, dim)
    end
end

function evaluate_constraint(
    c::Union{SharedNonlinearInequality, SharedNonlinearEquality}, x, u, p, t
)
    return c.func(x, u, p, t)
end

function constraint_jacobian(
    c::Union{SharedNonlinearInequality, SharedNonlinearEquality}, x, u, p, t
)
    c.jacobian !== nothing && return c.jacobian(x, u, p, t)
    nx = length(x); nu = length(u)
    z  = vcat(x, u)
    J  = ForwardDiff.jacobian(
        z_var -> c.func(z_var[1:nx], z_var[nx+1:end], p, t), z
    )
    return (J[:, 1:nx], J[:, nx+1:end])
end

# ============================================================================
# Ergonomic constructors
# ============================================================================

"""
    SharedInequality(players; func, dim, jacobian=nothing) -> SharedNonlinearInequality
"""
function SharedInequality(
    players::Vector{Int};
    func::Function,
    dim::Int,
    jacobian::Union{Nothing, Function} = nothing
)
    SharedNonlinearInequality(players, func, jacobian, dim)
end

"""
    SharedEquality(players; func, dim, jacobian=nothing) -> SharedNonlinearEquality
"""
function SharedEquality(
    players::Vector{Int};
    func::Function,
    dim::Int,
    jacobian::Union{Nothing, Function} = nothing
)
    SharedNonlinearEquality(players, func, jacobian, dim)
end