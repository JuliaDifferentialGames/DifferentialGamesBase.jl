# ============================================================================
# constraints/private_constraints.jl
#
# Concrete private constraint types. Each involves a single player and
# subtypes AbstractPrivateInequality or AbstractPrivateEquality directly —
# no wrapper struct.
#
# All types carry:
#   player::Int                   — which player this applies to
#   state_offset, state_dim       — for types that read state
#   control_offset, control_dim   — for types that read control
#
# Jacobians for ControlBounds and StateBounds are exact (analytical).
# PrivateNonlinear falls back to ForwardDiff unless an analytical jacobian
# is provided.
# ============================================================================

# ============================================================================
# ControlBounds — box constraint on player i's control
# ============================================================================

"""
    ControlBounds{T} <: AbstractPrivateInequality

Box constraint on player i's control: `u_min ≤ uᵢ ≤ u_max`.

Encoded as two stacked inequalities: `[u_min - uᵢ; uᵢ - u_max] ≤ 0`.

# Fields
- `player`         : Player index (1-based)
- `control_offset` : Zero-based offset of player i in joint control vector
- `control_dim`    : Dimension of player i's control mᵢ
- `lower`          : Lower bound vector (mᵢ,); use `-Inf` for unbounded
- `upper`          : Upper bound vector (mᵢ,); use `+Inf` for unbounded

# Jacobian
Exact analytical: `Jx = 0`, `Ju = [-I; I]` at player i's control slice.

# Example
```julia
# Player 1 thrust limits: |uᵢ| ≤ 5 N
c = ControlBounds(1, control_offset=0, control_dim=2,
                  lower=fill(-5.0, 2), upper=fill(5.0, 2))
```
"""
struct ControlBounds{T} <: AbstractPrivateInequality
    player::Int
    control_offset::Int
    control_dim::Int
    lower::Vector{T}
    upper::Vector{T}

    function ControlBounds(
        player::Int,
        control_offset::Int,
        control_dim::Int,
        lower::Vector{T},
        upper::Vector{T}
    ) where {T}
        @assert player > 0        "player must be positive"
        @assert control_offset >= 0
        @assert control_dim > 0
        @assert length(lower) == control_dim "lower must have length $control_dim"
        @assert length(upper) == control_dim "upper must have length $control_dim"
        @assert all(lower .<= upper)         "lower must be ≤ upper elementwise"
        new{T}(player, control_offset, control_dim, lower, upper)
    end
end

function ControlBounds(
    player::Int;
    control_offset::Int,
    control_dim::Int,
    lower::Vector{T},
    upper::Vector{T}
) where {T}
    ControlBounds(player, control_offset, control_dim, lower, upper)
end

function evaluate_constraint(c::ControlBounds, x, u, p, t)
    ui = player_slice(u, c.control_offset, c.control_dim)
    return vcat(c.lower .- ui, ui .- c.upper)
end

function constraint_jacobian(c::ControlBounds{T}, x, u, p, t) where {T}
    nx = length(x); nu = length(u); mi = c.control_dim
    # [lower - ui; ui - upper] → Ju[:, control_offset+1:control_offset+mi] = [-I; I]
    Jx = zeros(T, 2mi, nx)
    Ju = zeros(T, 2mi, nu)
    cr = c.control_offset+1 : c.control_offset+mi
    Ju[1:mi,    cr] = -Matrix{T}(I, mi, mi)
    Ju[mi+1:end, cr] =  Matrix{T}(I, mi, mi)
    return (Jx, Ju)
end

# ============================================================================
# StateBounds — box constraint on player i's state
# ============================================================================

"""
    StateBounds{T} <: AbstractPrivateInequality

Box constraint on player i's state: `x_min ≤ xᵢ ≤ x_max`.

Encoded as `[x_min - xᵢ; xᵢ - x_max] ≤ 0`.

# Fields
- `player`       : Player index
- `state_offset` : Zero-based offset of player i in joint state vector
- `state_dim`    : Dimension of player i's state nᵢ
- `lower`        : Lower bound (nᵢ,); use `-Inf` for unbounded
- `upper`        : Upper bound (nᵢ,); use `+Inf` for unbounded

# Example
```julia
# Player 2 position bounds: -10 ≤ x ≤ 10
c = StateBounds(2, state_offset=4, state_dim=4,
                lower=fill(-10.0, 4), upper=fill(10.0, 4))
```
"""
struct StateBounds{T} <: AbstractPrivateInequality
    player::Int
    state_offset::Int
    state_dim::Int
    lower::Vector{T}
    upper::Vector{T}

    function StateBounds(
        player::Int,
        state_offset::Int,
        state_dim::Int,
        lower::Vector{T},
        upper::Vector{T}
    ) where {T}
        @assert player > 0       "player must be positive"
        @assert state_offset >= 0
        @assert state_dim > 0
        @assert length(lower) == state_dim
        @assert length(upper) == state_dim
        @assert all(lower .<= upper)
        new{T}(player, state_offset, state_dim, lower, upper)
    end
end

function StateBounds(
    player::Int;
    state_offset::Int,
    state_dim::Int,
    lower::Vector{T},
    upper::Vector{T}
) where {T}
    StateBounds(player, state_offset, state_dim, lower, upper)
end

function evaluate_constraint(c::StateBounds, x, u, p, t)
    xi = player_slice(x, c.state_offset, c.state_dim)
    return vcat(c.lower .- xi, xi .- c.upper)
end

function constraint_jacobian(c::StateBounds{T}, x, u, p, t) where {T}
    nx = length(x); nu = length(u); ni = c.state_dim
    Jx = zeros(T, 2ni, nx)
    Ju = zeros(T, 2ni, nu)
    sr = c.state_offset+1 : c.state_offset+ni
    Jx[1:ni,    sr] = -Matrix{T}(I, ni, ni)
    Jx[ni+1:end, sr] =  Matrix{T}(I, ni, ni)
    return (Jx, Ju)
end

# ============================================================================
# PrivateNonlinear — general scalar/vector private constraint
# ============================================================================

"""
    PrivateNonlinear{F, J, CT} <: AbstractPrivateConstraint

General nonlinear private constraint for a single player.

The function `func` receives the full joint `(x, u, p, t)` — same convention
as cost terms. Use `player_slice` inside `func` to extract player i's slice.
This avoids a separate slicing layer and keeps ForwardDiff compatibility clean.

# Type Parameters
- `CT` : `:inequality` or `:equality` — determines which abstract supertype
         the concrete struct subtypes at definition time. See constructors below.

# Fields
- `player`    : Player index
- `func`      : `(x, u, p, t) → AbstractVector` — ForwardDiff-compatible
- `jacobian`  : `Nothing` or `(x, u, p, t) → (Jx, Ju)` — analytical override
- `dim`       : Output dimension of `func`

# Example
```julia
# Player 1 must stay inside a circle of radius 2 around origin
c = PrivateInequality(1,
    func = (x, u, p, t) -> [dot(x[1:2], x[1:2]) - 4.0],
    dim  = 1
)
```
"""
struct PrivateNonlinearInequality{F, J} <: AbstractPrivateInequality
    player::Int
    func::F
    jacobian::J       # Nothing or Function (x,u,p,t) → (Jx, Ju)
    dim::Int

    function PrivateNonlinearInequality(
        player::Int, func::F, jacobian::J, dim::Int
    ) where {F, J}
        @assert player > 0 "player must be positive"
        @assert dim > 0    "dim must be positive"
        new{F, J}(player, func, jacobian, dim)
    end
end

struct PrivateNonlinearEquality{F, J} <: AbstractPrivateEquality
    player::Int
    func::F
    jacobian::J
    dim::Int

    function PrivateNonlinearEquality(
        player::Int, func::F, jacobian::J, dim::Int
    ) where {F, J}
        @assert player > 0 "player must be positive"
        @assert dim > 0    "dim must be positive"
        new{F, J}(player, func, jacobian, dim)
    end
end

# Evaluation — same for both
function evaluate_constraint(c::Union{PrivateNonlinearInequality, PrivateNonlinearEquality},
                              x, u, p, t)
    return c.func(x, u, p, t)
end

# Jacobian — analytical if provided, ForwardDiff otherwise
function constraint_jacobian(c::Union{PrivateNonlinearInequality, PrivateNonlinearEquality},
                              x, u, p, t)
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
    PrivateInequality(player; func, dim, jacobian=nothing) -> PrivateNonlinearInequality

Private inequality constraint: `func(x, u, p, t) ≤ 0`.
"""
function PrivateInequality(
    player::Int;
    func::Function,
    dim::Int,
    jacobian::Union{Nothing, Function} = nothing
)
    PrivateNonlinearInequality(player, func, jacobian, dim)
end

"""
    PrivateEquality(player; func, dim, jacobian=nothing) -> PrivateNonlinearEquality

Private equality constraint: `func(x, u, p, t) = 0`.
"""
function PrivateEquality(
    player::Int;
    func::Function,
    dim::Int,
    jacobian::Union{Nothing, Function} = nothing
)
    PrivateNonlinearEquality(player, func, jacobian, dim)
end