using ForwardDiff

# ============================================================================
# AbstractPlayerDynamics hierarchy
#
# Per-player dynamics specification. Replaces the bare `dynamics::Function`
# field on PlayerSpec. Carries continuous/discrete semantics, dimension
# information, and the type parameter F for compile-time specialization.
#
# Cross-agent coupling design:
#
#   Separable dynamics (PD-GNEP):
#     Use ContinuousPlayerDynamics / DiscretePlayerDynamics with
#     signature fᵢ(xᵢ, uᵢ, p, t). Each player's state evolves independently
#     conditioned on their own (xᵢ, uᵢ). Coupling enters only through costs
#     and shared constraints. This is the FALCON / PD-GNEP setting.
#
#   Coupled dynamics (general GNEP / iLQGames):
#     Use CoupledPlayerDynamics with signature fᵢ(xᵢ, uᵢ, x_others, p, t)
#     where x_others is a NamedTuple of other players' states keyed by player
#     ID. DifferentialGame assembles these into a CoupledNonlinearDynamics at
#     the game level. This is the iLQGames / general nonlinear setting.
#
#     Alternatively, for games where dynamics share a joint state (LQ games,
#     spacecraft proximity), use LinearPlayerDynamics — the shared A matrix
#     already encodes full coupling through the joint state.
#
# ============================================================================

"""
    AbstractPlayerDynamics{T}

Per-player dynamics specification. Carries dimension information and
continuous/discrete semantics.

# Subtype Tree
```
AbstractPlayerDynamics{T}
├── ContinuousPlayerDynamics{T, F}   ẋᵢ = fᵢ(xᵢ, uᵢ, p, t)          [separable]
├── DiscretePlayerDynamics{T, F}     xᵢ(k+1) = fᵢ(xᵢ(k), uᵢ(k), p, k) [separable]
├── CoupledPlayerDynamics{T, F}      ẋᵢ = fᵢ(xᵢ, uᵢ, x_others, p, t)  [coupled]
└── LinearPlayerDynamics{T}          xᵢ(k+1) = Aᵢxᵢ + Bᵢuᵢ (LTI/LTV)  [shared state]
```
"""
abstract type AbstractPlayerDynamics{T} end

# ============================================================================
# ContinuousPlayerDynamics — separable, per-player signature
# ============================================================================

"""
    ContinuousPlayerDynamics{T, F} <: AbstractPlayerDynamics{T}

Continuous-time separable player dynamics: `ẋᵢ = fᵢ(xᵢ, uᵢ, p, t)`.

Player i's dynamics depend only on their own state `xᵢ` and control `uᵢ`.
Cross-agent coupling enters through shared constraints and costs, not dynamics.
This is the PD-GNEP / FALCON setting.

The solver handles discretization (RK4). The type parameter `F` enables
compile-time specialization of evaluate calls — zero dynamic dispatch in the
rollout hot loop.

# Fields
- `func::F`          : `fᵢ(x, u, p, t) -> ẋ`; must be ForwardDiff-compatible
- `state_dim::Int`   : nᵢ — dimension of player i's own state xᵢ
- `control_dim::Int` : mᵢ — dimension of player i's control uᵢ
- `jacobian`         : Optional `(x, u, p, t) -> (∂f/∂x, ∂f/∂u)`; if nothing,
  ForwardDiff is used. Provide analytical Jacobians for CWH dynamics.

# ForwardDiff Compatibility
Avoid type-restricting annotations inside `func`:
```julia
# Bad — breaks AD
fᵢ(x::Vector{Float64}, u, p, t) = ...
# Good — AD-compatible
fᵢ(x::AbstractVector, u, p, t) = ...
```
"""
struct ContinuousPlayerDynamics{T, F} <: AbstractPlayerDynamics{T}
    func::F
    state_dim::Int
    control_dim::Int
    jacobian::Union{Nothing, Function}

    function ContinuousPlayerDynamics(
        func::F,
        state_dim::Int,
        control_dim::Int;
        jacobian::Union{Nothing, Function} = nothing
    ) where {F}
        @assert state_dim > 0   "state_dim must be positive"
        @assert control_dim > 0 "control_dim must be positive"
        new{Float64, F}(func, state_dim, control_dim, jacobian)
    end
end

# ============================================================================
# DiscretePlayerDynamics — separable, per-player signature
# ============================================================================

"""
    DiscretePlayerDynamics{T, F} <: AbstractPlayerDynamics{T}

Discrete-time separable player dynamics: `xᵢ(k+1) = fᵢ(xᵢ(k), uᵢ(k), p, k)`.

Use when dynamics are natively discrete (CWH in discrete time, Markov models).
For continuous dynamics to be discretized by the solver, prefer
`ContinuousPlayerDynamics` — this keeps the integration scheme inside the
solver where it belongs.

# Fields
- `func::F`          : `fᵢ(x, u, p, k) -> x_next`; ForwardDiff-compatible
- `state_dim::Int`   : nᵢ
- `control_dim::Int` : mᵢ
- `jacobian`         : Optional analytical Jacobian
"""
struct DiscretePlayerDynamics{T, F} <: AbstractPlayerDynamics{T}
    func::F
    state_dim::Int
    control_dim::Int
    jacobian::Union{Nothing, Function}

    function DiscretePlayerDynamics(
        func::F,
        state_dim::Int,
        control_dim::Int;
        jacobian::Union{Nothing, Function} = nothing
    ) where {F}
        @assert state_dim > 0   "state_dim must be positive"
        @assert control_dim > 0 "control_dim must be positive"
        new{Float64, F}(func, state_dim, control_dim, jacobian)
    end
end

# ============================================================================
# CoupledPlayerDynamics — cross-agent coupling in dynamics
# ============================================================================

"""
    CoupledPlayerDynamics{T, F} <: AbstractPlayerDynamics{T}

Continuous-time dynamics where player i's state evolution depends on
other players' states: `ẋᵢ = fᵢ(xᵢ, uᵢ, x_others, p, t)`.

`x_others` is provided as a plain `AbstractVector` containing the concatenated
states of all players *other than i*, in player-index order (skipping player i).
The `DifferentialGame` constructor assembles `CoupledPlayerDynamics` instances
into a `CoupledNonlinearDynamics` at the game level, handling the slicing.

# Use Cases
- Spacecraft proximity: player i's relative dynamics depend on j's position
- Vehicle intersection: collision avoidance terms couple dynamics
- Formation flying: leader–follower state coupling
- Any game where ẋᵢ cannot be written without xⱼ for j ≠ i

# Fields
- `func::F`          : `fᵢ(xᵢ, uᵢ, x_others, p, t) -> ẋᵢ`; ForwardDiff-compatible
- `state_dim::Int`   : nᵢ — dimension of player i's own state
- `control_dim::Int` : mᵢ
- `jacobian`         : Optional `(xᵢ, uᵢ, x_others, p, t) -> (∂f/∂xᵢ, ∂f/∂uᵢ, ∂f/∂x_others)`;
  providing the x_others Jacobian enables efficient coupling Jacobian assembly.

# ForwardDiff Compatibility
Both `xᵢ` and `x_others` will receive dual numbers during linearization:
```julia
# Good
fᵢ(xi::AbstractVector, ui::AbstractVector, xo::AbstractVector, p, t) = ...
```

# Example — two-player proximity dynamics
```julia
# Player 1 dynamics depend on player 2's position
f1(x1, u1, x_others, p, t) = [
    x1[2] + 0.1*(x_others[1] - x1[1]),   # position attracted toward player 2
    u1[1]
]
dyn1 = CoupledPlayerDynamics(f1, 2, 1)
```
"""
struct CoupledPlayerDynamics{T, F} <: AbstractPlayerDynamics{T}
    func::F
    state_dim::Int
    control_dim::Int
    jacobian::Union{Nothing, Function}

    function CoupledPlayerDynamics(
        func::F,
        state_dim::Int,
        control_dim::Int;
        jacobian::Union{Nothing, Function} = nothing
    ) where {F}
        @assert state_dim > 0   "state_dim must be positive"
        @assert control_dim > 0 "control_dim must be positive"
        new{Float64, F}(func, state_dim, control_dim, jacobian)
    end
end

# ============================================================================
# LinearPlayerDynamics — shared joint state, LTI or LTV
# ============================================================================

"""
    LinearPlayerDynamics{T} <: AbstractPlayerDynamics{T}

Linear discrete-time dynamics (LTI or LTV) for a player participating in a
*shared joint state* game: `x(k+1) = A(k)x(k) + B₁(k)u₁(k) + ... + Bₙ(k)uₙ(k)`.

Player i stores both the shared system matrix `A` and their own control matrix `B`.
Cross-agent coupling is fully encoded in `A` — the joint state `x` evolves under
the sum of all players' control contributions.

Both `A` and `B` are stored per player. For shared-state LQ games all players
must have identical `A` matrices — validated at `DifferentialGame` construction,
not here, because per-player self-containment is intentional.

LTV is indicated by passing `Vector{Matrix{T}}` for `A` and/or `B`.

# Fields
- `A` : System matrix (LTI: `Matrix{T}`) or sequence (LTV: `Vector{Matrix{T}}`)
- `B` : Player i's control matrix (LTI) or sequence (LTV)
- `state_dim::Int`   : n — inferred from A
- `control_dim::Int` : mᵢ — inferred from B

# Example — 2-player spacecraft rendezvous
```julia
# CWH A matrix (shared joint state)
A_cwh = cwh_A_matrix(n_orbit, dt)
B1    = cwh_B_matrix(dt)[:, 1:2]   # chaser controls
B2    = cwh_B_matrix(dt)[:, 3:4]   # target controls (or zeros if passive)
dyn1  = LinearPlayerDynamics(A_cwh, B1)
dyn2  = LinearPlayerDynamics(A_cwh, B2)
```
"""
struct LinearPlayerDynamics{T} <: AbstractPlayerDynamics{T}
    A::Union{Matrix{T}, Vector{Matrix{T}}}
    B::Union{Matrix{T}, Vector{Matrix{T}}}
    state_dim::Int
    control_dim::Int

    function LinearPlayerDynamics(
        A::Union{Matrix{T}, Vector{Matrix{T}}},
        B::Union{Matrix{T}, Vector{Matrix{T}}}
    ) where {T}
        A1 = A isa Matrix ? A : A[1]
        B1 = B isa Matrix ? B : B[1]
        n  = size(A1, 1)
        m  = size(B1, 2)

        if A isa Matrix
            @assert size(A) == (n, n) "A must be square (n×n), got $(size(A))"
        else
            @assert all(size(Ak) == (n, n) for Ak in A) "All A matrices must be ($n×$n)"
        end

        if B isa Matrix
            @assert size(B, 1) == n "B must have $n rows, got $(size(B,1))"
        else
            @assert all(size(Bk, 1) == n for Bk in B) "All B matrices must have $n rows"
            @assert all(size(Bk, 2) == m for Bk in B) "All B matrices must have $m columns"
        end

        if A isa Vector && B isa Vector
            @assert length(A) == length(B) "LTV A and B sequences must have equal length"
        end

        new{T}(A, B, n, m)
    end
end

# ============================================================================
# Structural trait queries
# ============================================================================

"""
    is_continuous(dyn::AbstractPlayerDynamics) -> Bool
"""
is_continuous(::ContinuousPlayerDynamics) = true
is_continuous(::CoupledPlayerDynamics)    = true   # also continuous-time
is_continuous(::AbstractPlayerDynamics)   = false

"""
    is_discrete(dyn::AbstractPlayerDynamics) -> Bool
"""
is_discrete(::ContinuousPlayerDynamics) = false
is_discrete(::CoupledPlayerDynamics)    = false
is_discrete(::AbstractPlayerDynamics)   = true

"""
    is_linear(dyn::AbstractPlayerDynamics) -> Bool
"""
is_linear(::LinearPlayerDynamics)   = true
is_linear(::AbstractPlayerDynamics) = false

"""
    is_separable_dynamics(dyn::AbstractPlayerDynamics) -> Bool

True if dynamics depend only on player i's own (xᵢ, uᵢ). False for
`CoupledPlayerDynamics` where dynamics depend on x_others.
"""
is_separable_dynamics(::ContinuousPlayerDynamics) = true
is_separable_dynamics(::DiscretePlayerDynamics)   = true
is_separable_dynamics(::CoupledPlayerDynamics)    = false
is_separable_dynamics(::LinearPlayerDynamics)     = false  # shared state, not separable

"""
    is_ltv(dyn::LinearPlayerDynamics) -> Bool
"""
is_ltv(dyn::LinearPlayerDynamics) = dyn.A isa Vector

"""
    get_A(dyn::LinearPlayerDynamics{T}, k::Int) -> Matrix{T}
"""
get_A(dyn::LinearPlayerDynamics{T}, k::Int) where {T} =
    dyn.A isa Matrix ? dyn.A : dyn.A[k]

"""
    get_B(dyn::LinearPlayerDynamics{T}, k::Int) -> Matrix{T}
"""
get_B(dyn::LinearPlayerDynamics{T}, k::Int) where {T} =
    dyn.B isa Matrix ? dyn.B : dyn.B[k]

# ============================================================================
# evaluate_player_dynamics — single-player evaluation interface
# ============================================================================

"""
    evaluate_player_dynamics(dyn, xᵢ, uᵢ, p, t) -> ẋᵢ or xᵢ(k+1)
    evaluate_player_dynamics(dyn::CoupledPlayerDynamics, xᵢ, uᵢ, x_others, p, t)

Evaluate player dynamics at player-local state `xᵢ` and control `uᵢ`.

For `CoupledPlayerDynamics`, `x_others` (concatenated states of all other
players in index order) must be provided as the third positional argument.
"""
function evaluate_player_dynamics end

function evaluate_player_dynamics(
    dyn::ContinuousPlayerDynamics{T, F},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Real
) where {T, F}
    return dyn.func(x, u, p, t)
end

function evaluate_player_dynamics(
    dyn::DiscretePlayerDynamics{T, F},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t        # Int or Real
) where {T, F}
    return dyn.func(x, u, p, t)
end

function evaluate_player_dynamics(
    dyn::CoupledPlayerDynamics{T, F},
    x::AbstractVector,
    u::AbstractVector,
    x_others::AbstractVector,
    p,
    t::Real
) where {T, F}
    return dyn.func(x, u, x_others, p, t)
end

function evaluate_player_dynamics(
    dyn::LinearPlayerDynamics{T},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Int
) where {T}
    return get_A(dyn, t) * x + get_B(dyn, t) * u
end

# ============================================================================
# player_dynamics_jacobian — (∂f/∂xᵢ, ∂f/∂uᵢ) and optionally ∂f/∂x_others
# ============================================================================

"""
    player_dynamics_jacobian(dyn, xᵢ, uᵢ, p, t) -> (Jx, Ju)
    player_dynamics_jacobian(dyn::CoupledPlayerDynamics, xᵢ, uᵢ, x_others, p, t)
        -> (Jx, Ju, Jx_others)

Compute dynamics Jacobians. Falls back to ForwardDiff if no analytical
Jacobian is registered.

For `CoupledPlayerDynamics`, returns the additional coupling Jacobian
`Jx_others = ∂fᵢ/∂x_others` (shape nᵢ × n_others), which is required by
iLQGames linearization to populate off-diagonal blocks of the joint
dynamics Jacobian.
"""
function player_dynamics_jacobian end

function player_dynamics_jacobian(
    dyn::LinearPlayerDynamics{T},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Int
) where {T}
    return (get_A(dyn, t), get_B(dyn, t))
end

function player_dynamics_jacobian(
    dyn::Union{ContinuousPlayerDynamics{T,F}, DiscretePlayerDynamics{T,F}},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t
) where {T, F}
    if dyn.jacobian !== nothing
        return dyn.jacobian(x, u, p, t)
    end
    return _ad_player_jacobian(dyn.func, x, u, p, t)
end

function player_dynamics_jacobian(
    dyn::CoupledPlayerDynamics{T, F},
    x::AbstractVector,
    u::AbstractVector,
    x_others::AbstractVector,
    p,
    t::Real
) where {T, F}
    if dyn.jacobian !== nothing
        return dyn.jacobian(x, u, x_others, p, t)
    end
    return _ad_coupled_jacobian(dyn.func, x, u, x_others, p, t)
end

# ============================================================================
# Internal AD helpers
# ============================================================================

function _ad_player_jacobian(func::F, x, u, p, t) where {F}
    nx = length(x)
    z  = vcat(x, u)
    J  = ForwardDiff.jacobian(
        z_var -> func(z_var[1:nx], z_var[nx+1:end], p, t),
        z
    )
    return (J[:, 1:nx], J[:, nx+1:end])
end

function _ad_coupled_jacobian(func::F, x, u, x_others, p, t) where {F}
    nx = length(x)
    nu = length(u)
    no = length(x_others)
    z  = vcat(x, u, x_others)
    J  = ForwardDiff.jacobian(
        z_var -> func(
            z_var[1:nx],
            z_var[nx+1:nx+nu],
            z_var[nx+nu+1:end],
            p, t
        ),
        z
    )
    Jx       = J[:, 1:nx]
    Ju       = J[:, nx+1:nx+nu]
    Jx_others = J[:, nx+nu+1:end]
    return (Jx, Ju, Jx_others)
end