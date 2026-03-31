# ============================================================================
# Dynamics Specifications
# ============================================================================

"""
    DynamicsSpec{T}

Abstract base type for dynamics specifications.
"""
abstract type DynamicsSpec{T} end

# ============================================================================
# Linear Dynamics (unified LTI/LTV via type parameters)
# ============================================================================

"""
    LinearDynamics{T, AM, BM} <: DynamicsSpec{T}

Unified linear dynamics supporting both LTI and LTV formulations.

# Type Parameters
- `T`  : Numeric type (Float64, etc.)
- `AM` : `Matrix{T}` for LTI, `Vector{Matrix{T}}` for LTV
- `BM` : `Vector{Matrix{T}}` for LTI, `Vector{Vector{Matrix{T}}}` for LTV

# Mathematical Form
LTI: x(k+1) = A x(k) + Σᵢ Bᵢ uᵢ(k)
LTV: x(k+1) = A(k) x(k) + Σᵢ Bᵢ(k) uᵢ(k)

# Fields
- `A`            : System matrix or sequence of system matrices
- `B`            : Per-player control matrices or sequences thereof
- `state_dim`    : Shared state dimension n
- `control_dims` : Control dimensions [m₁, ..., mₙₚ]
- `n_steps`      : Number of timesteps N (nothing for LTI, Int for LTV)

# Construction
Use the named constructors rather than the raw parametric form:
```julia
# LTI
dyn = LinearDynamics(A, B)

# LTV
dyn = LinearDynamics(A_seq, B_seq)   # length-N vectors
```

# Accessors
Always use `get_A(dyn, k)` and `get_B(dyn, i, k)` — never access `.A` or `.B`
directly in solver code. This ensures LTI and LTV paths are interchangeable.

# References
Başar, T., & Olsder, G. J. (1999). Dynamic Noncooperative Game Theory (2nd ed.).
SIAM. Section 6.3.
"""
struct LinearDynamics{T, AM, BM} <: DynamicsSpec{T}
    A::AM
    B::BM
    state_dim::Int
    control_dims::Vector{Int}
    # Nothing for LTI (N is determined by the time horizon at solve time).
    # Int for LTV (must be consistent with time horizon).
    n_steps::Union{Nothing, Int}

    # Internal constructor — use named constructors below.
    function LinearDynamics{T, AM, BM}(
        A::AM,
        B::BM,
        state_dim::Int,
        control_dims::Vector{Int},
        n_steps::Union{Nothing, Int}
    ) where {T, AM, BM}
        new{T, AM, BM}(A, B, state_dim, control_dims, n_steps)
    end
end

# ─── LTI constructor ──────────────────────────────────────────────────────────

"""
    LinearDynamics(A::Matrix{T}, B::Vector{Matrix{T}}) -> LinearDynamics{T, Matrix{T}, ...}

Construct an LTI linear dynamics object.

All matrices must be consistent: `A` is (n × n), each `B[i]` is (n × mᵢ).
"""
function LinearDynamics(A::Matrix{T}, B::Vector{Matrix{T}}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n "A must be square (n × n), got $(size(A))"
    n_players = length(B)
    @assert n_players > 0 "B must contain at least one control matrix"
    control_dims = Vector{Int}(undef, n_players)
    for i in 1:n_players
        @assert size(B[i], 1) == n "B[$i] must have $n rows, got $(size(B[i], 1))"
        control_dims[i] = size(B[i], 2)
    end
    AM = Matrix{T}
    BM = Vector{Matrix{T}}
    LinearDynamics{T, AM, BM}(A, B, n, control_dims, nothing)
end

# ─── LTV constructor ──────────────────────────────────────────────────────────

"""
    LinearDynamics(A_seq::Vector{Matrix{T}}, B_seq::Vector{Vector{Matrix{T}}}) -> LinearDynamics{T, Vector{Matrix{T}}, ...}

Construct an LTV linear dynamics object.

# Arguments
- `A_seq` : Length-N vector of (n × n) matrices; `A_seq[k]` is the dynamics matrix at step k
- `B_seq` : Length-n_players vector of length-N vectors; `B_seq[i][k]` is player i's control
             matrix at step k, of size (n × mᵢ)

# Validation
All sequences must have the same length N. Control dimensions must be consistent
across timesteps for each player (mᵢ is fixed, only the matrix values vary).
"""
function LinearDynamics(
    A_seq::Vector{Matrix{T}},
    B_seq::Vector{Vector{Matrix{T}}}
) where {T}
    N = length(A_seq)
    @assert N > 0 "A_seq must be non-empty"
    n = size(A_seq[1], 1)

    for k in 1:N
        @assert size(A_seq[k]) == (n, n) "A_seq[$k] must be ($n × $n), got $(size(A_seq[k]))"
    end

    n_players = length(B_seq)
    @assert n_players > 0 "B_seq must contain at least one player"
    control_dims = Vector{Int}(undef, n_players)

    for i in 1:n_players
        @assert length(B_seq[i]) == N "B_seq[$i] must have length $N (one matrix per timestep)"
        mi = size(B_seq[i][1], 2)
        control_dims[i] = mi
        for k in 1:N
            @assert size(B_seq[i][k]) == (n, mi) "B_seq[$i][$k] must be ($n × $mi), got $(size(B_seq[i][k]))"
        end
    end

    AM = Vector{Matrix{T}}
    BM = Vector{Vector{Matrix{T}}}
    LinearDynamics{T, AM, BM}(A_seq, B_seq, n, control_dims, N)
end

# ─── Structural queries ────────────────────────────────────────────────────────

"""
    is_ltv(dyn::LinearDynamics) -> Bool

Returns `true` if the dynamics are time-varying (LTV), `false` for LTI.
Zero runtime cost — resolved at compile time via type dispatch.
"""
is_ltv(::LinearDynamics{T, Matrix{T}}) where {T} = false
is_ltv(::LinearDynamics{T, Vector{Matrix{T}}}) where {T} = true

# ─── Accessors ────────────────────────────────────────────────────────────────
#
# All solver code MUST use these accessors. Direct field access (dyn.A, dyn.B)
# is permitted only inside this file. The accessor abstraction is what allows
# LTI and LTV to be used interchangeably in FNELQ and the iLQGames inner loop.

"""
    get_A(dyn::LinearDynamics, k::Int) -> Matrix{T}

Return the system dynamics matrix at timestep k.
For LTI dynamics, k is ignored and the single stored matrix is returned.
For LTV dynamics, returns `A_seq[k]`.
"""
get_A(dyn::LinearDynamics{T, Matrix{T}}, k::Int) where {T} = dyn.A
get_A(dyn::LinearDynamics{T, Vector{Matrix{T}}}, k::Int) where {T} = dyn.A[k]

"""
    get_B(dyn::LinearDynamics, i::Int, k::Int) -> Matrix{T}

Return player i's control matrix at timestep k.
For LTI dynamics, k is ignored.
For LTV dynamics, returns `B_seq[i][k]`.
"""
get_B(dyn::LinearDynamics{T, Matrix{T}}, i::Int, k::Int) where {T} = dyn.B[i]
get_B(dyn::LinearDynamics{T, Vector{Matrix{T}}}, i::Int, k::Int) where {T} = dyn.B[i][k]

"""
    get_B_concatenated(dyn::LinearDynamics, k::Int) -> Matrix{T}

Return horizontally concatenated control matrix [B₁(k) | B₂(k) | ... | Bₙ(k)]
at timestep k. Allocates a new matrix each call — cache the result in hot loops.
"""
function get_B_concatenated(dyn::LinearDynamics, k::Int)
    hcat([get_B(dyn, i, k) for i in 1:length(dyn.control_dims)]...)
end

# ─── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, dyn::LinearDynamics{T}) where {T}
    variant = is_ltv(dyn) ? "LTV" : "LTI"
    n_players = length(dyn.control_dims)
    print(io, "LinearDynamics{$T} [$variant, n=$(dyn.state_dim), $(n_players) players, m=$(dyn.control_dims)]")
end

function Base.show(io::IO, ::MIME"text/plain", dyn::LinearDynamics{T}) where {T}
    variant = is_ltv(dyn) ? "LTV (N=$(dyn.n_steps))" : "LTI"
    n_players = length(dyn.control_dims)
    println(io, "LinearDynamics{$T}")
    println(io, "  Variant       : ", variant)
    println(io, "  State dim     : ", dyn.state_dim)
    println(io, "  Players       : ", n_players)
    println(io, "  Control dims  : ", dyn.control_dims)
end

# ============================================================================
# Separable Dynamics (PD-GNEP structure)
# ============================================================================

"""
    SeparableDynamics{T, F} <: DynamicsSpec{T}

Per-player separable dynamics: ẋᵢ = fᵢ(xᵢ, uᵢ, p, t).

The full dynamics Jacobian ∂f/∂x is block-diagonal. This is the defining
structural property of Partially-Decoupled GNEPs (PD-GNEPs) — dynamics are
separable but coupling enters through shared constraints and costs.

# Fields
- `player_dynamics` : Per-player dynamics functions [f₁, ..., fₙ]
- `state_dims`      : Per-player state dimensions [n₁, ..., nₙ]
- `control_dims`    : Per-player control dimensions [m₁, ..., mₙ]

# Notes
For FALCON and other PD-GNEP solvers, separability of dynamics is a
prerequisite for the partially-decoupled structure. Verify with
`has_separable_dynamics(game)` before dispatching to PD-GNEP solvers.
"""
struct SeparableDynamics{T, F} <: DynamicsSpec{T}
    player_dynamics::Vector{F}
    state_dims::Vector{Int}
    control_dims::Vector{Int}

    function SeparableDynamics(
        player_dynamics::Vector{F},
        state_dims::Vector{Int},
        control_dims::Vector{Int}
    ) where {F}
        n_players = length(player_dynamics)
        @assert length(state_dims) == n_players "state_dims length must equal number of players"
        @assert length(control_dims) == n_players "control_dims length must equal number of players"
        @assert all(>(0), state_dims) "All state dimensions must be positive"
        @assert all(>(0), control_dims) "All control dimensions must be positive"
        new{Float64, F}(player_dynamics, state_dims, control_dims)
    end
end

# ============================================================================
# Coupled Nonlinear Dynamics
# ============================================================================

"""
    CoupledNonlinearDynamics{T, F} <: DynamicsSpec{T}

General nonlinear coupled dynamics: ẋ = f(x, u, p, t).

Use when dynamics cannot be separated by player or when nonlinear coupling
exists between state components. ForwardDiff-compatible `func` is required
for linearization inside the iLQGames solver.

# Fields
- `func`        : Dynamics function f(x, u, p, t) -> ẋ; must be ForwardDiff-compatible
- `state_dim`   : Total state dimension
- `control_dim` : Total control dimension
- `jacobian`    : Optional analytical Jacobian (∂f/∂x, ∂f/∂u); if nothing, AD is used
"""
struct CoupledNonlinearDynamics{T, F} <: DynamicsSpec{T}
    func::F
    state_dim::Int
    control_dim::Int
    jacobian::Union{Nothing, Function}

    function CoupledNonlinearDynamics(
        func::F,
        state_dim::Int,
        control_dim::Int;
        jacobian::Union{Nothing, Function} = nothing
    ) where {F}
        @assert state_dim > 0 "State dimension must be positive"
        @assert control_dim > 0 "Control dimension must be positive"
        new{Float64, F}(func, state_dim, control_dim, jacobian)
    end
end

# ============================================================================
# DynamicsSpec interface — total state/control dimension queries
# ============================================================================

"""
    total_state_dim(dyn::DynamicsSpec) -> Int

Total state dimension across all players.
"""
total_state_dim(dyn::LinearDynamics) = dyn.state_dim
total_state_dim(dyn::SeparableDynamics) = sum(dyn.state_dims)
total_state_dim(dyn::CoupledNonlinearDynamics) = dyn.state_dim

"""
    total_control_dim(dyn::DynamicsSpec) -> Int

Total control dimension across all players.
"""
total_control_dim(dyn::LinearDynamics) = sum(dyn.control_dims)
total_control_dim(dyn::SeparableDynamics) = sum(dyn.control_dims)
total_control_dim(dyn::CoupledNonlinearDynamics) = dyn.control_dim

"""
    n_players(dyn::DynamicsSpec) -> Int

Number of players inferred from the dynamics structure.
"""
n_players(dyn::LinearDynamics) = length(dyn.control_dims)
n_players(dyn::SeparableDynamics) = length(dyn.player_dynamics)
# CoupledNonlinearDynamics does not encode n_players — query from GameProblem.