# ============================================================================
# constraints/continuous_time_constraints.jl
#
# ContinuousTimeConstraint — wraps a stage constraint that must hold
# *continuously* along a trajectory (not just at knot points).
#
# FALCON handles these by augmenting player i's state with an accumulator
# y^i ∈ R^dim that integrates q_c(C(x,u)) over each time interval:
#
#   ẏ^i_j = q_c(c_j(x, u, p, t))        j = 1..dim    (Eq. 8)
#
# and enforcing the node constraint (Eq. 9):
#
#   y^i_j(t_{k+1}) - y^i_j(t_k) ≤ ε     j = 1..dim
#
# which implies the integral of the cubic penalty over [t_k, t_{k+1}] is ≤ ε,
# guaranteeing continuous satisfaction up to tolerance ε when ε is small.
#
# The node constraints appear as hard per-player h_cvx inequalities in the
# joint SOCP (is_convex=true), not as coupling AL penalties.
#
# Design notes:
#   - Subtypes AbstractPrivateInequality: involves one player, C(x,u) ≤ 0.
#   - evaluate_constraint returns the raw constraint value c(x,u,p,t) for
#     diagnostics and infeasibility measurement.
#   - is_convex = false: handled specially in _linearise_dynamics / _linearise_hcvx,
#     NOT routed to the standard h_cvx or AL coupling paths.
#   - Solvers detect ContinuousTimeConstraint via isa(c, ContinuousTimeConstraint)
#     and build augmented dynamics/node constraints accordingly.
# ============================================================================

"""
    ContinuousTimeConstraint{F, T} <: AbstractPrivateInequality

Private constraint that must hold *continuously* along player `player`'s
trajectory. FALCON enforces it via the cubic exterior penalty integral:

    ẏ^i_j = q_c(c_j(x, u, p, t))  →  node constraint y^i_j(t_{k+1}) - y^i_j(t_k) ≤ ε

# Fields
- `player` : Player index (1-based)
- `func`   : `(x, u, p, t) → AbstractVector{T}` — the constraint values c(x,u,p,t) ≤ 0.
             Must be ForwardDiff-compatible.
- `dim`    : Output dimension of `func` (≥ 1)
- `ε`      : Node tolerance per component (Eq. 9); smaller ε → tighter constraint

# Example
```julia
# Player 1 must stay below height 5.0 at all times
c = ContinuousTimeConstraint(1,
    func = (x, u, p, t) -> [x[3] - 5.0],
    dim  = 1,
    ε    = 1e-3
)
```
"""
struct ContinuousTimeConstraint{F, T} <: AbstractPrivateInequality
    player::Int
    func::F
    dim::Int
    ε::T

    function ContinuousTimeConstraint(
        player::Int, func::F, dim::Int, ε::T
    ) where {F, T <: Real}
        @assert player > 0 "player must be positive"
        @assert dim > 0    "dim must be positive"
        @assert ε > 0      "ε must be positive"
        new{F, T}(player, func, dim, ε)
    end
end

function evaluate_constraint(c::ContinuousTimeConstraint, x, u, p, t)
    return c.func(x, u, p, t)
end

# Jacobian via ForwardDiff (same as PrivateNonlinearInequality)
function constraint_jacobian(c::ContinuousTimeConstraint, x, u, p, t)
    nx = length(x); nu = length(u)
    z  = vcat(x, u)
    J  = ForwardDiff.jacobian(
        z_var -> c.func(z_var[1:nx], z_var[nx+1:end], p, t), z
    )
    return (J[:, 1:nx], J[:, nx+1:end])
end

# Not convex in z (the continuous-time handling is special, not standard h_cvx)
is_convex(::ContinuousTimeConstraint) = false

# ── Ergonomic constructor ─────────────────────────────────────────────────────

"""
    continuous_time_constraint(player; func, dim, ε) -> ContinuousTimeConstraint

Continuous-time inequality constraint `func(x, u, p, t) ≤ 0` for `player`.

FALCON enforces it by integrating the cubic exterior penalty `q_c(c(x,u,p,t))`
over each time interval and bounding the increment by `ε` (Eq. 9).

# Arguments
- `player` : Player index
- `func`   : `(x, u, p, t) → Vector` — ForwardDiff-compatible
- `dim`    : Output dimension of `func`
- `ε`      : Node tolerance (default: 1e-3)

# Example
```julia
c = continuous_time_constraint(1;
    func = (x, u, p, t) -> [norm(u) - u_max],
    dim  = 1,
    ε    = 1e-3
)
```
"""
function continuous_time_constraint(
    player::Int;
    func::F,
    dim::Int,
    ε::T = 1e-3
) where {F, T <: Real}
    ContinuousTimeConstraint(player, func, dim, ε)
end

# ============================================================================
# ContinuousTimeSharedConstraint — multi-agent (coupling) counterpart
#
# The FALCON paper's own formulation is NOT private-only: Eq. 7 defines the
# violation metric Λ^i(t,x(t),u^i(t)) = Σ_ρ q_c^i(C_ρ(x(t),u^i(t))) with C_ρ
# a function of the FULL joint state x(t), not agent i's state alone, and
# Theorem IV.8(3) proves global convergence to a point that is "approximately
# coupling-feasible in the continuous-time sense... for every coupling
# constraint component ρ" — i.e. the paper's own convergence guarantee
# explicitly covers coupling (shared/multi-agent) constraints under this same
# continuous-time treatment. `ContinuousTimeConstraint` above only implements
# the private special case; this type closes that gap for constraints that
# genuinely couple ≥2 players (e.g. minimum inter-agent separation).
# ============================================================================

"""
    ContinuousTimeSharedConstraint{F, T} <: AbstractSharedInequality

Shared (multi-agent) counterpart to [`ContinuousTimeConstraint`](@ref): a
coupling constraint `c(x_joint, u_joint, p, t) ≤ 0` that must hold
*continuously* along every involved player's trajectory, enforced via the
same cubic exterior-penalty node scheme (Eq. 8-9): `ẏ_j = q_c(c_j(x,u,p,t))`,
node constraint `y_j(t_{k+1}) - y_j(t_k) ≤ ε`.

# Fields
- `players` : Player indices involved (≥2), sorted ascending
- `func`    : `(x_joint, u_joint, p, t) → AbstractVector{T}` — the joint
              constraint values `c(x,u,p,t) ≤ 0`. `x_joint`/`u_joint` are the
              FULL concatenated state/control across all players in the
              game (same convention `_linearise_coupling` already uses for
              ordinary `SharedInequality`), so `func` may reference any
              subset of players, not just the lowest-indexed one.
- `dim`     : Output dimension of `func` (≥ 1)
- `ε`       : Node tolerance per component (Eq. 9); smaller ε → tighter
              continuous-time enforcement

# Example
```julia
# Players 1 and 2 must stay ≥ d_min apart at all times, not just at knots
c = ContinuousTimeSharedConstraint([1, 2],
    (x, u, p, t) -> begin
        p1 = x[1:3]; p2 = x[n_x1+1:n_x1+3]
        [d_min - sqrt(sum(abs2, p1 .- p2) + ε_reg^2)]
    end,
    1, 1e-3
)
```
"""
struct ContinuousTimeSharedConstraint{F, T} <: AbstractSharedInequality
    players::Vector{Int}
    func::F
    dim::Int
    ε::T

    function ContinuousTimeSharedConstraint(
        players::Vector{Int}, func::F, dim::Int, ε::T
    ) where {F, T <: Real}
        @assert length(players) >= 2 "ContinuousTimeSharedConstraint requires ≥2 players"
        @assert allunique(players)   "Player indices must be distinct"
        @assert dim > 0              "dim must be positive"
        @assert ε > 0                "ε must be positive"
        new{F, T}(sort(players), func, dim, ε)
    end
end

function evaluate_constraint(c::ContinuousTimeSharedConstraint, x, u, p, t)
    return c.func(x, u, p, t)
end

function constraint_jacobian(c::ContinuousTimeSharedConstraint, x, u, p, t)
    nx = length(x); nu = length(u)
    z  = vcat(x, u)
    J  = ForwardDiff.jacobian(
        z_var -> c.func(z_var[1:nx], z_var[nx+1:end], p, t), z
    )
    return (J[:, 1:nx], J[:, nx+1:end])
end

# Not convex in z (special-cased continuous-time handling, same as the
# private ContinuousTimeConstraint) — routed to FALCON's dedicated CTC node-
# constraint path, not the standard AL coupling path.
is_convex(::ContinuousTimeSharedConstraint) = false

"""
    continuous_time_shared_constraint(players; func, dim, ε) -> ContinuousTimeSharedConstraint

Ergonomic constructor mirroring [`continuous_time_constraint`](@ref) for the
shared (multi-agent) case.

# Example
```julia
c = continuous_time_shared_constraint([1, 2];
    func = (x, u, p, t) -> [d_min - norm(x[1:3] .- x[n_x1+1:n_x1+3])],
    dim  = 1,
    ε    = 1e-3
)
```
"""
function continuous_time_shared_constraint(
    players::Vector{Int};
    func::F,
    dim::Int,
    ε::T = 1e-3
) where {F, T <: Real}
    ContinuousTimeSharedConstraint(players, func, dim, ε)
end
