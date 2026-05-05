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
