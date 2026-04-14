using LinearAlgebra
using ForwardDiff

# ============================================================================
# Discretization infrastructure
#
# Problem: solvers that enforce dynamics as constraints (FALCON, iLQGames,
# PANGOLIN) must evaluate the discrete-time map and its Jacobian with the
# *same* numerical method. Using DiffEq for constraint evaluation and
# ForwardDiff through a different integration path for linearization creates
# an inconsistency that manifests as an irremovable infeasibility floor —
# the constraint says x(k+1) = f_d(x(k), u(k)) but the linearization uses
# a different f_d, so the two never agree.
#
# Solution: `DiscreteApproximation` wraps a consistent (map, Jacobian) pair
# computed by a single `AbstractDiscretizationMethod`. Solvers receive a
# `DiscreteApproximation` and use it for both constraint evaluation and
# linearization. DiffEq integration (high-accuracy rollout for visualization
# or warm-starting) is a separate path that never enters the constraint layer.
#
# Three methods:
#
#   ZOHDiscretization   : Zero-order hold + RK4. Default. Works for any
#     continuous dynamics. Jacobian via ForwardDiff through the RK4 map.
#     dt must be small enough for RK4 to be accurate.
#
#   MatrixExpDiscretization : Exact for linear time-invariant systems.
#     x(k+1) = e^{A·dt}·x(k) + A⁻¹(e^{A·dt} - I)·B·u(k)  [invertible A]
#     Uses the augmented matrix exponential method (van Loan 1978) which
#     avoids explicit matrix inversion and handles near-singular A.
#     Only valid for LinearDynamics / LinearPlayerDynamics.
#
#   DiffEqDiscretization : Uses OrdinaryDiffEq solver (Tsit5 default) for
#     the forward map. Jacobian via ForwardDiff through the solver call.
#     Requires OrdinaryDiffEq to be loaded (handled by extension).
#     Useful when RK4 is insufficiently accurate for a given dt.
#
# Usage pattern (inside a solver):
#
#   da = discretize(dyn, dt; method=ZOHDiscretization())
#   x_next = step(da, x, u, p, t)       # evaluate discrete map
#   Jx, Ju = jacobian(da, x, u, p, t)  # consistent Jacobian
# ============================================================================

# ============================================================================
# AbstractDiscretizationMethod — tag types
# ============================================================================

"""
    AbstractDiscretizationMethod

Tag type specifying how continuous dynamics are discretized. Passed to
`discretize(dyn, dt; method=...)` to produce a `DiscreteApproximation`.

# Concrete Types
- `ZOHDiscretization`      — Zero-order hold + RK4; general purpose
- `MatrixExpDiscretization` — Exact matrix exponential; linear systems only
- `DiffEqDiscretization`   — OrdinaryDiffEq solver; requires extension
"""
abstract type AbstractDiscretizationMethod end

"""
    ZOHDiscretization <: AbstractDiscretizationMethod

Zero-order hold discretization using RK4. Suitable for any smooth continuous
dynamics. Jacobian is computed via ForwardDiff through the RK4 map, ensuring
constraint and linearization use exactly the same discrete map.

No external packages required.
"""
struct ZOHDiscretization <: AbstractDiscretizationMethod end

"""
    MatrixExpDiscretization <: AbstractDiscretizationMethod

Exact discretization for linear time-invariant systems via matrix exponential:
  x(k+1) = Φ·x(k) + Γ·u(k)
where Φ = e^{A·dt} and Γ = A⁻¹(Φ - I)·B (computed via augmented matrix
exponential to avoid explicit inversion).

Only valid for `LinearDynamics` and `LinearPlayerDynamics`.
Raises an error if applied to nonlinear dynamics.

Reference: Van Loan (1978), "Computing integrals involving the matrix exponential."
"""
struct MatrixExpDiscretization <: AbstractDiscretizationMethod end

"""
    DiffEqDiscretization{A} <: AbstractDiscretizationMethod

Discretization using an OrdinaryDiffEq algorithm. Requires the
`DifferentialGamesBaseDiffEqExt` extension to be loaded.

# Field
- `alg` : OrdinaryDiffEq algorithm (e.g. `Tsit5()`, `Vern7()`, `Rodas5()`)

# Construction
```julia
method = DiffEqDiscretization(Tsit5())    # explicit, non-stiff
method = DiffEqDiscretization(Vern7())    # higher-order explicit
method = DiffEqDiscretization(Rodas5())   # implicit, stiff systems
```
"""
struct DiffEqDiscretization{A} <: AbstractDiscretizationMethod
    alg::A
end

# ============================================================================
# DiscreteApproximation — the consistent (map, Jacobian) bundle
# ============================================================================

"""
    DiscreteApproximation{T, F, J}

A consistent discretization of continuous dynamics: the forward map and its
Jacobian are computed by the same numerical method.

Solvers (FALCON, iLQGames, PANGOLIN) receive a `DiscreteApproximation` and
call `step` / `jacobian` on it. This guarantees that dynamics constraints
and linearizations are consistent, preventing the artificial infeasibility
floor that arises when different numerical methods are used for the two.

# Fields
- `forward_map::F`  : `(x, u, p, t) -> x_next`; ForwardDiff-compatible
- `jac_map::J`      : `(x, u, p, t) -> (Jx, Ju)`; analytical or AD
- `state_dim::Int`
- `control_dim::Int` : Total joint control dimension
- `dt::T`
- `method::AbstractDiscretizationMethod`

# Interface
Use `da_step(da, x, u, p, t)` and `jacobian(da, x, u, p, t)` rather than
accessing fields directly.
"""
struct DiscreteApproximation{T, F, J}
    forward_map::F
    jac_map::J
    state_dim::Int
    control_dim::Int
    dt::T
    method::AbstractDiscretizationMethod
end

"""
    da_step(da::DiscreteApproximation, x, u, p, t) -> Vector

Evaluate the discrete map: x(k+1) = f_d(x(k), u(k), p, t).
Accepts ForwardDiff dual numbers in x and u.

Named `da_step` (discrete approximation step) to avoid clash with `Base.step`.
"""
da_step(da::DiscreteApproximation, x, u, p, t) = da.forward_map(x, u, p, t)

"""
    jacobian(da::DiscreteApproximation, x, u, p, t) -> (Jx, Ju)

Compute the Jacobian of the discrete map at (x, u, p, t).
Uses the same numerical method as `da_step` — guaranteed consistent.

Returns `(Jx, Ju)`:
- `Jx` : (n × n) — ∂f_d/∂x
- `Ju` : (n × m) — ∂f_d/∂u (joint)
"""
jacobian(da::DiscreteApproximation, x, u, p, t) = da.jac_map(x, u, p, t)

# ============================================================================
# discretize — construct DiscreteApproximation from continuous dynamics
# ============================================================================

"""
    discretize(dyn, dt; method=ZOHDiscretization()) -> DiscreteApproximation

Discretize continuous dynamics `dyn` at time step `dt` using `method`.

# Arguments
- `dyn`    : `CoupledNonlinearDynamics`, `SeparableDynamics`, or `LinearDynamics`
- `dt`     : Time step (positive Real)
- `method` : Discretization method (default `ZOHDiscretization()`)

# Dispatches
- `LinearDynamics` + `MatrixExpDiscretization` → exact matrix exponential
- `LinearDynamics` + `ZOHDiscretization`       → RK4 (works but MatrixExp preferred)
- Nonlinear + `ZOHDiscretization`              → RK4 + ForwardDiff Jacobian
- Nonlinear + `MatrixExpDiscretization`        → ERROR (not valid for nonlinear)
- Any + `DiffEqDiscretization`                 → OrdinaryDiffEq (extension required)

# Example
```julia
# General nonlinear dynamics
dyn = CoupledNonlinearDynamics((x,u,p,t) -> [x[2]; -x[1]+u[1]], 2, 1)
da  = discretize(dyn, 0.1)           # ZOH-RK4

# Linear dynamics — exact
dyn_lin = LinearDynamics(A, [B1, B2])
da_lin  = discretize(dyn_lin, 0.1; method=MatrixExpDiscretization())

# High-accuracy (requires OrdinaryDiffEq loaded)
da_v7 = discretize(dyn, 0.1; method=DiffEqDiscretization(Vern7()))
```
"""
function discretize(
    dyn::DynamicsSpec,
    dt::Real;
    method::AbstractDiscretizationMethod = ZOHDiscretization()
)
    @assert dt > 0 "dt must be positive"
    return _build_discrete_approximation(dyn, dt, method)
end

# ============================================================================
# ZOHDiscretization — exact for LinearDynamics, RK4 for continuous types
# ============================================================================

# LinearDynamics is already a discrete map — use exact evaluation, no RK4.
# RK4 must not be called on LinearDynamics because _rk4_step passes t+dt/2
# (Float64) to evaluate_dynamics which requires t::Int for LinearDynamics.
function _build_discrete_approximation(
    dyn::LinearDynamics{T},
    dt::Real,
    ::ZOHDiscretization
) where {T}
    n = total_state_dim(dyn)
    m = total_control_dim(dyn)

    fwd = (x, u, p, t) -> evaluate_dynamics(dyn, x, u, p, Int(t))
    jac = (x, u, p, t) -> dynamics_jacobian(dyn, x, u, p, Int(t))

    DiscreteApproximation{T, typeof(fwd), typeof(jac)}(fwd, jac, n, m, dt, ZOHDiscretization())
end

# Continuous dynamics: RK4 with ZOH control
function _build_discrete_approximation(
    dyn::Union{CoupledNonlinearDynamics{T}, SeparableDynamics{T}},
    dt::Real,
    ::ZOHDiscretization
) where {T}
    n = total_state_dim(dyn)
    m = total_control_dim(dyn)

    fwd = (x, u, p, t) -> _rk4_step(dyn, x, u, p, t, dt)

    jac = (x, u, p, t) -> begin
        z = vcat(x, u)
        J = ForwardDiff.jacobian(
            z_var -> _rk4_step(dyn, z_var[1:n], z_var[n+1:end], p, t, dt),
            z
        )
        (J[:, 1:n], J[:, n+1:end])
    end

    DiscreteApproximation{T, typeof(fwd), typeof(jac)}(fwd, jac, n, m, dt, ZOHDiscretization())
end

# ============================================================================
# MatrixExpDiscretization — exact for LinearDynamics (LTI only)
# ============================================================================
#
# Uses the augmented matrix exponential method (Van Loan 1978):
#
#   M = exp([A  B; 0  0] · dt) = [Φ  Γ; 0  I]
#
# where Φ = e^{Adt} and Γ = ∫₀^{dt} e^{As}B ds.
# This avoids explicit matrix inversion and is numerically stable even
# when A is singular or near-singular.
#
# For LTV LinearDynamics, a different approach would be needed (Magnus expansion
# or step-by-step matrix exponential per timestep) — not implemented here.
# Only LTI is supported; LTV raises an error directing user to ZOH.

function _build_discrete_approximation(
    dyn::LinearDynamics{T, Matrix{T}},   # LTI only
    dt::Real,
    ::MatrixExpDiscretization
) where {T}
    n    = dyn.state_dim
    np   = length(dyn.control_dims)
    m    = sum(dyn.control_dims)
    A    = dyn.A
    B_full = get_B_concatenated(dyn, 1)  # n × m joint B

    # Augmented matrix exponential: [A B; 0 0] · dt
    aug    = zeros(T, n + m, n + m)
    aug[1:n, 1:n] = A
    aug[1:n, n+1:end] = B_full
    M_exp  = exp(aug .* dt)

    Φ = M_exp[1:n, 1:n]         # e^{A·dt}
    Γ = M_exp[1:n, n+1:end]     # ∫₀^{dt} e^{As}B ds

    # Forward map: exact linear step
    fwd = (x, u, p, t) -> Φ * x + Γ * u

    # Jacobian: exact, constant (LTI)
    jac = (x, u, p, t) -> (Φ, Γ)

    DiscreteApproximation{T, typeof(fwd), typeof(jac)}(
        fwd, jac, n, m, dt, MatrixExpDiscretization()
    )
end

# LTV LinearDynamics: direct to ZOH with informative error
function _build_discrete_approximation(
    dyn::LinearDynamics{T, Vector{Matrix{T}}},   # LTV
    dt::Real,
    ::MatrixExpDiscretization
) where {T}
    error("""
    MatrixExpDiscretization requires LTI LinearDynamics.
    LTV LinearDynamics detected (A is a Vector{Matrix}).
    Use ZOHDiscretization() or DiffEqDiscretization() instead.
    """)
end

# Nonlinear dynamics: matrix exponential is not valid
function _build_discrete_approximation(
    dyn::Union{CoupledNonlinearDynamics, SeparableDynamics},
    dt::Real,
    ::MatrixExpDiscretization
)
    error("""
    MatrixExpDiscretization is only valid for LinearDynamics.
    For nonlinear dynamics, use ZOHDiscretization() or DiffEqDiscretization().
    """)
end

# ============================================================================
# DiffEqDiscretization — delegated to extension
# ============================================================================
#
# The actual implementation lives in DifferentialGamesBaseDiffEqExt.jl.
# If the extension is not loaded, a clear error is raised.

function _build_discrete_approximation(
    dyn::DynamicsSpec,
    dt::Real,
    method::DiffEqDiscretization
)
    error("""
    DiffEqDiscretization requires OrdinaryDiffEq to be loaded.
    Add `using OrdinaryDiffEq` before calling discretize with DiffEqDiscretization.
    If you have OrdinaryDiffEq installed, ensure DifferentialGamesBase is loaded first.
    """)
end

# ============================================================================
# RK4 step — shared implementation used by ZOHDiscretization and rollout
# ============================================================================

"""
    _rk4_step(dyn, x, u, p, t, dt) -> x_next

One RK4 integration step with zero-order hold on control over [t, t+dt].
ForwardDiff-compatible: x and u may be dual numbers.

Used by ZOHDiscretization (for constraint consistency) and as the fallback
in rollout when the DiffEq extension is not loaded.
"""
function _rk4_step(
    dyn::DynamicsSpec,
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Real,
    dt::Real
)
    k1 = evaluate_dynamics(dyn, x,             u, p, t)
    k2 = evaluate_dynamics(dyn, x + dt/2 * k1, u, p, t + dt/2)
    k3 = evaluate_dynamics(dyn, x + dt/2 * k2, u, p, t + dt/2)
    k4 = evaluate_dynamics(dyn, x + dt    * k3, u, p, t + dt)
    return x + (dt/6) * (k1 + 2k2 + 2k3 + k4)
end

# ============================================================================
# validate_discretization — sanity checks before a solve
# ============================================================================

"""
    validate_discretization(da::DiscreteApproximation, game::GameProblem)

Check that a `DiscreteApproximation` is compatible with the game's state
and control dimensions. Raises AssertionError on mismatch.
"""
function validate_discretization(da::DiscreteApproximation, game::GameProblem)
    @assert(da.state_dim == total_state_dim(game.dynamics),
        "DiscreteApproximation state_dim $(da.state_dim) ≠ game state_dim $(total_state_dim(game.dynamics))")
    @assert(da.control_dim == total_control_dim(game.dynamics),
        "DiscreteApproximation control_dim $(da.control_dim) ≠ game control_dim $(total_control_dim(game.dynamics))")
end