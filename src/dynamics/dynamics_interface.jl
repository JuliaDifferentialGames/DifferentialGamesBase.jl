using LinearAlgebra
using ForwardDiff

# ============================================================================
# dynamics_interface.jl
#
# Include order: discretization.jl MUST be included before this file in
# DifferentialGamesBase.jl because _rk4_step is defined there.
#
# Two distinct paths — never mix them:
#
#   SOLVER PATH  — DiscreteApproximation (from discretization.jl)
#     step(da, x, u, p, t) and jacobian(da, x, u, p, t)
#     Same discrete map for constraint evaluation and linearization.
#     Prevents the artificial infeasibility floor.
#
#   ROLLOUT PATH — rollout / rollout_strategy
#     High-accuracy DiffEq or RK4 for visualization and warm-starting.
#     Never enters any solver constraint layer.
# ============================================================================

# ============================================================================
# evaluate_dynamics — single-step evaluation
# ============================================================================

"""
    evaluate_dynamics(dyn::DynamicsSpec{T}, x, u, p, t) -> Vector

Evaluate dynamics at (x, u, p, t). All implementations are ForwardDiff-compatible.

- `LinearDynamics`           : Discrete map x(k+1)=A(k)x+ΣBᵢuᵢ. `t` is Int (timestep k).
- `SeparableDynamics`        : Continuous RHS [ẋ₁;…;ẋₙ]. `t` is Real.
- `CoupledNonlinearDynamics` : Continuous RHS ẋ. `t` is Real.
"""
function evaluate_dynamics end

function evaluate_dynamics(
    dyn::LinearDynamics{T},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Int
) where {T}
    result = get_A(dyn, t) * x
    offset = 0
    for i in 1:length(dyn.control_dims)
        mi     = dyn.control_dims[i]
        result = result + get_B(dyn, i, t) * u[offset+1:offset+mi]
        offset += mi
    end
    return result
end

function evaluate_dynamics(
    dyn::SeparableDynamics{T},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Real
) where {T}
    s_offs = [0; cumsum(dyn.state_dims)]
    c_offs = [0; cumsum(dyn.control_dims)]
    parts  = map(1:length(dyn.player_dynamics)) do i
        dyn.player_dynamics[i](
            x[s_offs[i]+1 : s_offs[i+1]],
            u[c_offs[i]+1 : c_offs[i+1]],
            p, t
        )
    end
    return vcat(parts...)
end

function evaluate_dynamics(
    dyn::CoupledNonlinearDynamics{T},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Real
) where {T}
    return dyn.func(x, u, p, t)
end

# ============================================================================
# dynamics_jacobian — continuous-RHS Jacobian
#
# NOTE: For solver constraint/linearization use, call jacobian(da, x, u, p, t)
# on a DiscreteApproximation instead. This function gives the Jacobian of the
# continuous RHS, which is NOT the same as the discrete-map Jacobian.
# ============================================================================

"""
    dynamics_jacobian(dyn::DynamicsSpec{T}, x, u, p, t) -> (Jx, Ju)

Jacobian of the dynamics at (x, u, p, t).

For solver use, prefer `jacobian(da, ...)` on a `DiscreteApproximation`.
This function gives the continuous-RHS Jacobian (or exact linear map for
`LinearDynamics`), not a discretized Jacobian.
"""
function dynamics_jacobian end

function dynamics_jacobian(
    dyn::LinearDynamics{T},
    x::AbstractVector, u::AbstractVector, p, t::Int
) where {T}
    return (get_A(dyn, t), get_B_concatenated(dyn, t))
end

function dynamics_jacobian(
    dyn::CoupledNonlinearDynamics{T},
    x::AbstractVector, u::AbstractVector, p, t::Real
) where {T}
    dyn.jacobian !== nothing && return dyn.jacobian(x, u, p, t)
    return _ad_jac((xx, uu) -> dyn.func(xx, uu, p, t), x, u)
end

function dynamics_jacobian(
    dyn::SeparableDynamics{T},
    x::AbstractVector, u::AbstractVector, p, t::Real
) where {T}
    s_offs = [0; cumsum(dyn.state_dims)]
    c_offs = [0; cumsum(dyn.control_dims)]
    n_tot  = sum(dyn.state_dims); m_tot = sum(dyn.control_dims)
    Te = eltype(x)
    Jx = zeros(Te, n_tot, n_tot); Ju = zeros(Te, n_tot, m_tot)
    for i in 1:length(dyn.player_dynamics)
        sr = s_offs[i]+1:s_offs[i+1]; cr = c_offs[i]+1:c_offs[i+1]
        fi = dyn.player_dynamics[i]
        Jxi, Jui = _ad_jac((xx, uu) -> fi(xx, uu, p, t), x[sr], u[cr])
        Jx[sr, sr] = Jxi; Ju[sr, cr] = Jui
    end
    return (Jx, Ju)
end

function _ad_jac(f::F, x, u) where {F}
    nx = length(x)
    J  = ForwardDiff.jacobian(z -> f(z[1:nx], z[nx+1:end]), vcat(x, u))
    return (J[:, 1:nx], J[:, nx+1:end])
end

# ============================================================================
# dynamics_residual — nonlinear constraint D^i(z^i) for solver use
# ============================================================================

"""
    dynamics_residual(dyn, i, z_flat, dt) -> Vector{T}

Evaluate the nonlinear discrete-time dynamics residual for player `i`:

    D^i(z^i) = [x^i_{t+1} - f^i(x^i_t, u^i_t, dt)]_{t=0}^{N-1}

where `z_flat` is the flat trajectory vector for player i in the layout
`[x_0; u_0; x_1; u_1; ...; x_{N-1}; u_{N-1}; x_N]` with `n_x` state
and `n_u` control dimensions per step.

`dt` is the time step for Euler discretisation: `x_{t+1} ≈ x_t + dt·f(x_t, u_t)`.

Returns a vector of length `N·n_x` stacking the per-step residuals.
Zero at a dynamically feasible trajectory.
"""
function dynamics_residual(
    dyn::SeparableDynamics{T},
    i::Int,
    z_flat::AbstractVector,
    dt::Real
) where {T}
    n_x = dyn.state_dims[i]
    n_u = dyn.control_dims[i]
    step = n_x + n_u
    N    = (length(z_flat) - n_x) ÷ step
    @assert length(z_flat) == N * step + n_x "z_flat length mismatch for player $i"
    fi   = dyn.player_dynamics[i]
    res  = Vector{eltype(z_flat)}(undef, N * n_x)
    for t in 0:N-1
        x_t   = z_flat[t*step + 1         : t*step + n_x]
        u_t   = z_flat[t*step + n_x + 1   : t*step + n_x + n_u]
        x_tp1 = z_flat[(t+1)*step + 1     : (t+1)*step + n_x]
        x_next = x_t .+ dt .* fi(x_t, u_t, nothing, t * dt)
        res[t*n_x+1 : (t+1)*n_x] = x_tp1 .- x_next
    end
    return res
end

# ============================================================================
# DiffEq integration stubs — overridden by DifferentialGamesBaseDiffEqExt
# ============================================================================

"""
    _diffeq_step(dyn, x, u, p, t, dt; alg, abstol, reltol) -> x_next

Single integration step via OrdinaryDiffEq. Stub — overridden by extension.
"""
function _diffeq_step end

"""
    _diffeq_available() -> Bool

Returns true if the DiffEq extension is loaded.
"""
_diffeq_available() = false

# ============================================================================
# _rollout_step — dispatch to correct integration per dynamics type
# ============================================================================

# LinearDynamics: always exact discrete map
function _rollout_step(
    dyn::LinearDynamics, x, u, p, times, k::Int;
    integrator=nothing, abstol=nothing, reltol=nothing
)
    return evaluate_dynamics(dyn, x, u, p, k)
end

# Continuous: DiffEq if loaded and integrator specified, else RK4
function _rollout_step(
    dyn::Union{CoupledNonlinearDynamics, SeparableDynamics},
    x, u, p, times, k::Int;
    integrator = nothing,
    abstol::Real = 1e-8,
    reltol::Real = 1e-6
)
    dt = times[k+1] - times[k]
    t  = times[k]
    if integrator !== nothing && _diffeq_available()
        return _diffeq_step(dyn, x, u, p, t, dt; alg=integrator, abstol, reltol)
    else
        return _rk4_step(dyn, x, u, p, t, dt)   # defined in discretization.jl
    end
end

# ============================================================================
# rollout — trajectory propagation (visualization / warm-starting)
# ============================================================================

"""
    rollout(dyn, x0, U, p, times; integrator=nothing, abstol=1e-8, reltol=1e-6) -> Matrix

Propagate dynamics from x0 under control sequence U.

NOT the solver path. For solvers, use `step(da, ...)` on a `DiscreteApproximation`.

# Arguments
- `dyn`        : Dynamics specification
- `x0`         : Initial state
- `U`          : Joint controls, shape (m_total × N)
- `p`          : Parameters
- `times`      : Time vector, length N+1

# Keyword Arguments
- `integrator` : OrdinaryDiffEq algorithm (e.g. `Tsit5()`); `nothing` uses RK4 fallback.
  Ignored for `LinearDynamics` (always uses exact discrete map).
- `abstol`, `reltol` : Tolerances forwarded to DiffEq solver.

# Returns
`X :: Matrix` shape (n × N+1).
"""
function rollout(
    dyn::DynamicsSpec{T},
    x0::AbstractVector,
    U::AbstractMatrix,
    p,
    times::AbstractVector;
    integrator = nothing,
    abstol::Real = 1e-8,
    reltol::Real = 1e-6
) where {T}
    N = size(U, 2)
    @assert length(times) == N + 1 "times must have N+1 entries for N steps"
    @assert(length(x0) == total_state_dim(dyn),
        "x0 length $(length(x0)) ≠ state dim $(total_state_dim(dyn))")

    X = Matrix{eltype(x0)}(undef, length(x0), N + 1)
    X[:, 1] = x0
    for k in 1:N
        X[:, k+1] = _rollout_step(dyn, X[:, k], U[:, k], p, times, k;
                                   integrator, abstol, reltol)
    end
    return X
end

# ============================================================================
# rollout_strategy — AbstractStrategy-driven rollout
# ============================================================================

"""
    rollout_strategy(dyn, x0, strategy, p; η=1.0, integrator=nothing,
                     abstol=1e-8, reltol=1e-6) -> (X, U)

Roll out an `AbstractStrategy` (open-loop or feedback) against the dynamics.

- `OpenLoopStrategy`: applies ûᵢ(k) directly; x is ignored in control law.
- `FeedbackStrategy`: applies uᵢ(k) = ûᵢ(k) - Pᵢ(k)·δx(k) - η·αᵢ(k).

# Returns
`(X, U)` — (n × N+1) states and (m_total × N) joint controls.
"""
function rollout_strategy(
    dyn::DynamicsSpec{T},
    x0::AbstractVector,
    strategy::AbstractStrategy{T},
    p;
    η::Real = one(T),
    integrator = nothing,
    abstol::Real = 1e-8,
    reltol::Real = 1e-6
) where {T}
    N       = n_steps(strategy)
    m_total = sum(get_control_dims(strategy))
    times   = get_times(strategy)

    X = Matrix{eltype(x0)}(undef, length(x0), N + 1)
    U = Matrix{eltype(x0)}(undef, m_total, N)
    X[:, 1] = x0

    for k in 1:N
        u_k     = apply_strategy(strategy, X[:, k], k; η)
        U[:, k] = u_k
        X[:, k+1] = _rollout_step(dyn, X[:, k], u_k, p, times, k;
                                   integrator, abstol, reltol)
    end
    return (X, U)
end