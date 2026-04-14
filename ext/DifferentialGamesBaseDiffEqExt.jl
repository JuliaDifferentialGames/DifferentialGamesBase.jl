module DifferentialGamesBaseDiffEqExt

# ============================================================================
# Package extension: DifferentialGamesBase + OrdinaryDiffEq
#
# Loaded automatically when both packages are in the environment.
# Provides two things:
#
# 1. _diffeq_step — high-accuracy single-step integration for rollout
#    (visualization / warm-starting path only, NOT for solver constraints)
#
# 2. _build_discrete_approximation for DiffEqDiscretization — a consistent
#    (map, Jacobian) pair where both are computed via the same ODE solve.
#    The Jacobian is obtained by ForwardDiff through the ODE solver call,
#    which requires the dynamics to be ForwardDiff-compatible end-to-end.
#
# Zero-order hold assumption:
#   Control u is held constant over [t, t+dt]. Each interval is solved as
#   a fresh ODEProblem — correct for ZOH and avoids event detection overhead.
#
# Fixed vs adaptive stepping:
#   Default (from rollout): adaptive=false, dt=interval_length.
#   For DiffEqDiscretization: also adaptive=false to ensure the discrete map
#   is deterministic (same input → same output), which is required for a
#   consistent constraint/Jacobian pair.
# ============================================================================

using DifferentialGamesBase
using OrdinaryDiffEq
using ForwardDiff

import DifferentialGamesBase:
    _diffeq_step,
    _diffeq_available,
    _build_discrete_approximation,
    evaluate_dynamics,
    DiffEqDiscretization,
    DiscreteApproximation,
    DynamicsSpec,
    CoupledNonlinearDynamics,
    SeparableDynamics,
    LinearDynamics,
    total_state_dim,
    total_control_dim

# ============================================================================
# Signal that DiffEq is available
# ============================================================================

DifferentialGamesBase._diffeq_available() = true

# ============================================================================
# _diffeq_step — rollout path (visualization only)
# ============================================================================

"""
    _diffeq_step(dyn, x, u, p, t, dt; alg, abstol, reltol) -> x_next

Integrate one interval [t, t+dt] via OrdinaryDiffEq with ZOH control.
Used exclusively by the rollout path — not by solver constraints.
"""
function DifferentialGamesBase._diffeq_step(
    dyn::Union{CoupledNonlinearDynamics, SeparableDynamics},
    x::AbstractVector,
    u::AbstractVector,
    p,
    t::Real,
    dt::Real;
    alg = Tsit5(),
    abstol::Real = 1e-8,
    reltol::Real = 1e-6
)
    ode_f = ODEFunction(
        (du, x_s, params, time) -> begin
            du .= evaluate_dynamics(dyn, x_s, u, params, time)
        end
    )
    prob = ODEProblem(ode_f, x, (t, t + dt), p)
    sol  = solve(prob, alg;
        adaptive   = false,
        dt         = dt,
        save_start = false,
        save_end   = true,
        dense      = false
    )
    return sol.u[end]
end

# ============================================================================
# DiffEqDiscretization — consistent (map, Jacobian) pair via ODE solver
# ============================================================================
#
# The Jacobian is computed by ForwardDiff through the ODE solve. This requires:
# 1. The dynamics function is ForwardDiff-compatible (no Float64 type assertions)
# 2. The ODE algorithm supports dual number inputs (explicit Runge-Kutta does;
#    implicit methods like Rodas5 may not differentiate through correctly)
#
# Recommendation: use Tsit5() or Vern7() for ForwardDiff-through-ODE.
# For stiff systems where implicit methods are needed, use ZOHDiscretization
# or MatrixExpDiscretization with linearized dynamics instead.

function DifferentialGamesBase._build_discrete_approximation(
    dyn::DynamicsSpec{T},
    dt::Real,
    method::DiffEqDiscretization
) where {T}
    alg = method.alg
    n   = total_state_dim(dyn)
    m   = total_control_dim(dyn)

    # Forward map via ODE solve
    fwd = (x, u, p, t) -> begin
        ode_f = ODEFunction(
            (du, x_s, params, time) -> begin
                du .= evaluate_dynamics(dyn, x_s, u, params, time)
            end
        )
        prob = ODEProblem(ode_f, x, (t, t + dt), p)
        sol  = solve(prob, alg;
            adaptive   = false,
            dt         = dt,
            save_start = false,
            save_end   = true,
            dense      = false
        )
        sol.u[end]
    end

    # Jacobian via ForwardDiff through the ODE solver
    # Requires dyn to be ForwardDiff-compatible and alg to be explicit
    jac = (x, u, p, t) -> begin
        z = vcat(x, u)
        J = ForwardDiff.jacobian(
            z_var -> begin
                ode_f = ODEFunction(
                    (du, x_s, params, time) -> begin
                        du .= evaluate_dynamics(dyn, x_s, z_var[n+1:end], params, time)
                    end
                )
                prob = ODEProblem(ode_f, z_var[1:n], (t, t + dt), p)
                sol  = solve(prob, alg;
                    adaptive   = false,
                    dt         = dt,
                    save_start = false,
                    save_end   = true,
                    dense      = false
                )
                sol.u[end]
            end,
            z
        )
        (J[:, 1:n], J[:, n+1:end])
    end

    DiscreteApproximation{T, typeof(fwd), typeof(jac)}(
        fwd, jac, n, m, dt, method
    )
end

end # module