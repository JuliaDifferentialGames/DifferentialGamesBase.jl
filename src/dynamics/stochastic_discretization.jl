# ============================================================================
# dynamics/stochastic_discretization.jl
#
# Stochastic analogue of DiscreteApproximation (see dynamics/discretization.jl
# for the full rationale: map and Jacobian must be built by the same method).
#
# `StochasticDiscreteApproximation` bundles a mean map, its Jacobian, a
# process-covariance map, and a sample map built consistently from a single
# `AbstractStochasticDynamics`. This is the common representation that both
# `ItoSDEDynamics` (via numerical discretization) and `MarkovKernelDynamics`
# (directly, since it's already discrete) map into — never the other way
# around (there is generally no SDE whose discretization reproduces an
# arbitrary Markov kernel).
#
# Include order: after dynamics/stochastic_dynamics.jl, dynamics/discretization.jl
# (reuses `_rk4_step`), and dynamics/dynamics_interface.jl (reuses
# `evaluate_dynamics`/`dynamics_jacobian`).
# ============================================================================

using Random

"""
    StochasticDiscreteApproximation{T, F, J, Q, S, M}

Consistent discrete-time (mean, Jacobian, covariance, sample) bundle for an
`AbstractStochasticDynamics`.

# Fields
- `mean_map::F`       : `(x,u,p,t) -> mean of x_{k+1}`
- `jac_map::J`         : `(x,u,p,t) -> (Jx, Ju)`, Jacobian of `mean_map`
- `covariance_map::Q`  : `(x,u,p,t) -> process covariance of x_{k+1}` (or `nothing`)
- `sample_map::S`      : `(rng,x,u,p,t) -> sample of x_{k+1}`
- `method`             : discretization method used to build the above
- `state_dim::Int`
- `control_dim::Int`
- `dt::T`
"""
struct StochasticDiscreteApproximation{T, F, J, Q, S, M}
    mean_map::F
    jac_map::J
    covariance_map::Q
    sample_map::S
    method::M
    state_dim::Int
    control_dim::Int
    dt::T
end

"""
    mean_step(sd::StochasticDiscreteApproximation, x, u, p, t) -> Vector

Evaluate the mean of `x_{k+1}` given `(x,u,p,t)`.
"""
mean_step(sd::StochasticDiscreteApproximation, x, u, p, t) = sd.mean_map(x, u, p, t)

"""
    jacobian(sd::StochasticDiscreteApproximation, x, u, p, t) -> (Jx, Ju)

Jacobian of [`mean_step`](@ref), computed consistently with it (same method).
"""
jacobian(sd::StochasticDiscreteApproximation, x, u, p, t) = sd.jac_map(x, u, p, t)

"""
    process_covariance(sd::StochasticDiscreteApproximation, x, u, p, t) -> Matrix

Covariance of `x_{k+1}` given `(x,u,p,t)` (or `nothing` if the wrapped
dynamics don't expose one).
"""
process_covariance(sd::StochasticDiscreteApproximation, x, u, p, t) = sd.covariance_map(x, u, p, t)

"""
    sample_step(rng, sd::StochasticDiscreteApproximation, x, u, p, t) -> Vector
    sample_step(sd::StochasticDiscreteApproximation, x, u, p, t) -> Vector

Draw a sample of `x_{k+1}` given `(x,u,p,t)`, using `rng` (defaults to
`Random.default_rng()`).
"""
sample_step(rng::AbstractRNG, sd::StochasticDiscreteApproximation, x, u, p, t) =
    sd.sample_map(rng, x, u, p, t)
sample_step(sd::StochasticDiscreteApproximation, x, u, p, t) =
    sample_step(Random.default_rng(), sd, x, u, p, t)

"""
    LinearizedGaussianDiscretization <: AbstractDiscretizationMethod

The default [`discretize_stochastic`](@ref) method: exact for `ItoSDEDynamics`
with `LinearDynamics` drift and `ConstantDiffusion`; a first-order
(linearized-mean, linearized-covariance) approximation otherwise.
"""
struct LinearizedGaussianDiscretization <: AbstractDiscretizationMethod end

"""
    discretize_stochastic(dyn::AbstractStochasticDynamics, dt; method=LinearizedGaussianDiscretization())
        -> StochasticDiscreteApproximation

Build a `StochasticDiscreteApproximation` from stochastic dynamics `dyn`.

# Dispatches
- `ItoSDEDynamics` with `LinearDynamics` drift + `ConstantDiffusion`
  → exact discrete Gaussian step (Van Loan augmented exponential), the path
  that reproduces `ProcessNoiseModel`'s additive-Gaussian semantics exactly.
- `ItoSDEDynamics` with nonlinear/separable drift → RK4 mean step (reusing
  `_rk4_step`) + linearized covariance recursion `Σ_{k+1} = AΣA' + GG'dt`.
- `MarkovKernelDynamics` → `mean_map`/`sample_map` delegate straight to the kernel.
"""
function discretize_stochastic(
    dyn::AbstractStochasticDynamics,
    dt::Real;
    method::AbstractDiscretizationMethod = LinearizedGaussianDiscretization()
)
    @assert dt > 0 "dt must be positive"
    return _build_stochastic_discrete_approximation(dyn, dt, method)
end

# ─── ItoSDEDynamics, LinearDynamics drift + constant diffusion: exact ─────────

function _build_stochastic_discrete_approximation(
    dyn::ItoSDEDynamics{T, <:LinearDynamics{T, Matrix{T}}, <:ConstantDiffusion{T}},
    dt::Real,
    ::LinearizedGaussianDiscretization
) where {T}
    # LinearDynamics already IS the discrete-time map (x_{k+1} = Ax_k + Bu_k,
    # per dynamics_interface.jl's documented convention — evaluate_dynamics
    # requires t::Int, the timestep index, not a continuous time). Discretizing
    # it again via MatrixExpDiscretization would double-exponentiate A; use
    # ZOHDiscretization, whose LinearDynamics dispatch is exact passthrough.
    da = discretize(dyn.drift, dt; method = ZOHDiscretization())
    G  = dyn.diffusion.G
    # Discrete process covariance: x_{k+1} = f(x_k,u_k) + w_k, w_k ~ N(0, GGᵀ)
    # — one noise draw per discrete step, matching ProcessNoiseModel exactly
    # (no extra dt scaling; dt is already baked into A/B).
    Q  = G * G'

    mean = (x, u, p, t) -> da_step(da, x, u, p, t)
    jac  = (x, u, p, t) -> jacobian(da, x, u, p, t)
    cov  = (x, u, p, t) -> Q
    samp = (rng, x, u, p, t) -> mean(x, u, p, t) .+ cholesky(Symmetric(Q)).L * randn(rng, T, size(Q, 1))

    n = da.state_dim
    m = da.control_dim
    StochasticDiscreteApproximation{T, typeof(mean), typeof(jac), typeof(cov), typeof(samp),
        LinearizedGaussianDiscretization}(mean, jac, cov, samp, LinearizedGaussianDiscretization(), n, m, dt)
end

# ─── ItoSDEDynamics, general drift: RK4 mean + linearized covariance ─────────

function _build_stochastic_discrete_approximation(
    dyn::ItoSDEDynamics{T},
    dt::Real,
    ::LinearizedGaussianDiscretization
) where {T}
    n = total_state_dim(dyn)
    m = total_control_dim(dyn)

    mean = (x, u, p, t) -> _rk4_step(dyn.drift, x, u, p, t, dt)
    jac  = (x, u, p, t) -> begin
        z = vcat(x, u)
        J = ForwardDiff.jacobian(
            z_var -> _rk4_step(dyn.drift, z_var[1:n], z_var[n+1:end], p, t, dt), z)
        (J[:, 1:n], J[:, n+1:end])
    end
    cov = (x, u, p, t) -> begin
        G = diffusion_matrix(dyn, x, u, p, t)
        dt .* (G * G')
    end
    samp = (rng, x, u, p, t) -> begin
        μ = mean(x, u, p, t)
        Σ = cov(x, u, p, t)
        μ .+ cholesky(Symmetric(Σ) + 1e-12I).L * randn(rng, T, n)
    end

    StochasticDiscreteApproximation{T, typeof(mean), typeof(jac), typeof(cov), typeof(samp),
        LinearizedGaussianDiscretization}(mean, jac, cov, samp, LinearizedGaussianDiscretization(), n, m, dt)
end

# ─── MarkovKernelDynamics: already discrete, delegate to the kernel ──────────

"""
    transition_distribution(dyn::MarkovKernelDynamics, x, u, p, t)

Return the next-state distribution object from the wrapped kernel (e.g. a
`POMDPs.jl` transition distribution). Available whenever the kernel supports
it; not required for `mean_step`/`sample_step` to work.
"""
transition_distribution(dyn::MarkovKernelDynamics, x, u, p, t) = dyn.kernel(x, u, p, t)

function _build_stochastic_discrete_approximation(
    dyn::MarkovKernelDynamics{T},
    dt::Real,
    ::AbstractDiscretizationMethod
) where {T}
    mean = (x, u, p, t) -> _kernel_mean(dyn, x, u, p, t)
    jac  = (x, u, p, t) -> begin
        z = vcat(x, u)
        n = dyn.state_dim
        J = ForwardDiff.jacobian(z_var -> _kernel_mean(dyn, z_var[1:n], z_var[n+1:end], p, t), z)
        (J[:, 1:n], J[:, n+1:end])
    end
    cov  = (x, u, p, t) -> _kernel_cov(dyn, x, u, p, t)
    samp = (rng, x, u, p, t) -> _kernel_sample(rng, dyn, x, u, p, t)

    StochasticDiscreteApproximation{T, typeof(mean), typeof(jac), typeof(cov), typeof(samp),
        typeof(dyn.kernel)}(mean, jac, cov, samp, dyn.kernel, dyn.state_dim, dyn.control_dim, dt)
end

# These three helpers are intentionally duck-typed against whatever the
# wrapped kernel returns (e.g. a POMDPs.jl distribution via `rand`/`mean`/`cov`
# if defined, or a plain NamedTuple for simple test kernels). The POMDPs.jl
# extension may specialize these further.
_kernel_mean(dyn::MarkovKernelDynamics, x, u, p, t) = _dist_mean(dyn.kernel(x, u, p, t))
_kernel_cov(dyn::MarkovKernelDynamics, x, u, p, t) = _dist_cov(dyn.kernel(x, u, p, t))
_kernel_sample(rng::AbstractRNG, dyn::MarkovKernelDynamics, x, u, p, t) = _dist_sample(rng, dyn.kernel(x, u, p, t))

_dist_mean(d) = hasproperty(d, :mean) ? d.mean : d
_dist_cov(d) = hasproperty(d, :cov) ? d.cov : nothing
_dist_sample(rng, d) = hasproperty(d, :sample) ? d.sample : d
