# ============================================================================
# uncertainty/propagation.jl
#
# Uncertainty propagation is NOT a chance-constraint evaluator — it is the
# separate concern of how a `GaussianBelief` evolves under stochastic
# dynamics. Kept as its own interface (`AbstractUncertaintyPropagator`) so a
# `ChanceConstraint` can be compiled against any propagator/evaluator pair
# without redefinition (see constraints/chance_constraints.jl).
#
# Include order: after uncertainty/gaussian_belief.jl and
# dynamics/stochastic_discretization.jl (StochasticDiscreteApproximation).
# ============================================================================

"""
    AbstractUncertaintyPropagator

Interface for propagating a `GaussianBelief` forward one step under a
`StochasticDiscreteApproximation`. Implement
`propagate(propagator, sd, belief, u, p, t) -> GaussianBelief`.
"""
abstract type AbstractUncertaintyPropagator end

"""
    LinearizedGaussianPropagation <: AbstractUncertaintyPropagator

First-order (extended-Kalman-style) moment propagation:

    μ_{k+1} = mean_step(sd, μ_k, u_k, p, t)
    Σ_{k+1} = A_k Σ_k A_kᵀ + Q_k,   A_k = jacobian(sd, μ_k, u_k, p, t)[1],
                                    Q_k = process_covariance(sd, μ_k, u_k, p, t)

Exact for linear dynamics with additive Gaussian noise (reproduces the old
`ProcessNoiseModel` covariance recursion exactly); a first-order
approximation otherwise. The default propagator for Phase 1.
"""
struct LinearizedGaussianPropagation <: AbstractUncertaintyPropagator end

"""
    propagate(propagator::AbstractUncertaintyPropagator, sd::StochasticDiscreteApproximation,
               belief::GaussianBelief, u, p, t) -> GaussianBelief

Propagate `belief` forward one step under `sd` using `propagator`.
"""
function propagate(
    ::LinearizedGaussianPropagation,
    sd::StochasticDiscreteApproximation,
    belief::GaussianBelief{T},
    u::AbstractVector,
    p,
    t
) where {T}
    μ = mean(belief)
    Σ = covariance(belief)
    μ_next = mean_step(sd, μ, u, p, t)
    A, _ = jacobian(sd, μ, u, p, t)
    Q = process_covariance(sd, μ, u, p, t)
    Σ_next = A * Σ * A' + Q
    GaussianBelief(μ_next, Symmetric(Σ_next) |> Matrix)
end

"""
    UnscentedPropagation <: AbstractUncertaintyPropagator

Unscented-transform moment propagation. Reserved interface point —
not implemented in this round. See docs/src/api/stochastic.md, Phase 3.
"""
struct UnscentedPropagation <: AbstractUncertaintyPropagator end
propagate(::UnscentedPropagation, args...) =
    error("UnscentedPropagation is not implemented yet — see docs/src/api/stochastic.md, Phase 3 roadmap.")

"""
    ParticlePropagation <: AbstractUncertaintyPropagator

Particle/Monte-Carlo moment propagation. Reserved interface point —
not implemented in this round. See docs/src/api/stochastic.md, Phase 3.
"""
struct ParticlePropagation <: AbstractUncertaintyPropagator
    n_particles::Int
end
propagate(::ParticlePropagation, args...) =
    error("ParticlePropagation is not implemented yet — see docs/src/api/stochastic.md, Phase 3 roadmap.")
