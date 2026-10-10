# ============================================================================
# uncertainty/gaussian_belief.jl
#
# Gaussian representation of state uncertainty, shared by the Phase 1
# linearized-Gaussian chance-constraint pipeline and (per the roadmap in
# docs/src/api/stochastic.md) future EKF/UKF belief-space POMDP work.
# ============================================================================

"""
    GaussianBelief{T}

A Gaussian belief over the joint state: `x ~ N(μ, Σ)`.

# Fields
- `μ::Vector{T}`
- `Σ::Matrix{T}`
"""
struct GaussianBelief{T}
    μ::Vector{T}
    Σ::Matrix{T}

    function GaussianBelief{T}(μ::Vector{T}, Σ::Matrix{T}) where {T}
        @assert length(μ) == size(Σ, 1) == size(Σ, 2) "μ and Σ dimensions must agree"
        @assert issymmetric(Σ) "Σ must be symmetric"
        new{T}(μ, Σ)
    end
end

GaussianBelief(μ::Vector{T}, Σ::Matrix{T}) where {T} = GaussianBelief{T}(μ, Σ)

"""
    mean(b::GaussianBelief) -> Vector

The belief's mean `μ`.
"""
mean(b::GaussianBelief) = b.μ

"""
    covariance(b::GaussianBelief) -> Matrix

The belief's covariance `Σ`.
"""
covariance(b::GaussianBelief) = b.Σ

Base.length(b::GaussianBelief) = length(b.μ)

function Base.show(io::IO, b::GaussianBelief{T}) where {T}
    print(io, "GaussianBelief{$T}(dim=", length(b), ")")
end
