# ============================================================================
# uncertainty/kalman_filter.jl
#
# Common-observation linear-Gaussian filtering for LQ-type stochastic games.
#
# SCOPE: every player receives the SAME measurement history
#     y_{k+1} = H_{k+1} x_{k+1} + v_{k+1},  v ~ N(0, V),
# so the players share one estimate x̂_k and one posterior covariance Σᵉ_k. The
# separation of estimation and (feedback-Nash) control is only valid under this
# common-information, linear-Gaussian, additive-noise structure; private or
# asymmetric observations are NOT covered (see `PrivateObservation`).
#
# The physical-state covariance under u_k = -P_k x̂_k - α_k is NOT the posterior
# estimation covariance: x̂ is random and feeds back into x. The recursion below
# tracks the joint law of (x_k, e_k = x_k - x̂_k) so that both are reported
# separately.
#
# Convention: the prior is x₁ ~ N(x̄₁, Σ₀); the first measurement y₁ is assimilated
# BEFORE the first control, and y_{k+1} after each transition k, so u_k is always a
# function of the estimate given y₁,…,y_k.
# ============================================================================

"""
    linear_observation(H, V) -> SharedObservation

Common linear measurement `y = H x + v`, `v ~ N(0, V)`.
"""
function linear_observation(H::AbstractMatrix, V::AbstractMatrix)
    Hm = Matrix{Float64}(H)
    SharedObservation((x, v, p, t) -> Hm * x .+ v, size(Hm, 1), Matrix{Float64}(V))
end

"""
    observation_matrix(obs::SharedObservation, x, p, t) -> Matrix

Jacobian `∂h/∂x` of the observation model at `x` (with zero measurement noise).
"""
observation_matrix(obs::SharedObservation, x::AbstractVector, p, t) =
    ForwardDiff.jacobian(z -> obs.observation_model(z, zeros(eltype(z), obs.obs_dim), p, t), Vector{Float64}(x))

"""
    is_affine_observation(obs, n; p=nothing, t=1, rtol=1e-8) -> Bool

Whether `h(x, v)` is affine in `(x, v)` with unit noise gain (`h = Hx + c + v`),
checked at random points. Required for the exact common-information LQ results.
"""
function is_affine_observation(obs::SharedObservation, n::Int; p = nothing, t = 1, rtol = 1e-8)
    rng = Random.MersenneTwister(0)
    h = obs.observation_model
    z = zeros(obs.obs_dim)
    c = h(zeros(n), z, p, t)
    H = observation_matrix(obs, zeros(n), p, t)
    for _ in 1:3
        x = randn(rng, n); v = randn(rng, obs.obs_dim)
        isapprox(h(x, v, p, t), H * x + c + v; rtol = rtol, atol = 1e-10) || return false
    end
    return true
end

"Kalman gain `K = Σ⁻ Hᵀ (H Σ⁻ Hᵀ + V)⁻¹` (solved, never inverted)."
kalman_gain(Σ⁻::AbstractMatrix, H::AbstractMatrix, V::AbstractMatrix) =
    Matrix((Σ⁻ * H') / Symmetric(H * Σ⁻ * H' + V))

"Joseph-form posterior covariance `(I-KH)Σ⁻(I-KH)ᵀ + KVKᵀ`, symmetrised."
function kalman_joseph(Σ⁻::AbstractMatrix, K::AbstractMatrix, H::AbstractMatrix, V::AbstractMatrix)
    IKH = I - K * H
    return Matrix(Symmetric(IKH * Σ⁻ * IKH' + K * V * K'))
end

"""
    common_information_moments(A, B, P, W, H, V, Σ0) -> NamedTuple

Second moments for `u_k = -P_k x̂_k - α_k` under common observations. `A[k]` (n×n),
`B[k]` (n×m, joint input), `P[k]` (m×n, joint gain), `W[k]` (n×n), `H[j]`
(measurement matrix of `y_j`, `j = 1…N+1`; `H[1]` is assimilated before the first
control), `V` (noise covariance). Transitions are `k = 1…N`; state moments have
`N+1` entries.

Fields: `state_covariances` Cov(x_k); `estimate_covariances` Cov(x̂_k);
`posterior_covariances` Cov(e_k) (the filter's Σ_{k|k}); `state_estimate_covariances`
Cov(x_k, x̂_k); `control_covariances` Cov(u_k) and
`state_control_covariances` Cov(x_k, u_k) for k = 1…N; `initial_kalman_gain` (for y₁)
and `kalman_gains[k]` (gain used to assimilate y_{k+1} after transition k).
"""
function common_information_moments(A, B, P, W, H, V, Σ0)
    N = length(A); n = size(A[1], 1)
    Σ0m = Matrix{Float64}(Σ0)
    K0 = kalman_gain(Σ0m, H[1], V)                # assimilate y₁ into the prior
    IKH0 = I - K0 * H[1]
    Σe = kalman_joseph(Σ0m, K0, H[1], V)          # Cov(e₁)
    C0 = Σ0m * IKH0'                              # Cov(x₁, e₁)
    Σz = [Σ0m C0; C0' Σe]                         # Cov((x_k, e_k))
    out = (state_covariances = Matrix{Float64}[], estimate_covariances = Matrix{Float64}[],
           posterior_covariances = Matrix{Float64}[], state_estimate_covariances = Matrix{Float64}[],
           control_covariances = Matrix{Float64}[], state_control_covariances = Matrix{Float64}[],
           kalman_gains = Matrix{Float64}[], initial_kalman_gain = K0)
    record!(Σz) = begin
        Σx = Σz[1:n, 1:n]; C = Σz[1:n, n+1:end]; Σee = Σz[n+1:end, n+1:end]
        push!(out.state_covariances, Matrix(Symmetric(Σx)))
        push!(out.posterior_covariances, Matrix(Symmetric(Σee)))
        push!(out.estimate_covariances, Matrix(Symmetric(Σx - C - C' + Σee)))
        push!(out.state_estimate_covariances, Σx - C)
    end
    record!(Σz)
    for k in 1:N
        Σx̂ = out.estimate_covariances[k]; Σxx̂ = out.state_estimate_covariances[k]
        push!(out.control_covariances, Matrix(Symmetric(P[k] * Σx̂ * P[k]')))
        push!(out.state_control_covariances, -Σxx̂ * P[k]')
        Σ⁻ = Matrix(Symmetric(A[k] * Σz[n+1:end, n+1:end] * A[k]' + W[k]))
        K = kalman_gain(Σ⁻, H[k+1], V)
        push!(out.kalman_gains, K)
        IKH = I - K * H[k+1]
        F = A[k] - B[k] * P[k]
        M = [F B[k]*P[k]; zeros(n, n) IKH*A[k]]
        G = [Matrix(1.0I, n, n) zeros(n, size(K, 2)); IKH -K]
        Σξ = [W[k] zeros(n, size(V, 1)); zeros(size(V, 1), n) V]
        Σz = Matrix(Symmetric(M * Σz * M' + G * Σξ * G'))
        record!(Σz)
    end
    return out
end
