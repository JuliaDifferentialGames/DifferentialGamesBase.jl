# ============================================================================
# problems/stochastic_game.jl
#
# Concrete AbstractStochasticGame subtype, now expressed in terms of the
# general ItoSDEDynamics/StochasticDiscreteApproximation pipeline
# (dynamics/stochastic_dynamics.jl, dynamics/stochastic_discretization.jl)
# rather than a standalone additive-noise vocabulary. `ProcessNoiseModel` is
# kept as a backward-compatible convenience constructor: it builds the
# `ItoSDEDynamics` that reproduces its old semantics exactly (additive
# Gaussian noise on the joint state), so existing call sites (and the
# SFNELQ stochastic-LQ Nash solver in DifferentialGamesBaseSolvers.jl, which
# only ever consumed additive/constant-diffusion noise) keep working.
#
# Still wraps rather than duplicates — dynamics, objectives, and constraints
# all still come from `nominal_game`.
# ============================================================================

"""
    ProcessNoiseModel{T}

Backward-compatible data holder for additive Gaussian process noise on the
joint state: `x_{k+1} = f(x_k, u_k) + w_k`, `w_k ~ N(0, Σ)`. On its own it
carries no drift information; `StochasticGameProblem(nominal_game,
::ProcessNoiseModel)` is what turns it into the `ItoSDEDynamics` that
reproduces this exact semantics, using `nominal_game.dynamics` (must be
`LinearDynamics`) as the drift.

Distinct from `SharedObservation`/`PrivateObservation`'s noise fields, which
model *measurement* noise (what a player observes about the state), not
*dynamics* noise (how the true state evolves) — see the design note in
`abstract/information_structure.jl`.

# Fields
- `Σ::Union{Matrix{T}, Vector{Matrix{T}}}` : Process noise covariance, either
  constant (LTI, supported) or one matrix per time step (LTV — not yet
  supported by the `ItoSDEDynamics` conversion; raises an error, see
  docs/src/api/stochastic.md Phase roadmap).
"""
struct ProcessNoiseModel{T}
    Σ::Union{Matrix{T}, Vector{Matrix{T}}}

    function ProcessNoiseModel{T}(Σ::Matrix{T}) where {T}
        _check_psd_covariance(Σ, "Process noise covariance")
        new{T}(Σ)
    end

    function ProcessNoiseModel{T}(Σ::Vector{Matrix{T}}) where {T}
        @assert !isempty(Σ) "Process noise covariance sequence must be non-empty"
        for Σk in Σ
            _check_psd_covariance(Σk, "Each process noise covariance")
        end
        new{T}(Σ)
    end
end

# Covariances may be singular (zero or rank-deficient disturbances are legitimate),
# so only symmetry and positive *semi*definiteness are required.
function _check_psd_covariance(Σ::AbstractMatrix{T}, what::AbstractString) where {T}
    @assert issymmetric(Σ) "$what must be symmetric"
    ev = eigvals(Symmetric(Σ))
    tol = sqrt(eps(T)) * max(one(T), maximum(abs, ev))
    @assert all(>=(-tol), ev) "$what must be positive semidefinite"
    return nothing
end

# Factor Σ = G Gᵀ for possibly singular PSD Σ (Cholesky fails on rank-deficient input).
function _psd_factor(Σ::AbstractMatrix{T}) where {T}
    F = eigen(Symmetric(Σ))
    return Matrix(F.vectors * Diagonal(sqrt.(max.(F.values, zero(T)))))
end

ProcessNoiseModel(Σ::Matrix{T}) where {T} = ProcessNoiseModel{T}(Σ)
ProcessNoiseModel(Σ::Vector{Matrix{T}}) where {T} = ProcessNoiseModel{T}(Σ)

is_ltv(model::ProcessNoiseModel) = model.Σ isa Vector
covariance(model::ProcessNoiseModel, k::Int) = is_ltv(model) ? model.Σ[k] : model.Σ

"""
    _ito_sde_from_process_noise(drift, pnm::ProcessNoiseModel) -> ItoSDEDynamics

Build the additive-noise `ItoSDEDynamics` equivalent to a (legacy)
`ProcessNoiseModel`.
"""
function _ito_sde_from_process_noise(drift::DynamicsSpec{T}, pnm::ProcessNoiseModel{T}) where {T}
    is_ltv(pnm) && error("LTV ProcessNoiseModel is not yet supported by the ItoSDEDynamics conversion — see docs/src/api/stochastic.md Phase roadmap.")
    Σ = pnm.Σ
    G = _psd_factor(Σ)
    ItoSDEDynamics(drift, constant_diffusion(G), size(G, 2))
end

"""
    StochasticGameProblem{T} <: AbstractStochasticGame{T}

A deterministic `nominal_game` with stochastic dynamics layered on top of
its drift. Nothing about the nominal game's dynamics/objectives/constraints
is duplicated; `stochastic_dynamics` wraps `nominal_game.dynamics` as the
drift of an `ItoSDEDynamics` (or, for the Phase 1 POMDPs.jl adapter, is a
`MarkovKernelDynamics`).

# Fields
- `nominal_game::AbstractDeterministicGame{T}`
- `stochastic_dynamics::AbstractStochasticDynamics{T}`
"""
struct StochasticGameProblem{T} <: AbstractStochasticGame{T}
    nominal_game::AbstractDeterministicGame{T}
    stochastic_dynamics::AbstractStochasticDynamics{T}
    initial_covariance::Matrix{T}
    process_covariances::Union{Nothing, Vector{Matrix{T}}}

    function StochasticGameProblem{T}(
        nominal_game::AbstractDeterministicGame{T},
        stochastic_dynamics::AbstractStochasticDynamics{T},
        initial_covariance::Matrix{T},
        process_covariances::Union{Nothing, Vector{Matrix{T}}} = nothing
    ) where {T}
        n = total_state_dim(nominal_game.dynamics)
        @assert total_state_dim(stochastic_dynamics) == n "stochastic_dynamics state dim must match nominal_game"
        @assert size(initial_covariance) == (n, n) "initial_covariance must be n×n"
        _check_psd_covariance(initial_covariance, "Initial covariance")
        if process_covariances !== nothing
            @assert all(W -> size(W) == (n, n), process_covariances) "each process covariance must be n×n"
        end
        new{T}(nominal_game, stochastic_dynamics, initial_covariance, process_covariances)
    end
end

function StochasticGameProblem{T}(g::AbstractDeterministicGame{T}, sd::AbstractStochasticDynamics{T}) where {T}
    n = total_state_dim(g.dynamics)
    StochasticGameProblem{T}(g, sd, zeros(T, n, n), nothing)
end

StochasticGameProblem(
    nominal_game::AbstractDeterministicGame{T},
    stochastic_dynamics::AbstractStochasticDynamics{T};
    initial_covariance::Union{Nothing, AbstractMatrix} = nothing
) where {T} = StochasticGameProblem{T}(nominal_game, stochastic_dynamics,
    initial_covariance === nothing ?
        zeros(T, total_state_dim(nominal_game.dynamics), total_state_dim(nominal_game.dynamics)) :
        Matrix{T}(initial_covariance), nothing)

"""
    StochasticGameProblem(nominal_game, pnm::ProcessNoiseModel; initial_covariance=nothing)

Backward-compatible constructor. Converts `pnm` into the `ItoSDEDynamics`
reproducing its additive-noise semantics, using `nominal_game.dynamics` (must be
`LinearDynamics`) as the drift. A per-step (LTV) `pnm` is stored in
`process_covariances` (step `k` uses `process_covariances[k]`); the SDE then
carries `pnm.Σ[1]` only as a placeholder. `initial_covariance` is the
covariance of the uncertain initial state `x₀ ~ N(x0, Σ₀)` (default zero).
"""
function StochasticGameProblem(nominal_game::AbstractDeterministicGame{T}, pnm::ProcessNoiseModel{T};
        initial_covariance::Union{Nothing, AbstractMatrix} = nothing) where {T}
    n = total_state_dim(nominal_game.dynamics)
    Σ0 = initial_covariance === nothing ? zeros(T, n, n) : Matrix{T}(initial_covariance)
    if is_ltv(pnm)
        sde = _ito_sde_from_process_noise(nominal_game.dynamics, ProcessNoiseModel(pnm.Σ[1]))
        return StochasticGameProblem{T}(nominal_game, sde, Σ0, Vector{Matrix{T}}(pnm.Σ))
    end
    sde = _ito_sde_from_process_noise(nominal_game.dynamics, pnm)
    StochasticGameProblem{T}(nominal_game, sde, Σ0, nothing)
end

"""
    DiscreteGaussianNoise(W)

Discrete-time additive Gaussian process noise `w_k ~ N(0, W_k)`; `W` is one PSD
matrix (LTI) or a vector of them (one per step). Alias of [`ProcessNoiseModel`](@ref).
"""
const DiscreteGaussianNoise = ProcessNoiseModel

n_players(g::StochasticGameProblem) = n_players(g.nominal_game)
time_horizon(g::StochasticGameProblem) = time_horizon(g.nominal_game)

"""
    get_W(g::StochasticGameProblem) -> Matrix

Noise covariance `GGᵀ` for the wrapped `ItoSDEDynamics`'s diffusion, read
off `.diffusion` rather than a bare `ProcessNoiseModel`. Consumed by
SFNELQ (`DifferentialGamesBaseSolvers.jl`).
"""
function get_W(g::StochasticGameProblem, k::Int)
    g.process_covariances === nothing || return g.process_covariances[k]
    return get_W(g)
end

function get_W(g::StochasticGameProblem)
    g.process_covariances === nothing || error("get_W(g) is ambiguous for time-varying noise; use get_W(g, k)")
    sde = g.stochastic_dynamics
    @assert sde isa ItoSDEDynamics "get_W requires ItoSDEDynamics stochastic dynamics"
    G = diffusion_matrix(sde, g.nominal_game.initial_state,
        zeros(total_control_dim(sde)), nothing, 1)
    return G * G'
end

"""
    has_multiplicative_noise(g::StochasticGameProblem) -> Bool

`true` unless the diffusion is provably state/control-independent
(`ConstantDiffusion`, i.e. additive noise — the certainty-equivalent case).
"""
has_multiplicative_noise(g::StochasticGameProblem) = !is_constant_diffusion(g.stochastic_dynamics)

"""
    is_certainty_equivalent(g::StochasticGameProblem) -> Bool

`true` iff the noise is additive (constant diffusion) — the case in which
the stochastic-LQ Nash feedback gains coincide with the deterministic LQ
Nash gains (only the value-function constant shifts).
"""
is_certainty_equivalent(g::StochasticGameProblem) = !has_multiplicative_noise(g)

function Base.show(io::IO, g::StochasticGameProblem{T}) where {T}
    print(io, "StochasticGameProblem{$T} wrapping ", g.nominal_game,
          " [", typeof(g.stochastic_dynamics).name.name, "]")
end
