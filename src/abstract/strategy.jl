using LinearAlgebra

# ============================================================================
# AbstractStrategy hierarchy
#
# Two distinct mathematical objects:
#
#   OpenLoopStrategy  : γᵢ(t) = ûᵢ(t)
#     - Function of time only; player commits at t=0
#     - Corresponds to OpenLoopInformation
#     - FALCON output; ALGAMES open-loop mode
#     - Representation: sequence {ûᵢ(k)}_{k=1}^N per player
#
#   FeedbackStrategy  : γᵢ(t, x) = ûᵢ(t) - Pᵢ(t)·δx(t) - η·αᵢ(t)
#     - Function of time and state; player responds to current state
#     - Corresponds to PerfectStateInformation
#     - FNELQ output; iLQGames outer-loop output
#     - Representation: gains {Pᵢ(k)}, feedforward {αᵢ(k)},
#       nominal trajectory (x̂, {ûᵢ})
#
# These are not interchangeable: they correspond to different equilibrium
# concepts (open-loop Nash vs feedback Nash) and different information
# structures. GNEPSolution carries an AbstractStrategy field so that the
# solution type is agnostic to which was computed.
#
# apply_strategy dispatches on the concrete type:
#   OpenLoopStrategy → reads ûᵢ(k) directly, ignores current x
#   FeedbackStrategy → applies affine correction to x
#
# Both are ForwardDiff-compatible: apply_strategy accepts dual numbers in x
# (needed for sensitivity analysis and iLQGames gradient computation).
# ============================================================================

"""
    AbstractStrategy{T}

Root abstract type for player strategy representations.

# Subtype Tree
```
AbstractStrategy{T}
├── OpenLoopStrategy{T}   — {ûᵢ(k)} sequences; open-loop Nash
└── FeedbackStrategy{T}   — {Pᵢ(k), αᵢ(k), x̂, ûᵢ}; feedback Nash
```

# Interface
Every subtype must implement:
- `apply_strategy(s, x, k; η) -> Vector` — joint control at timestep k
- `n_steps(s) -> Int`
- `n_players(s) -> Int`
- `control_dims(s) -> Vector{Int}`
- `get_times(s) -> Vector{T}`
- `get_nominal_control(s, i, k) -> Vector{T}` — ûᵢ(k)
"""
abstract type AbstractStrategy{T} end

# ============================================================================
# OpenLoopStrategy
# ============================================================================

"""
    OpenLoopStrategy{T} <: AbstractStrategy{T}

Open-loop strategy: player i plays `ûᵢ(k)` regardless of current state.

Corresponds to `OpenLoopInformation`. This is the equilibrium concept targeted
by FALCON (open-loop Nash of a PD-GNEP) and open-loop iterative best response.

# Fields
- `controls::Vector{Matrix{T}}` : `controls[i]` is (mᵢ × N); `controls[i][:, k]` = ûᵢ(k)
- `control_dims::Vector{Int}`   : `[m₁, …, mₙ]`
- `times::Vector{T}`            : Length N+1

# Construction
```julia
# From raw control matrices
s = OpenLoopStrategy([U1, U2], [m1, m2], times)

# Zero strategy (all controls zero)
s = zero_open_loop_strategy(n_players, control_dims, N, times)
```
"""
struct OpenLoopStrategy{T} <: AbstractStrategy{T}
    controls::Vector{Matrix{T}}
    control_dims::Vector{Int}
    times::Vector{T}
    n_players::Int

    function OpenLoopStrategy(
        controls::Vector{Matrix{T}},
        control_dims::Vector{Int},
        times::Vector{T}
    ) where {T}
        np = length(controls)
        N  = length(times) - 1
        @assert N > 0 "times must have at least 2 entries"
        @assert length(control_dims) == np "control_dims length must equal n_players"
        for i in 1:np
            @assert(size(controls[i], 1) == control_dims[i],
                "controls[$i] must have $(control_dims[i]) rows")
            @assert(size(controls[i], 2) == N,
                "controls[$i] must have $N columns")
        end
        new{T}(controls, control_dims, times, np)
    end
end

"""
    zero_open_loop_strategy(n_players, control_dims, N, times) -> OpenLoopStrategy{T}

Construct a zero open-loop strategy (all controls zero).
"""
function zero_open_loop_strategy(
    n_players::Int,
    control_dims::Vector{Int},
    N::Int,
    times::Vector{T} = collect(range(zero(T), one(T), length=N+1))
) where {T <: AbstractFloat}
    controls = [zeros(T, control_dims[i], N) for i in 1:n_players]
    OpenLoopStrategy(controls, control_dims, times)
end

# ============================================================================
# FeedbackStrategy
# ============================================================================

"""
    FeedbackStrategy{T} <: AbstractStrategy{T}

Affine state-feedback strategy:
  `uᵢ(k) = ûᵢ(k) - Pᵢ(k)·(x(k) - x̂(k)) - η·αᵢ(k)`

Corresponds to `PerfectStateInformation`. This is the equilibrium concept
targeted by FNELQ (finite-horizon feedback Nash) and iLQGames.

The nominal trajectory (x̂, {ûᵢ}) serves two roles:
1. It is the linearization point used to compute gains Pᵢ(k)
2. It is the open-loop baseline that the feedback correction perturbs

The step-size parameter η ∈ [0,1] interpolates:
- η = 0 : pure open-loop (recovers ûᵢ(k) - Pᵢ(k)·δx(k), no feedforward)
- η = 1 : full LQ feedback solution

# Fields
- `gains::Vector{Vector{Matrix{T}}}` : `gains[i][k]` = Pᵢ(k), shape (mᵢ × n)
- `feedforward::Vector{Vector{Vector{T}}}` : `feedforward[i][k]` = αᵢ(k), shape (mᵢ,)
- `nominal_states::Matrix{T}` : x̂, shape (n × N+1)
- `nominal_controls::Vector{Matrix{T}}` : ûᵢ, shape (mᵢ × N) per player
- `control_dims::Vector{Int}` : `[m₁, …, mₙ]`
- `times::Vector{T}` : Length N+1
- `n_players::Int`
"""
struct FeedbackStrategy{T} <: AbstractStrategy{T}
    gains::Vector{Vector{Matrix{T}}}
    feedforward::Vector{Vector{Vector{T}}}
    nominal_states::Matrix{T}
    nominal_controls::Vector{Matrix{T}}
    control_dims::Vector{Int}
    times::Vector{T}
    n_players::Int

    function FeedbackStrategy(
        gains::Vector{Vector{Matrix{T}}},
        feedforward::Vector{Vector{Vector{T}}},
        nominal_states::Matrix{T},
        nominal_controls::Vector{Matrix{T}},
        control_dims::Vector{Int},
        times::Vector{T}
    ) where {T}
        np = length(gains)
        N  = length(times) - 1
        n  = size(nominal_states, 1)

        @assert N > 0 "times must have at least 2 entries"
        @assert length(feedforward)      == np "feedforward length must equal n_players"
        @assert length(nominal_controls) == np "nominal_controls length must equal n_players"
        @assert length(control_dims)     == np "control_dims length must equal n_players"
        @assert size(nominal_states, 2)  == N + 1 "nominal_states must have N+1 columns"

        for i in 1:np
            @assert length(gains[i])       == N "gains[$i] must have $N entries"
            @assert length(feedforward[i]) == N "feedforward[$i] must have $N entries"
            @assert(size(nominal_controls[i], 2) == N,
                "nominal_controls[$i] must have $N columns")
            @assert(size(nominal_controls[i], 1) == control_dims[i],
                "nominal_controls[$i] must have $(control_dims[i]) rows")
            for k in 1:N
                @assert(size(gains[i][k]) == (control_dims[i], n),
                    "gains[$i][$k] must be ($(control_dims[i]) × $n)")
                @assert(length(feedforward[i][k]) == control_dims[i],
                    "feedforward[$i][$k] must have length $(control_dims[i])")
            end
        end

        new{T}(gains, feedforward, nominal_states, nominal_controls,
               control_dims, times, np)
    end
end

"""
    zero_feedback_strategy(n_players, n, control_dims, N, times) -> FeedbackStrategy{T}

Construct a zero feedback strategy (zero gains, zero feedforward, zero nominal).
Used as the initial iterate in iLQGames when no warmstart is provided.
"""
function zero_feedback_strategy(
    n_players::Int,
    n::Int,
    control_dims::Vector{Int},
    N::Int,
    times::Vector{T} = collect(range(zero(T), one(T), length=N+1))
) where {T <: AbstractFloat}
    gains    = [[zeros(T, control_dims[i], n) for _ in 1:N] for i in 1:n_players]
    ff       = [[zeros(T, control_dims[i])    for _ in 1:N] for i in 1:n_players]
    x_nom    = zeros(T, n, N+1)
    u_nom    = [zeros(T, control_dims[i], N)  for i in 1:n_players]
    FeedbackStrategy(gains, ff, x_nom, u_nom, control_dims, times)
end

# ============================================================================
# AbstractStrategy interface — implemented for both subtypes
# ============================================================================

"""
    n_steps(s::AbstractStrategy) -> Int
"""
n_steps(s::AbstractStrategy) = length(s.times) - 1

"""
    n_players(s::AbstractStrategy) -> Int
"""
n_players(s::AbstractStrategy) = s.n_players

"""
    get_times(s::AbstractStrategy) -> Vector{T}
"""
get_times(s::AbstractStrategy) = s.times

"""
    get_control_dims(s::AbstractStrategy) -> Vector{Int}
"""
get_control_dims(s::AbstractStrategy) = s.control_dims

"""
    control_offsets(s::AbstractStrategy) -> Vector{Int}

Zero-based start indices of each player's controls in the joint vector.
"""
control_offsets(s::AbstractStrategy) = [0; cumsum(s.control_dims)[1:end-1]]

"""
    get_nominal_control(s::AbstractStrategy, i::Int, k::Int) -> Vector{T}

Return ûᵢ(k) for player i at timestep k.
"""
get_nominal_control(s::OpenLoopStrategy, i::Int, k::Int) = s.controls[i][:, k]
get_nominal_control(s::FeedbackStrategy, i::Int, k::Int) = s.nominal_controls[i][:, k]

# FeedbackStrategy-specific accessors
"""
    get_gain(s::FeedbackStrategy{T}, i, k) -> Matrix{T}
"""
get_gain(s::FeedbackStrategy, i::Int, k::Int) = s.gains[i][k]

"""
    get_feedforward(s::FeedbackStrategy{T}, i, k) -> Vector{T}
"""
get_feedforward(s::FeedbackStrategy, i::Int, k::Int) = s.feedforward[i][k]

"""
    get_nominal_state(s::FeedbackStrategy{T}, k) -> Vector{T}
"""
get_nominal_state(s::FeedbackStrategy, k::Int) = s.nominal_states[:, k]

"""
    state_dim(s::FeedbackStrategy) -> Int
"""
state_dim(s::FeedbackStrategy) = size(s.nominal_states, 1)

# ============================================================================
# apply_strategy — compute joint control at timestep k
# ============================================================================

"""
    apply_strategy(s::AbstractStrategy{T}, x, k; η=1.0) -> Vector{T}

Compute the joint control vector [u₁; u₂; …; uₙ] at timestep k.

# OpenLoopStrategy
  Returns ûᵢ(k) concatenated. The state `x` is ignored (open-loop).
  The `η` parameter has no effect.

# FeedbackStrategy
  Returns `uᵢ(k) = ûᵢ(k) - Pᵢ(k)·(x - x̂(k)) - η·αᵢ(k)` concatenated.
  - η = 1 : full feedback law
  - η = 0 : open-loop nominal + proportional feedback, no feedforward

Both implementations accept ForwardDiff dual numbers in `x`.
"""
function apply_strategy(
    s::OpenLoopStrategy{T},
    x::AbstractVector,
    k::Int;
    η::Real = one(T)
) where {T}
    offs    = control_offsets(s)
    m_total = sum(s.control_dims)
    u       = Vector{eltype(x)}(undef, m_total)
    for i in 1:s.n_players
        rng      = offs[i]+1 : offs[i]+s.control_dims[i]
        u[rng]   = s.controls[i][:, k]
    end
    return u
end

function apply_strategy(
    s::FeedbackStrategy{T},
    x::AbstractVector,
    k::Int;
    η::Real = one(T)
) where {T}
    δx      = x - s.nominal_states[:, k]
    offs    = control_offsets(s)
    m_total = sum(s.control_dims)
    u       = Vector{eltype(x)}(undef, m_total)
    for i in 1:s.n_players
        rng    = offs[i]+1 : offs[i]+s.control_dims[i]
        û_ik   = s.nominal_controls[i][:, k]
        P_ik   = s.gains[i][k]
        α_ik   = s.feedforward[i][k]
        u[rng] = û_ik - P_ik * δx - η .* α_ik
    end
    return u
end

# ============================================================================
# Conversion: FeedbackStrategy → OpenLoopStrategy
#
# Useful for warm-starting FALCON with iLQGames output, or for extracting
# the open-loop component of a feedback solution at a fixed initial state.
# ============================================================================

"""
    to_open_loop(s::FeedbackStrategy{T}, x0::AbstractVector; η=1.0) -> OpenLoopStrategy{T}

Evaluate the feedback strategy from initial state `x0` to produce an
open-loop strategy. The resulting controls are the sequence of joint
controls that the feedback law would generate from `x0` under the
*nominal* dynamics (i.e., using the stored nominal states for δx computation).

Note: this does not re-simulate the dynamics. It simply evaluates
`uᵢ(k) = ûᵢ(k) - Pᵢ(k)·(x̂(k) - x̂(k)) - η·αᵢ(k)` which for any
x = x̂ gives `uᵢ(k) = ûᵢ(k) - η·αᵢ(k)`. For genuine open-loop extraction
from a specific x₀ ≠ x̂(0), use `rollout_strategy` instead.
"""
function to_open_loop(s::FeedbackStrategy{T}) where {T}
    N  = n_steps(s)
    np = s.n_players
    # At nominal: δx = 0, so u = û - η·α. With η=1: u = û - α.
    # We store the full nominal controls as the open-loop sequence.
    OpenLoopStrategy(
        [copy(s.nominal_controls[i]) for i in 1:np],
        copy(s.control_dims),
        copy(s.times)
    )
end

# ============================================================================
# Display
# ============================================================================

function Base.show(io::IO, s::OpenLoopStrategy{T}) where {T}
    print(io, "OpenLoopStrategy{$T} [$(s.n_players) players, N=$(n_steps(s)) steps, ",
          "m=$(s.control_dims)]")
end

function Base.show(io::IO, ::MIME"text/plain", s::OpenLoopStrategy{T}) where {T}
    println(io, "OpenLoopStrategy{$T}")
    println(io, "  Players      : ", s.n_players)
    println(io, "  Timesteps N  : ", n_steps(s))
    println(io, "  Control dims : ", s.control_dims)
    println(io, "  Time range   : [$(s.times[1]), $(s.times[end])]")
end

function Base.show(io::IO, s::FeedbackStrategy{T}) where {T}
    print(io, "FeedbackStrategy{$T} [$(s.n_players) players, N=$(n_steps(s)) steps, ",
          "n=$(state_dim(s)), m=$(s.control_dims)]")
end

function Base.show(io::IO, ::MIME"text/plain", s::FeedbackStrategy{T}) where {T}
    println(io, "FeedbackStrategy{$T}")
    println(io, "  Players      : ", s.n_players)
    println(io, "  Timesteps N  : ", n_steps(s))
    println(io, "  State dim n  : ", state_dim(s))
    println(io, "  Control dims : ", s.control_dims)
    println(io, "  Time range   : [$(s.times[1]), $(s.times[end])]")
end