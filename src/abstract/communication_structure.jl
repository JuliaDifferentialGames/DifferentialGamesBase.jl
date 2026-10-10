# ============================================================================
# abstract/communication_structure.jl
#
# AbstractCommunicationTopology — which other players' information a given
# player's controller may use when computing feedback gains.
#
# Deliberately orthogonal to AbstractInformationStructure (which describes
# what a player observes of the *state*) and to CommunicationCost/
# CommunicationConstraint (which are physical/strategic proximity
# cost/constraint on the trajectories, not a statement about who can see whose
# controller output). Structs only for now — no solver consumes this yet;
# receding-horizon/tube solvers are the natural future consumers.
# ============================================================================

"""
    AbstractCommunicationTopology

Abstract type describing which players' strategies/states a given player's
controller can use, distinct from `AbstractInformationStructure` (state
observation) and `CommunicationCost`/`CommunicationConstraint` (physical
proximity cost/constraint).
"""
abstract type AbstractCommunicationTopology end

"""
    FullCommunication <: AbstractCommunicationTopology

Every player's controller may use every other player's state/strategy.
"""
struct FullCommunication <: AbstractCommunicationTopology end

"""
    NoCommunication <: AbstractCommunicationTopology

Each player's controller may use only its own state/strategy.
"""
struct NoCommunication <: AbstractCommunicationTopology end

"""
    GraphCommunication <: AbstractCommunicationTopology

Player `i` may use player `j`'s information iff `adjacency[i, j]`.

# Fields
- `adjacency::SparseMatrixCSC{Bool}`
"""
struct GraphCommunication <: AbstractCommunicationTopology
    adjacency::SparseMatrixCSC{Bool}
end

"""
    DelayedCommunication{S<:AbstractCommunicationTopology} <: AbstractCommunicationTopology

Wraps an inner topology `inner`, delaying every exchange by `delay_steps` discrete steps.
"""
struct DelayedCommunication{S<:AbstractCommunicationTopology} <: AbstractCommunicationTopology
    inner::S
    delay_steps::Int

    function DelayedCommunication{S}(inner::S, delay_steps::Int) where {S<:AbstractCommunicationTopology}
        @assert delay_steps >= 0 "delay_steps must be non-negative"
        new{S}(inner, delay_steps)
    end
end

DelayedCommunication(inner::S, delay_steps::Int) where {S<:AbstractCommunicationTopology} =
    DelayedCommunication{S}(inner, delay_steps)

"""
    BandwidthLimitedCommunication{S<:AbstractCommunicationTopology} <: AbstractCommunicationTopology

Wraps an inner topology `inner`, capping each exchange to `max_bits_per_step` bits.
"""
struct BandwidthLimitedCommunication{S<:AbstractCommunicationTopology} <: AbstractCommunicationTopology
    inner::S
    max_bits_per_step::Int

    function BandwidthLimitedCommunication{S}(
        inner::S, max_bits_per_step::Int
    ) where {S<:AbstractCommunicationTopology}
        @assert max_bits_per_step > 0 "max_bits_per_step must be positive"
        new{S}(inner, max_bits_per_step)
    end
end

BandwidthLimitedCommunication(inner::S, max_bits_per_step::Int) where {S<:AbstractCommunicationTopology} =
    BandwidthLimitedCommunication{S}(inner, max_bits_per_step)
