# ============================================================================
# Time Horizon Specifications
# ============================================================================

"""
    TimeHorizon{T}

Abstract base type for time horizon specifications.
"""
abstract type TimeHorizon{T} end

"""
    ContinuousTime{T} <: TimeHorizon{T}

Continuous-time horizon for differential games.

# Fields
- `tf::T` : Final time
- `integrator_type::Symbol` : ODE integration scheme (:rk4, :tsit5, :euler, etc.)

# Notes
Solver performs ODE integration to propagate dynamics.
Choice of integrator affects accuracy and computational cost.
"""
struct ContinuousTime{T} <: TimeHorizon{T}
    tf::T
    integrator_type::Symbol
    
    function ContinuousTime(tf::T; integrator_type::Symbol = :rk4) where {T}
        @assert tf > 0 "Time horizon must be positive"
        @assert integrator_type in (:euler, :rk4, :tsit5, :radau) "Unknown integrator type"
        new{T}(tf, integrator_type)
    end
end

"""
    DiscreteTime{T} <: TimeHorizon{T}

Discrete-time horizon with fixed time step.

# Fields
- `tf::T` : Final time
- `dt::T` : Time step
- `N::Int` : Number of time steps (computed as ceil(tf/dt))

# Notes
Standard for direct transcription methods.
State and control discretized at times [0, dt, 2dt, ..., N*dt].
"""
struct DiscreteTime{T} <: TimeHorizon{T}
    tf::T
    dt::T
    N::Int
    
    function DiscreteTime(tf::T, dt::T) where {T}
        @assert tf > 0 "Time horizon must be positive"
        @assert dt > 0 "Time step must be positive"
        @assert dt < tf "Time step must be less than time horizon"
        
        N = Int(ceil(tf / dt))
        new{T}(tf, dt, N)
    end
end