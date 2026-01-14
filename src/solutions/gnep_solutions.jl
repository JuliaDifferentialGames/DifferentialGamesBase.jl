"""
    Trajectory{T}

Represents a trajectory (state and control) over time.

# Fields
- `states::Matrix{T}` : State trajectory, size (n, K) where K is number of time steps
- `controls::Matrix{T}` : Control trajectory, size (m, K)
- `times::Vector{T}` : Time points
- `cost::T` : Total cost for this trajectory
"""
struct Trajectory{T}
    states::Matrix{T}      # (n × K)
    controls::Matrix{T}    # (m × K)
    times::Vector{T}       # (K,)
    cost::T
end

"""
    GNEPSolution{T}

General solution structure for PD-GNEP problems.

# Fields
- `trajectories::Vector{Trajectory{T}}` : Per-player trajectories
- `costs::Vector{T}` : Final costs for each player
- `equilibrium_type::Symbol` : Type of equilibrium (:Nash, :GeneralizedNash, etc.)
- `converged::Bool` : Whether the solver converged
- `iterations::Int` : Number of iterations taken
- `solve_time::Float64` : Wall-clock time in seconds
- `metadata::Dict{Symbol, Any}` : Additional solver-specific information

# Notes
For PD-GNEP with separable dynamics, each player's trajectory is computed independently
given the equilibrium control strategies.
"""
struct GNEPSolution{T}
    trajectories::Vector{Trajectory{T}}  # One per player
    costs::Vector{T}                     # Cost for each player
    equilibrium_type::Symbol             # :Nash, :GeneralizedNash, etc.
    converged::Bool
    iterations::Int
    solve_time::Float64
    metadata::Dict{Symbol, Any}          # Solver-specific data
    
    function GNEPSolution{T}(
        trajectories::Vector{Trajectory{T}},
        costs::Vector{T},
        equilibrium_type::Symbol = :Nash;
        converged::Bool = true,
        iterations::Int = 0,
        solve_time::Float64 = 0.0,
        metadata::Dict{Symbol, Any} = Dict{Symbol, Any}()
    ) where T
        @assert length(trajectories) == length(costs) "Mismatch in number of players"
        new{T}(trajectories, costs, equilibrium_type, converged, iterations, 
               solve_time, metadata)
    end
end

# Convenience constructor with keyword arguments
function GNEPSolution(
    trajectories::Vector{Trajectory{T}},
    costs::Vector{T};
    equilibrium_type::Symbol = :Nash,
    converged::Bool = true,
    iterations::Int = 0,
    solve_time::Float64 = 0.0,
    metadata...
) where T
    return GNEPSolution{T}(trajectories, costs, equilibrium_type;
                          converged=converged, iterations=iterations,
                          solve_time=solve_time, 
                          metadata=Dict{Symbol, Any}(metadata...))
end