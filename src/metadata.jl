# ============================================================================
# Game Metadata
# ============================================================================

"""
    CouplingGraph

Encodes coupling structure between players for solver exploitation.

# Fields
- `cost_coupling::SparseMatrixCSC{Bool}` : cost_coupling[i,j] = true if Jᵢ depends on uⱼ
- `constraint_coupling::Vector{Vector{Int}}` : Players involved in each shared constraint
- `dynamics_coupling::Union{Nothing, SparseMatrixCSC{Bool}}` : For coupled dynamics

# Notes
Sparse coupling enables:
- Parallel computation of independent subproblems
- Reduced communication in distributed algorithms
- Improved convergence rates for iterative methods
"""
struct CouplingGraph
    cost_coupling::SparseMatrixCSC{Bool}
    constraint_coupling::Vector{Vector{Int}}
    dynamics_coupling::Union{Nothing, SparseMatrixCSC{Bool}}
end

"""
    GameMetadata

Cached information about game structure for solver efficiency.

# Fields
- `state_dims::Vector{Int}` : State dimensions per player
- `control_dims::Vector{Int}` : Control dimensions per player
- `state_offsets::Vector{Int}` : Starting indices in stacked state vector
- `control_offsets::Vector{Int}` : Starting indices in stacked control vector
- `coupling_graph::CouplingGraph` : Coupling structure between players
- `is_potential::Bool` : Whether game has potential function structure
- `potential_function::Union{Nothing, Function}` : Potential function if exists
- `communication::Union{Nothing, AbstractCommunicationTopology}` : Which other
  players' information each player's controller may use; `nothing` means
  unspecified (solvers should treat this as `FullCommunication()`).

# Notes
Computed once at problem construction, reused by solvers.
"""
struct GameMetadata
    state_dims::Vector{Int}
    control_dims::Vector{Int}
    state_offsets::Vector{Int}
    control_offsets::Vector{Int}
    coupling_graph::CouplingGraph
    is_potential::Bool
    potential_function::Union{Nothing, Function}
    communication::Union{Nothing, AbstractCommunicationTopology}
end

GameMetadata(
    state_dims::Vector{Int}, control_dims::Vector{Int},
    state_offsets::Vector{Int}, control_offsets::Vector{Int},
    coupling_graph::CouplingGraph, is_potential::Bool,
    potential_function::Union{Nothing, Function}
) = GameMetadata(state_dims, control_dims, state_offsets, control_offsets,
                  coupling_graph, is_potential, potential_function, nothing)