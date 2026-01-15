module DifferentialGamesBase

using LinearAlgebra
using SparseArrays
using ForwardDiff

# Includes
include("problems/base.jl")
include("objectives.jl")           
include("constraints.jl")   
include("player.jl")
include("metadata.jl")
include("dynamics.jl")
include("time_horizon.jl")
include("problems/GNEP.jl") 
include("helper.jl")
        

# Exports - Abstract Types
export 
    AbstractGame,
    DynamicsSpec,
    TimeHorizon,
    
    # Cost types (from objectives.jl)
    AbstractCost,
    AbstractStageCost,
    AbstractTerminalCost,
    AbstractConvexConstraint,
    
    # Constraint types (from constraints.jl)
    AbstractConstraint,
    AbstractEqualityConstraint,
    AbstractInequalityConstraint,
    PrivateConstraint,
    SharedConstraint

# Exports - Concrete Dynamics Types
export
    SeparableDynamics,
    LinearDynamics,
    CoupledNonlinearDynamics

# Exports - Time Horizon Types
export
    ContinuousTime,
    DiscreteTime

# Exports - Cost Types
export
    # Stage costs
    LQStageCost,
    DiagonalLQStageCost,
    NonlinearStageCost,
    SeparableCost,
    CoupledCost,
    
    # Terminal costs
    LQTerminalCost,
    DiagonalLQTerminalCost,
    NonlinearTerminalCost,
    
    # Player objective
    PlayerObjective

# Exports - Constraint Types
export
    # Convex constraints
    LinearConstraint,
    BoundConstraint,
    NormConstraint,
    
    # Nonlinear constraints
    NonlinearConstraint,
    
    # Constraint evaluation
    evaluate_constraint,
    constraint_jacobian

# Exports - Cost Evaluation Interface
export
    evaluate_stage_cost,
    evaluate_terminal_cost,
    stage_cost_gradient,
    terminal_cost_gradient,
    stage_cost_hessian,
    terminal_cost_hessian,
    total_cost,
    diagnose_scaling

# Exports - Game Problem Types
export
    GameProblem,
    PlayerSpec,
    CouplingGraph,
    GameMetadata

# Exports - Game Constructors
export
    PDGNEProblem,
    LQGameProblem,
    UnconstrainedLQGame

# Exports - Property Query Functions
export
    has_separable_dynamics,
    is_lq_game,
    is_potential_game,
    has_shared_constraints,
    is_unconstrained,
    is_pd_gnep,
    is_lq_pd_gnep,
    is_separable

# Exports - Helper Functions
export
    num_players,
    state_dim,
    control_dim,
    get_objective,
    
    # Dynamics evaluation
    evaluate_dynamics,
    dynamics_jacobian,
    
    # Constraint type checks
    is_equality,
    is_inequality,
    is_private,
    is_shared

end