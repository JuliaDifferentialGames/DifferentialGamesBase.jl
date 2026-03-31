module DifferentialGamesBase

using LinearAlgebra
using SparseArrays
using ForwardDiff

# Includes
include("problems/base.jl")
include("objectives.jl")            
include("constraints.jl")
include("metadata.jl")
include("dynamics.jl")
include("time_horizon.jl")
include("player_spec.jl")   
include("problems/GNEP.jl")           
include("utils.jl")
include("solutions/gnep_solutions.jl")
include("solve.jl")
include("problems/IGNEP.jl")
include("solutions/ignep_solutions.jl")
        

# ============================================================================
# Exports - Abstract Types
# ============================================================================
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

# ============================================================================
# Exports - Concrete Dynamics Types
# ============================================================================
export
    SeparableDynamics,
    LinearDynamics,
    CoupledNonlinearDynamics

# ============================================================================
# Exports - Dynamics Accessors (LTI/LTV unified interface)
# ============================================================================
export
    get_A,
    get_B,
    get_B_concatenated,
    is_ltv,
    total_state_dim,
    total_control_dim

# ============================================================================
# Exports - Time Horizon Types
# ============================================================================
export
    ContinuousTime,
    DiscreteTime

# ============================================================================
# Exports - Cost Types
# ============================================================================
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

# ============================================================================
# Exports - Cost Accessors (LTI/LTV unified interface)
# ============================================================================
export
    get_Q,
    get_R,
    get_M,
    get_q,
    get_r

# ============================================================================
# Exports - Constraint Types
# ============================================================================
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

# ============================================================================
# Exports - Cost Evaluation Interface
# ============================================================================
export
    evaluate_stage_cost,
    evaluate_terminal_cost,
    stage_cost_gradient,
    terminal_cost_gradient,
    stage_cost_hessian,
    terminal_cost_hessian,
    total_cost,
    diagnose_scaling

# ============================================================================
# Exports - Game Problem Types
# ============================================================================
export
    GameProblem,
    PlayerSpec,
    CouplingGraph,
    GameMetadata

# ============================================================================
# Exports - Game Constructors
# ============================================================================
export
    PDGNEProblem,
    LQGameProblem,
    LTVLQGameProblem,
    UnconstrainedLQGame

# ============================================================================
# Exports - Property Query Functions
# ============================================================================
export
    has_separable_dynamics,
    is_lq_game,
    is_potential_game,
    has_shared_constraints,
    is_unconstrained,
    is_pd_gnep,
    is_lq_pd_gnep,
    is_separable,
    n_steps

# ============================================================================
# Exports - Helper Functions
# ============================================================================
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

# ============================================================================
# Exports - Solution
# ============================================================================
export 
    Trajectory,
    GameSolution,
    get_trajectory,
    get_cost,
    get_costs,
    is_nash_equilibrium

# ============================================================================
# Exports - Solver
# ============================================================================
export 
    WarmstartData,
    GameSolver,
    solve,
    _solve, 
    solver_capabilities,
    required_capabilities

# ============================================================================
# Exports - Inverse Game: Knowledge Specification
# ============================================================================
export
    PlayerKnowledge,
    KnownObjective,
    UnknownObjective

# ============================================================================
# Exports - Inverse Game: Observation Model
# ============================================================================
export
    ObservationModel,
    observe,
    observation_dim,
    FullStateObservation,
    NoisyObservation

# ============================================================================
# Exports - Inverse Game: Forward Solver Wrapper
# ============================================================================
export
    ForwardSolverWrapper,
    solve_forward,
    predict_next_state

# ============================================================================
# Exports - Inverse Game: Problem Types and Constructors
# ============================================================================
export
    InverseGameProblem,
    InversePDGNEProblem

# ============================================================================
# Exports - Inverse Game: Accessors
# ============================================================================
export
    unknown_players,
    known_players,
    known_objective,
    n_unknown,
    as_forward_problem

# ============================================================================
# Exports - Inverse Game: Solver State and Observation Data
# ============================================================================
export
    ObservationData,
    push_observation!,
    InverseSolverState

# ============================================================================
# Exports - Inverse Game: Solution
# ============================================================================
export
    InverseGameSolution,
    get_weights,
    get_weight_history 

end