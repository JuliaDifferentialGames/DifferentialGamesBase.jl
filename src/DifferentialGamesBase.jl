module DifferentialGamesBase


# Includes
include("problems/base.jl")
include("problems/GNEP.jl")

# Exports
export 
    # Abstract Types 
    DifferentialGame, 
    InverseDifferentialGame,
    AbstractLQGame,

    # Player-based Interface 
    ConstraintSpec,
    Player,

    # Problems 
    AbstractGNEP, 
    AbstractPDGNEP, 
    AbstractPotentialGame, 
    AbstractLexicographicGame, 
    AbstractLQGame, 
    LQGameProblem, 
    GNEProblem,
    PDGNEProblem, 
    PotentialGameProblem, 
    LexicographicGameProblem, 

    # Helper functions
    num_players, 
    state_dim, 
    control_dim

end