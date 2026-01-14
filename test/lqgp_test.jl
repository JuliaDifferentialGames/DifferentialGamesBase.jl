using Test
using LinearAlgebra
using DifferentialGamesBase  # your package module name

@testset "LQGame construction" begin
    # Define test matrices and vectors
    A = [0.0 1.0; -1.0 -0.5]
    B = reshape([0.0; 1.0], 2, 1)
    Q = Matrix{Float64}(I(2)) 
    R = Matrix{Float64}(1.0 * I(2))  
    x0 = [1.0, 0.0]

    # Define game parameters
    n_players = 2
    tf = 10.0  # final time

    # Create per-player control matrices
    B_players = [B, B]  # Each player has control matrix B

    # Create per-player cost matrices
    Q_players = [Matrix(Q), Matrix(Q)]  # Each player has same state cost
    R_players = [Matrix(R[1:1, 1:1]), Matrix(R[2:2, 2:2])]  # Each player controls 1 input

    # Terminal cost matrices (typically same as running cost Q)
    Qf_players = [Matrix(Q), Matrix(Q)]

    # Control dimensions per player
    control_dims = [1, 1]  # Each player controls 1 input

    # Create the LQ game problem
    game = LQGameProblem(
        A,
        B_players,
        Q_players,
        R_players,
        Qf_players,
        x0,
        tf,
        control_dims
    )
end