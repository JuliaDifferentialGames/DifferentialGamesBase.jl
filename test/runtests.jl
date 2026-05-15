using Test
using LinearAlgebra
using SparseArrays
using DifferentialGamesBase

@testset "DifferentialGamesBase" begin
    include("test_phase0.jl")
    include("test_phase1.jl")
    include("test_phase2.jl")
    include("test_phase3.jl")
    include("test_phase4.jl")
    include("game_building_tests.jl")
    include("inverse_game_building_tests.jl")
    include("ltv_tests.jl")
end