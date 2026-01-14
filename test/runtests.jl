using DifferentialGamesBase
using Test

@testset "DifferentialGamesBase.jl" begin
    # Write your tests here.
    @test test_f(2,1) == 7
    @test test_f(2,3) == 13
end
