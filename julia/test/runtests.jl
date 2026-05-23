using DomcsisEpicTinkerBox
using Test

@testset "DomcsisEpicTinkerBox.jl" begin
    @testset "greet" begin
        @test greet() === nothing
    end

    @testset "mean" begin
        @test mean([1.0, 2.0, 3.0]) ≈ 2.0
        @test mean([0, 0, 0]) ≈ 0.0
        @test mean([-1.0, 1.0]) ≈ 0.0
        @test mean([42.0]) ≈ 42.0
        @test_throws ArgumentError mean(Float64[])
    end
end
