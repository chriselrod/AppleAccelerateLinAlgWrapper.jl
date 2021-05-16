using AppleAccelerateLinAlgWrapper
using Test

@testset "AppleAccelerateLinAlgWrapper.jl" begin

    for T ∈ (Float32,Float64)
        A = rand(T,10,10);
        B = rand(T,10,10);
        @test AppleAccelerateLinAlgWrapper.gemm(A,B) ≈ A*B
        @test AppleAccelerateLinAlgWrapper.rdiv(A,B) ≈ A/B
        @test AppleAccelerateLinAlgWrapper.ldiv(A,B) ≈ A\B
    end
end
