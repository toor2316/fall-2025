using Test
using Random
using Distributions
using LinearAlgebra
include("ToorPS1.jl")

@testset "ToorPS1 Tests" begin
    @testset "q1 function tests" begin
        # Test q1 function
        Random.seed!(1234)  # Set same seed as in main code
        A, B, C, D = q1()
        
        # Test dimensions
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == size(A)
        
        # Test value ranges
        @test all(-5 .<= A .<= 10)  # A should be between -5 and 10
        @test typeof(B) == Matrix{Float64}  # B should be a matrix of Float64
        
        # Test C construction
        @test C[:, 1:5] == A[1:5, 1:5]
        @test C[:, 6:7] == B[1:5, 6:7]
        
        # Test D is boolean matrix
        @test eltype(D) == Bool
        @test all(x -> x in [true, false], D)
    end
    
    @testset "q2 function tests" begin
        # Create test matrices
        test_A = [1.0 2.0; 3.0 4.0]
        test_B = [2.0 3.0; 4.0 5.0]
        
        # Test element-wise multiplication
        result = q2(test_A, test_B)
        expected = test_A .* test_B
        @test result ≈ expected
    end
    
    @testset "Matrix operations tests" begin
        # Test matrix creation and operations
        Random.seed!(1234)
        A = -5 .+ rand(10,7)
        B = rand(Normal(-2,15),10,7)
        
        # Test dimensions
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        
        # Test basic matrix operations
        C = [A[1:5,1:5] B[1:5,6:7]]
        @test size(C) == (5, 7)
        
        # Test dummy variable creation
        D = A .* (A .<= 0)
        @test size(D) == size(A)
        @test all(x -> x <= 0, filter(!iszero, D))
    end
end
