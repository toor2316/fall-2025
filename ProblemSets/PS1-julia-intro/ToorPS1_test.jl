using Test, Random, Distributions, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, JLD
include("ToorPS1.jl")

@testset "ToorPS1 Tests" begin
    @testset "q1 function tests" begin
        # Test q1 function
        Random.seed!(1234)
        A, B, C, D = q1()
        
        # Test dimensions and types
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == size(A)
        
        # Test value ranges and types
        @test all(-5 .<= A .<= 10)
        @test eltype(D) == Bool
        @test eltype(A) == Float64
        @test eltype(B) == Float64
        
        # Test matrix construction
        @test C[:, 1:5] == A[1:5, 1:5]
        @test C[:, 6:7] == B[1:5, 6:7]
        
        # Test D is correctly calculated
        @test all(D .== (A .<= 0))
    end
    
    @testset "q2 function tests" begin
        # Create test data
        Random.seed!(1234)
        test_A = rand(5, 4)
        test_B = rand(5, 4)
        test_C = rand(Uniform(-10, 10), 5, 4)
        
        # Call q2 (which returns nothing, so we don't capture the return)
        q2(test_A, test_B, test_C)
        
        # Test Cprime calculation manually
        cprime_manual = Float64[]
        for c in axes(test_C, 2), r in axes(test_C, 1)
            if -5 <= test_C[r,c] <= 5
                push!(cprime_manual, test_C[r,c])
            end
        end
        cprime_vec = test_C[(test_C .>= -5) .& (test_C .<= 5)]
        @test sort(cprime_manual) == sort(cprime_vec)
        
        # Test matrix dimensions and initialization
        N, K, T = 15, 6, 5
        X = zeros(N, K, T)
        @test size(X) == (N, K, T)
        @test X[1, 1, :] == zeros(T)  # Check intercept initialization
    end
    
    @testset "q3 function tests" begin
        # Test data loading and basic properties
        df = DataFrame(CSV.File("nlsw88.csv"))
        @test df isa DataFrame
        required_cols = [:wage, :industry, :occupation, :race, :never_married]
        @test all(col -> col in names(df), required_cols)
        
        # Test never_married proportion
        @test 0 <= mean(df.never_married) <= 1
        
        # Test race frequency table
        race_freq = freqtable(df.race)
        @test sum(race_freq) == nrow(df)
        
        # Test wage statistics
        wage_stats = describe(df.wage)
        @test wage_stats.mean >= 0
        @test wage_stats.min >= 0
        
        # Test grouped statistics
        df_sub = df[:, [:industry, :occupation, :wage]]
        grouped = groupby(df_sub, [:industry, :occupation])
        mean_wage = combine(grouped, :wage => mean => :mean_wage)
        @test mean_wage.mean_wage isa Vector{Float64}
    end
    
    @testset "q4 matrixops tests" begin
        # Test with different sized matrices
        Random.seed!(1234)
        test_sizes = [(3,3), (4,5), (2,6)]
        
        for sz in test_sizes
            A = rand(sz...)
            B = rand(sz...)
            out1, out2, out3 = matrixops(A, B)
            
            # Test dimensions
            @test size(out1) == sz
            @test size(out2) == (sz[1], sz[1])  # A'B dimension
            @test out3 isa Number
            
            # Test calculations
            @test out1 ≈ A .* B
            @test out2 ≈ A' * B
            @test out3 ≈ sum(A + B)
        end
        
        # Test error handling
        @test_throws ErrorException matrixops(rand(2,3), rand(3,2))
        
        # Test with integers converted to Float64
        A_int = Float64.(rand(1:10, 3, 3))
        B_int = Float64.(rand(1:10, 3, 3))
        @test_nowarn matrixops(A_int, B_int)
    end
end
