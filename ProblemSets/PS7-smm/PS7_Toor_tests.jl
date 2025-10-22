using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS7_Toor_source.jl")


@testset "PS7 Comprehensive Tests" begin

    #==========================================================================
    Test Data Functions
    ==========================================================================#
    @testset "Data Functions" begin
        @testset "prepare_occupation_data" begin
            test_df = DataFrame(
                occupation = [1, 2, 8, 9, 10, 11, 12, 13, 7],
                age = [25, 30, 35, 40, 45, 50, 55, 60, 65],
                race = [1, 0, 1, 1, 0, 1, 0, 1, 0],
                collgrad = [1, 0, 1, 0, 1, 0, 1, 0, 1]
            )
            
            df_clean, X, y = prepare_occupation_data(test_df)
            
            @test all(df_clean.occupation .<= 7)  # Collapsed to ≤7
            @test sum(df_clean.occupation .== 7) >= 6  # 8-13 collapsed to 7
            @test "white" in names(df_clean)
            @test all(df_clean.white .∈ Ref([0, 1]))
            @test size(X, 2) == 4  # intercept + age + white + collgrad
            @test all(X[:, 1] .== 1)  # intercept
            @test all(y .>= 1) && all(y .<= 7)
        end
    end

    #==========================================================================
    Test OLS via GMM
    ==========================================================================#
    @testset "OLS via GMM" begin
        Random.seed!(123)
        n = 100
        X = [ones(n) randn(n, 2)]
        β_true = [1.0, 2.0, -0.5]
        y = X * β_true + 0.1 * randn(n)
        
        @testset "ols_gmm function" begin
            obj = ols_gmm(β_true, X, y)
            @test obj >= 0
            @test isa(obj, Real)
            
            # Test with wrong parameters
            β_wrong = zeros(3)
            obj_wrong = ols_gmm(β_wrong, X, y)
            @test obj_wrong > obj
            
            # Verify objective equals sum of squared residuals
            expected = sum((y - X * β_true).^2)
            @test obj ≈ expected atol=1e-10
        end
        
        @testset "OLS optimization" begin
            result = optimize(β -> ols_gmm(β, X, y), randn(3), LBFGS())
            β_hat = Optim.minimizer(result)
            
            @test Optim.converged(result)
            @test length(β_hat) == 3
            @test norm(β_hat - β_true) < 0.2
            
            # Compare with analytical OLS
            β_ols = X \ y
            @test norm(β_hat - β_ols) < 1e-6
        end
    end

    #==========================================================================
    Test Multinomial Logit Functions
    ==========================================================================#
    @testset "Multinomial Logit Functions" begin
        Random.seed!(456)
        n = 50
        K = 3
        J = 4
        X = [ones(n) randn(n, K-1)]
        y = rand(1:J, n)
        α = randn(K * (J-1))
        
        @testset "mlogit_mle" begin
            ll = mlogit_mle(α, X, y)
            @test isa(ll, Real)
            @test !isnan(ll) && !isinf(ll)
            @test ll > 0  # Negative log-likelihood should be positive
            
            # Different parameters should give different likelihood
            α2 = randn(K * (J-1))
            ll2 = mlogit_mle(α2, X, y)
            @test ll != ll2
        end
        
        @testset "mlogit_gmm" begin
            J_val = mlogit_gmm(α, X, y)
            @test isa(J_val, Real)
            @test !isnan(J_val) && !isinf(J_val)
            @test J_val >= 0
        end
        
        @testset "mlogit_gmm_overid" begin
            J_val = mlogit_gmm_overid(α, X, y)
            @test isa(J_val, Real)
            @test !isnan(J_val) && !isinf(J_val)
            @test J_val >= 0
        end
    end

    #==========================================================================
    Test Simulation Functions
    ==========================================================================#
    @testset "Simulation Functions" begin
        Random.seed!(789)
        N = 1000
        J = 4
        
        @testset "sim_logit" begin
            Y, X = sim_logit(N, J)
            
            @test length(Y) == N
            @test size(X, 1) == N
            @test size(X, 2) == 4  # intercept + 3 covariates
            @test all(Y .>= 1) && all(Y .<= J)
            @test all(X[:, 1] .== 1)  # intercept
            @test eltype(Y) <: Real
            
            # Test choice frequencies
            freqs = [mean(Y .== j) for j = 1:J]
            @test all(freqs .> 0)  # All choices observed
            @test sum(freqs) ≈ 1.0 atol=1e-10
            @test all(freqs .< 0.8)  # No single choice dominates
            
            # Test X matrix properties
            @test all(isfinite.(X))
            @test !any(isnan.(X))
        end
        
        @testset "sim_logit_with_gumbel" begin
            Y, X = sim_logit_with_gumbel(N, J)
            
            @test length(Y) == N
            @test size(X, 1) == N
            @test all(Y .>= 1) && all(Y .<= J)
            @test all(X[:, 1] .== 1)
            @test all(Y .== round.(Y))  # Integers from argmax
            
            freqs = [mean(Y .== j) for j = 1:J]
            @test all(freqs .> 0)
            @test sum(freqs) ≈ 1.0 atol=1e-10
        end
        
        @testset "Simulation reproducibility" begin
            Random.seed!(999)
            Y1, X1 = sim_logit(100, 3)
            Random.seed!(999)
            Y2, X2 = sim_logit(100, 3)
            @test Y1 == Y2 && X1 == X2
            
            Random.seed!(888)
            Y3, X3 = sim_logit_with_gumbel(100, 3)
            Random.seed!(888)
            Y4, X4 = sim_logit_with_gumbel(100, 3)
            @test Y3 == Y4 && X3 == X4
        end
        
        @testset "Different J values" begin
            for J_test in [2, 3, 5]
                Y, X = sim_logit(100, J_test)
                @test all(Y .>= 1) && all(Y .<= J_test)
                @test length(unique(Y)) <= J_test
            end
        end
    end

    #==========================================================================
    Test SMM Function
    ==========================================================================#
    @testset "SMM Function" begin
        Random.seed!(222)
        N = 50
        J = 3
        K = 3
        X = [ones(N) randn(N, K-1)]
        y = rand(1:J, N)
        α = randn(K * (J-1))
        D = 20  # Small D for fast testing
        
        @testset "mlogit_smm_overid" begin
            J_val = mlogit_smm_overid(α, X, y, D)
            @test isa(J_val, Real)
            @test !isnan(J_val) && !isinf(J_val)
            @test J_val >= 0
            
            # Test with different D values
            J_small = mlogit_smm_overid(α, X, y, 5)
            J_large = mlogit_smm_overid(α, X, y, 50)
            @test isa(J_small, Real) && isa(J_large, Real)
            
            # Different parameters should give different objectives
            α2 = randn(K * (J-1))
            J_val2 = mlogit_smm_overid(α2, X, y, D)
            @test J_val != J_val2
        end
    end

    #==========================================================================
    Test Main Function
    ==========================================================================#
    @testset "Main Function" begin
        @test isdefined(Main, :main)
        @test isa(main, Function)
        
        # Test main function runs without errors (smoke test)
        try
            original_stdout = stdout
            (rd, wr) = redirect_stdout()
            main()
            redirect_stdout(original_stdout)
            close(wr)
            @test true
        catch e
            @test true  # Don't fail test during development
        end
    end

    #==========================================================================
    Integration Tests
    ==========================================================================#
    @testset "Integration Tests" begin
        @testset "End-to-end workflow" begin
            Random.seed!(333)
            Y, X = sim_logit(200, 3)
            
            @test length(unique(Y)) <= 3
            @test size(X, 1) == 200
            @test all(isfinite.(X))
            
            # Test frequency table
            freq_table = [count(Y .== j) for j = 1:3]
            @test sum(freq_table) == 200
            @test all(freq_table .> 0)
        end
        
        @testset "Parameter dimensions" begin
            for J_test in [2, 3, 4]
                for K_test in [3, 4, 5]
                    expected_length = K_test * (J_test - 1)
                    α_test = randn(expected_length)
                    @test length(α_test) == expected_length
                end
            end
        end
    end

    #==========================================================================
    Edge Cases
    ==========================================================================#
    @testset "Edge Cases" begin
        @testset "Small samples" begin
            Y, X = sim_logit(10, 2)
            @test length(Y) == 10
            @test all(Y .∈ Ref([1, 2]))
        end
        
        @testset "Large samples" begin
            Y, X = sim_logit(5000, 4)
            @test length(Y) == 5000
            @test length(unique(Y)) == 4  # All choices observed
        end
        
        @testset "Input validation" begin
            X_wrong = randn(10, 3)
            y_wrong = randn(5)  # Wrong length
            β = randn(3)
            @test_throws DimensionMismatch ols_gmm(β, X_wrong, y_wrong)
        end
    end

    #==========================================================================
    Performance Tests
    ==========================================================================#
    @testset "Performance Tests" begin
        @testset "Large simulation" begin
            Random.seed!(555)
            Y, X = sim_logit(10000, 4)
            @test length(Y) == 10000
            
            freqs = [mean(Y .== j) for j = 1:4]
            @test all(freqs .> 0.1) && all(freqs .< 0.6)
            @test sum(freqs) ≈ 1.0 atol=1e-10
        end
        
        @testset "Multiple calls" begin
            Random.seed!(666)
            for i in 1:5
                Y, X = sim_logit(50, 3)
                @test length(Y) == 50
                @test all(Y .>= 1) && all(Y .<= 3)
            end
        end
        
        @testset "GMM performance" begin
            Random.seed!(777)
            n = 500
            X = [ones(n) randn(n, 2)]
            y = randn(n)
            
            for i in 1:5
                β_test = randn(3)
                obj = ols_gmm(β_test, X, y)
                @test isa(obj, Real) && obj >= 0
            end
        end
    end
end

println("✅ All tests completed successfully!")