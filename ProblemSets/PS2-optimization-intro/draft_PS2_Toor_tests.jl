using Test, Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

#:Reading the source code# 
include("PS2_Toor_source.jl") 

@testset "Problem Set 2 Tests" begin
    
    @testset "Question 1 - Basic Optimization" begin
        # Test f(x) function
        @test f([-7.0]) ≈ 964.0 rtol=1e-3  # Known value near the maximum
        @test f([0.0]) == -2.0  # Test at x = 0
        
        # Test minusf(x) function
        @test minusf([-7.0]) ≈ -964.0 rtol=1e-3
        @test minusf([0.0]) == 2.0
        
        # Test optimization result
        result = optimize(minusf, [-7.0], BFGS())
        @test Optim.minimizer(result)[1] ≈ -7.38 rtol=1e-2  # Should be close to -7.38
        @test Optim.minimum(result) < 0  # Should find a negative minimum
    end

    @testset "Question 2 - OLS Tests" begin
        # Create test data
        X_test = [1.0 2.0; 1.0 3.0; 1.0 4.0]
        y_test = [2.0, 4.0, 6.0]
        
        # Test OLS function
        @test ols([1.0, 1.0], X_test, y_test) ≥ 0  # SSR should be non-negative
        
        # Test analytical solution
        bols_test = inv(X_test'*X_test)*X_test'*y_test
        @test length(bols_test) == size(X_test, 2)  # Check dimensions
        @test all(isfinite.(bols_test))  # Check for finite values
        
        # Test standard errors calculation
        N_test = size(X_test, 1)
        K_test = size(X_test, 2)
        MSE_test = sum(((y_test-X_test*bols_test).^2)/(N_test-K_test))
        @test MSE_test ≥ 0  # MSE should be non-negative
    end

    @testset "Question 3 - Logit Tests" begin
        # Test logit function with simple data
        X_test = [1.0 0.0; 1.0 1.0]
        y_test = [0, 1]
        
        # Test basic properties
        ll = logit([0.0, 0.0], X_test, y_test)
        @test isfinite(ll)  # Log-likelihood should be finite
        @test ll > -Inf  # Log-likelihood should be greater than negative infinity
        
        # Test with extreme probabilities
        ll_extreme = logit([100.0, 100.0], X_test, y_test)
        @test ll_extreme ≠ ll  # Should be different from neutral case
    end

    @testset "Question 4 - GLM Tests" begin
        # Create simple test data
        df_test = DataFrame(
            married = [1, 0, 1, 0],
            age = [25, 30, 35, 40],
            white = [1, 0, 1, 0],
            collgrad = [1, 0, 1, 0]
        )
        
        # Test GLM model
        model = glm(@formula(married ~ age + white + collgrad), df_test, Binomial(), LogitLink())
        @test length(coef(model)) == 4  # Should have 4 coefficients
        @test all(isfinite.(coef(model)))  # All coefficients should be finite
    end

    @testset "Question 5 - Multinomial Logit Tests" begin
        # Create test data
        X_test = [1.0 0.0; 1.0 1.0; 1.0 0.0]
        y_test = [1, 2, 3]
        alpha_test = zeros(4)  # 2 vars * (3-1) categories
        
        # Test mlogit function
        ll = mlogit(alpha_test, X_test, y_test)
        @test isfinite(ll)  # Log-likelihood should be finite
        @test ll > -Inf  # Log-likelihood should be greater than negative infinity
        
        # Test dimensions
        K = size(X_test, 2)
        J = length(unique(y_test))
        @test length(alpha_test) == K*(J-1)  # Check parameter vector dimension
    end

    @testset "Data Processing Tests" begin
        # Load dataset
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df_test = CSV.read(HTTP.get(url).body, DataFrame)
        
        # Test occupation recoding
        df_clean = dropmissing(df_test, :occupation)
        df_clean.occupation = ifelse.(df_clean.occupation .> 7, 7, df_clean.occupation)
        @test maximum(df_clean.occupation) ≤ 7  # Should be ≤ 7 after recoding
        @test minimum(df_clean.occupation) ≥ 1  # Should be ≥ 1
    end
end
