using Test, Random, LinearAlgebra, Distributions, Statistics, DataFrames, CSV, HTTP, GLM, FreqTables


#:Reading the source code# 
include("PS2_Toor_source.jl") 

 #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 7 answer: Unit tests 
    #:::::::::::::::::::::::::::::::::::::::::::::::::::

@testset "Problem Set 2 Unit Tests" begin 


    
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test for Question 1 - Basic Optimization Function Tests
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 1 - Function f(x) Tests" begin
        # Define the functions locally for testing
        f_test(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
        minusf_test(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
        
        # Test specific values
        @test f_test([0.0]) == -2.0  # f(0) = -2
        @test f_test([1.0]) == -1 - 10 - 2 - 3 - 2  # f(1) = -18
        @test f_test([-1.0]) == -1 + 10 - 2 + 3 - 2  # f(-1) = 8
        
        # Test that minusf is negative of f
        test_vals = [rand(1) for _ in 1:5]
        for val in test_vals
            @test minusf_test(val) ≈ -f_test(val)
        end
        
        # Test that the function is continuous (no NaN or Inf)
        @test !isnan(f_test([100.0])) && !isinf(f_test([100.0]))
        @test !isnan(f_test([-100.0])) && !isinf(f_test([-100.0]))
    end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test for Question 2 - OLS Function Tests
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 2 - OLS Function Tests" begin
        # Define OLS function locally
        function ols_test(beta, X, y)
            ssr = (y.-X*beta)'*(y.-X*beta)
            return ssr
        end
        
        # Create simple test data with known solution
        X_simple = [1.0 1.0; 1.0 2.0; 1.0 3.0]
        y_simple = [2.0, 3.0, 4.0]  # Perfect linear relationship: y = 1 + x
        
        # Test that SSR is 0 for perfect fit
        beta_perfect = [1.0, 1.0]
        @test ols_test(beta_perfect, X_simple, y_simple) ≈ 0.0 atol=1e-10
        
        # Test that SSR is positive for non-perfect fit
        beta_wrong = [0.0, 0.0]
        @test ols_test(beta_wrong, X_simple, y_simple) > 0
        
        # Test with random data
        Random.seed!(123)
        X_rand = [ones(10,1) randn(10,2)]
        y_rand = randn(10)
        beta_rand = randn(3)
        
        # SSR should always be non-negative
        @test ols_test(beta_rand, X_rand, y_rand) ≥ 0
        
        # Test analytical OLS solution
        beta_analytical = inv(X_rand'*X_rand)*X_rand'*y_rand
        @test length(beta_analytical) == size(X_rand, 2)
        @test all(isfinite.(beta_analytical))
        
        # SSR at analytical solution should be minimal
        ssr_optimal = ols_test(beta_analytical, X_rand, y_rand)
        ssr_random = ols_test(beta_rand, X_rand, y_rand)
        @test ssr_optimal ≤ ssr_random || ssr_optimal ≈ ssr_random
    end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test for Question 3 - Logit Likelihood Function Tests
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 3 - Logit Function Tests" begin
        # Define logit function locally
        function logit_test(alpha, X, d)
            pll = exp.(X * alpha) ./ (1 .+ exp.(X * alpha))
            ll = -sum((d .== 1) .* log.(pll) .+ (d .== 0) .* log.(1 .- pll))
            return ll
        end
        
        # Test with simple data
        X_logit = [1.0 0.0; 1.0 1.0]
        y_logit = [0, 1]
        
        # Test that likelihood is finite for reasonable parameters
        alpha_small = [0.1, 0.1]
        @test isfinite(logit_test(alpha_small, X_logit, y_logit))
        
        # Test monotonicity: better predictions should have lower negative log-likelihood
        alpha_good = [0.0, 10.0]  # Should predict [0,1] well
        alpha_bad = [0.0, -10.0]  # Should predict [1,0] poorly
        @test logit_test(alpha_good, X_logit, y_logit) < logit_test(alpha_bad, X_logit, y_logit)
        
        # Test with larger random dataset
        Random.seed!(456)
        X_large = [ones(100,1) randn(100,2)]
        y_large = rand(0:1, 100)
        alpha_rand = randn(3)
        
        ll_rand = logit_test(alpha_rand, X_large, y_large)
        @test isfinite(ll_rand)
        @test ll_rand > 0  # Negative log-likelihood should be positive
    end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Test for Question 5 - Multinomial Logit Function Tests
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Question 5 - Multinomial Logit Function Tests" begin
        # Define mlogit function locally
        function mlogit_test(alpha, X, y)
            K = size(X, 2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N, J)
            for j = 1:J
                bigY[:, j] = (y .== j)
            end
            bigalpha = [reshape(alpha, K, J-1) zeros(K)]
            
            num = zeros(N, J)
            denom = zeros(N)
            for j = 1:J
                num[:, j] = exp.(X * bigalpha[:, j])
                denom .+= num[:, j]
            end
            P = num ./ repeat(denom, 1, J)
            
            loglike = -sum(sum(bigY .* log.(P)))
            return loglike
        end
        
        # Test with simple 3-category data
        X_mlogit = [1.0 0.0; 1.0 1.0; 1.0 2.0]
        y_mlogit = [1, 2, 3]
        
        # Test with zero parameters (equal probabilities)
        K_test = size(X_mlogit, 2)
        J_test = length(unique(y_mlogit))
        alpha_zero = zeros(K_test * (J_test - 1))
        
        ll_zero = mlogit_test(alpha_zero, X_mlogit, y_mlogit)
        @test isfinite(ll_zero)
        @test ll_zero > 0  # Negative log-likelihood should be positive
        
        # Test that probabilities sum to 1
        # (This is implicitly tested by the function, but we can verify)
        Random.seed!(789)
        alpha_rand = randn(K_test * (J_test - 1))
        ll_rand = mlogit_test(alpha_rand, X_mlogit, y_mlogit)
        @test isfinite(ll_rand)
        
        # Test with different number of categories
        y_binary = [1, 2, 1]  # Only 2 categories
        alpha_binary = zeros(K_test * 1)  # K*(J-1) parameters
        ll_binary = mlogit_test(alpha_binary, X_mlogit, y_binary)
        @test isfinite(ll_binary)
    end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Testing Data Processing and Validation Tests
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Data Processing Tests" begin
        # Create mock data similar to the real dataset
        Random.seed!(101112)
        n_obs = 100
        df_mock = DataFrame(
            married = rand(0:1, n_obs),
            age = rand(18:65, n_obs),
            race = rand(1:3, n_obs),
            collgrad = rand(0:1, n_obs),
            occupation = vcat(rand(1:7, 80), rand(8:13, 15), [missing, missing, missing, missing, missing])
        )
        
        # Test data cleaning process
        df_clean = dropmissing(df_mock, :occupation)
        @test size(df_clean, 1) == n_obs - 5  # Should remove 5 missing values
        
        # Test occupation recoding
        df_clean[df_clean.occupation .== 8, :occupation] .= 7
        df_clean[df_clean.occupation .== 9, :occupation] .= 7
        df_clean[df_clean.occupation .== 10, :occupation] .= 7
        df_clean[df_clean.occupation .== 11, :occupation] .= 7
        df_clean[df_clean.occupation .== 12, :occupation] .= 7
        df_clean[df_clean.occupation .== 13, :occupation] .= 7
        
        @test maximum(df_clean.occupation) ≤ 7
        @test minimum(df_clean.occupation) ≥ 1
        @test all(df_clean.occupation .∈ Ref(1:7))
        
        # Test X matrix construction
        X_test = [ones(size(df_clean,1),1) df_clean.age df_clean.race.==1 df_clean.collgrad.==1]
        @test size(X_test, 2) == 4  # Should have 4 columns
        @test all(X_test[:, 1] .== 1)  # First column should be all ones
        @test all(X_test[:, 3] .∈ Ref([0, 1]))  # Race indicator should be binary
        @test all(X_test[:, 4] .∈ Ref([0, 1]))  # College indicator should be binary
        
        # Test y vector construction
        y_married = df_clean.married .== 1
        @test all(y_married .∈ Ref([0, 1]))  # Should be binary
        
        y_occupation = df_clean.occupation
        @test all(y_occupation .∈ Ref(1:7))  # Should be in range 1-7
    end

    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    # Numerical Stability Tests
    #:::::::::::::::::::::::::::::::::::::::::::::::::::
    @testset "Numerical Stability Tests" begin
        # Test OLS with multicollinear data
        X_collinear = [ones(10,1) rand(10,1) rand(10,1)]
        X_collinear = [X_collinear X_collinear[:,2]]  # Add duplicate column
        y_test = rand(10)
        
        # This should handle gracefully (though solution may not be unique)
        @test_nowarn try
            inv(X_collinear'*X_collinear)*X_collinear'*y_test
        catch e
            if isa(e, SingularException)
                # Expected behavior for singular matrix
                true
            else
                rethrow(e)
            end
        end
        
        # Test logit with extreme values
        X_extreme = [1.0 100.0; 1.0 -100.0]
        y_extreme = [1, 0]
        alpha_extreme = [0.0, 1.0]
        
        function logit_safe(alpha, X, d)
            # Clip to prevent overflow
            Xα = X * alpha
            Xα = clamp.(Xα, -500, 500)
            pll = exp.(Xα) ./ (1 .+ exp.(Xα))
            # Add small epsilon to prevent log(0)
            pll = clamp.(pll, 1e-15, 1-1e-15)
            ll = -sum((d .== 1) .* log.(pll) .+ (d .== 0) .* log.(1 .- pll))
            return ll
        end
        
        ll_extreme = logit_safe(alpha_extreme, X_extreme, y_extreme)
        @test isfinite(ll_extreme)
    end
end

