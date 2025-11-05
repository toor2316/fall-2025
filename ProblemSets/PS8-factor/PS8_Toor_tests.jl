using Test, Random, LinearAlgebra, Statistics, Distributions, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

cd(@__DIR__)

include("PS8_Toor_source.jl")


@testset "PS8 Factor Model Tests" begin
    
    # Set random seed for reproducibility
    Random.seed!(1234)
    
    # Load real data for testing
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS8-factor/nlsy.csv"
    df = load_data(url)
    
    #==========================================================================
    # Question 1: Load data and base regression
    ==========================================================================#
    @testset "Question 1: Data Loading and Base Regression" begin
        @test size(df, 1) > 0  # Has observations
        @test size(df, 2) == 13  # Has 13 columns
        @test "logwage" in names(df)
        @test "black" in names(df)
        @test "asvabAR" in names(df)
        
        # Test base regression runs
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
        @test result isa StatsModels.TableRegressionModel
        @test length(coef(result)) == 7  # 6 predictors + intercept
    end
    
    #==========================================================================
    # Question 2: ASVAB Correlations
    ==========================================================================#
    @testset "Question 2: ASVAB Correlations" begin
        cordf = compute_asvab_correlations(df)
        
        @test size(cordf) == (6, 6)  # 6x6 correlation matrix
        @test all(diag(Matrix(cordf)) .≈ 1.0)  # Diagonal should be 1
        @test all(Matrix(cordf) .≤ 1.0)  # All correlations ≤ 1
        @test all(Matrix(cordf) .≥ -1.0)  # All correlations ≥ -1
        
        # Check symmetry
        cormat = Matrix(cordf)
        @test cormat ≈ cormat'  # Should be symmetric
        
        # Check that correlations are reasonable (positive for ability tests)
        @test all(cormat .> 0.0)  # All ASVAB scores should be positively correlated
    end
    
    #==========================================================================
    # Question 3: Full Regression with all ASVABs
    ==========================================================================#
    @testset "Question 3: Full Regression" begin
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                            asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
        
        @test result isa StatsModels.TableRegressionModel
        @test length(coef(result)) == 13  # 12 predictors + intercept
        @test r2(result) > 0  # R² should be positive
        
        # Check for multicollinearity warning (high standard errors)
        ses = stderror(result)
        @test any(ses .> 0.01)  # At least some standard errors should be non-trivial
    end
    
    #==========================================================================
    # Question 4: PCA Regression
    ==========================================================================#
    @testset "Question 4: PCA Regression" begin
        df_pca = deepcopy(df)
        df_pca = generate_pca!(df_pca)
        
        @test "asvabPCA" in names(df_pca)
        @test length(df_pca.asvabPCA) == size(df_pca, 1)
        @test all(isfinite.(df_pca.asvabPCA))
        
        # Test PCA regression
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df_pca)
        @test result isa StatsModels.TableRegressionModel
        @test length(coef(result)) == 8  # 7 predictors + intercept
        
        # PCA coefficient should be significant (|t| > 2)
        tstat = coef(result)[end] / stderror(result)[end]
        @test abs(tstat) > 2
    end
    
    #==========================================================================
    # Question 5: Factor Analysis Regression
    ==========================================================================#
    @testset "Question 5: Factor Analysis Regression" begin
        df_factor = deepcopy(df)
        df_factor = generate_factor!(df_factor)
        
        @test "asvabFactor" in names(df_factor)
        @test length(df_factor.asvabFactor) == size(df_factor, 1)
        @test all(isfinite.(df_factor.asvabFactor))
        
        # Test factor regression
        result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df_factor)
        @test result isa StatsModels.TableRegressionModel
        @test length(coef(result)) == 8  # 7 predictors + intercept
        
        # Factor coefficient should be significant (|t| > 2)
        tstat = coef(result)[end] / stderror(result)[end]
        @test abs(tstat) > 2
    end
    
    #==========================================================================
    # Question 6: Factor Model Functions
    ==========================================================================#
    @testset "Question 6: Factor Model Preparation" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Check dimensions
        N = size(df, 1)
        @test size(X) == (N, 7)  # 6 covariates + constant
        @test size(y) == (N,)
        @test size(Xfac) == (N, 4)  # 3 covariates + constant
        @test size(asvabs) == (N, 6)  # 6 ASVAB tests
        
        # Check that last column of X and Xfac are constants
        @test all(X[:, end] .== 1.0)
        @test all(Xfac[:, end] .== 1.0)
        
        # Check no missing values
        @test all(isfinite.(X))
        @test all(isfinite.(y))
        @test all(isfinite.(Xfac))
        @test all(isfinite.(asvabs))
    end
    
    @testset "Question 6: Factor Model Likelihood" begin
        # Create small test dataset
        N_test = 100
        K = 7
        L = 4
        J = 6
        
        X_test = randn(N_test, K)
        X_test[:, end] .= 1.0
        Xfac_test = randn(N_test, L)
        Xfac_test[:, end] .= 1.0
        asvabs_test = randn(N_test, J)
        y_test = randn(N_test)
        
        # Create parameter vector
        γ = randn(L, J)
        β = randn(K)
        α = rand(J+1)
        σ = 0.5 .* ones(J+1)
        θ_test = vcat(γ[:], β, α, σ)
        
        # Test likelihood computation
        negloglike = factor_model(θ_test, X_test, Xfac_test, asvabs_test, y_test, 7)
        
        @test isfinite(negloglike)
        @test negloglike > 0  # Negative log-likelihood should be positive
        
        # Test that changing parameters changes likelihood
        θ_test2 = θ_test .+ 0.1
        negloglike2 = factor_model(θ_test2, X_test, Xfac_test, asvabs_test, y_test, 7)
        @test negloglike2 != negloglike
    end
    
    @testset "Question 6: Starting Values" begin
        X, y, Xfac, asvabs = prepare_factor_matrices(df)
        
        # Test that starting values can be constructed
        svals = vcat(
            Xfac\asvabs[:,1],
            Xfac\asvabs[:,2],
            Xfac\asvabs[:,3], 
            Xfac\asvabs[:,4],
            Xfac\asvabs[:,5], 
            Xfac\asvabs[:,6],
            X\y,
            rand(7), 
            0.5*ones(7)
        )
        
        # Check dimensions
        L = size(Xfac, 2)
        K = size(X, 2)
        J = size(asvabs, 2)
        expected_length = J*L + K + (J+1) + (J+1)
        @test length(svals) == expected_length
        
        # Check all finite
        @test all(isfinite.(svals))
        
        # Check σ parameters are positive
        σ_start = svals[end-J:end]
        @test all(σ_start .> 0)
    end
    
    @testset "Question 6: Optimization (Small Test)" begin
        # Use smaller dataset for faster testing
        df_small = df[1:200, :]
        X, y, Xfac, asvabs = prepare_factor_matrices(df_small)
        
        # Create starting values
        svals = vcat(
            Xfac\asvabs[:,1],
            Xfac\asvabs[:,2],
            Xfac\asvabs[:,3], 
            Xfac\asvabs[:,4],
            Xfac\asvabs[:,5], 
            Xfac\asvabs[:,6],
            X\y,
            rand(7), 
            0.5*ones(7)
        )
        
        # Test that optimization can run (with fewer iterations)
        td = TwiceDifferentiable(θ -> factor_model(θ, X, Xfac, asvabs, y, 5),
                                 svals; autodiff = :forward)
        
        result = optimize(td, svals, Newton(linesearch = LineSearches.BackTracking()),
                         Optim.Options(g_tol = 1e-3, iterations = 10, show_trace = false))
        
        @test Optim.converged(result) || result.iterations == 10
        @test isfinite(result.minimum)
        @test all(isfinite.(result.minimizer))
    end
    
    #==========================================================================
    # Integration Tests
    ==========================================================================#
    @testset "Integration: Compare Models" begin
        # Run all models and compare R²
        base_result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
        full_result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                                 asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
        
        df_pca = generate_pca!(deepcopy(df))
        pca_result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df_pca)
        
        df_factor = generate_factor!(deepcopy(df))
        factor_result = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), df_factor)
        
        # R² should increase when adding information
        @test r2(full_result) > r2(base_result)
        @test r2(pca_result) > r2(base_result)
        @test r2(factor_result) > r2(base_result)
        
        # PCA and Factor should have similar R²
        @test abs(r2(pca_result) - r2(factor_result)) < 0.05
    end
    
end

println("\n" * "="^80)
println("All tests completed!")
println("="^80)