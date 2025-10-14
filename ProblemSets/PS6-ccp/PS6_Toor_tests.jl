using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

cd(@__DIR__)

include("PS6_Toor_source.jl") 


Random.seed!(12345) 

#Comprehensive Unit tests


@testset "PS6 Rust Model CCP Estimation Tests" begin
    

    @testset "Data Loading and Reshaping" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        
        @test_nowarn df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        
        # Test DataFrame structure
        @test isa(df_long, DataFrame)
        @test :bus_id in names(df_long)
        @test :time in names(df_long)
        @test :Y in names(df_long)
        @test :Odometer in names(df_long)
        @test :Xstate in names(df_long)
        @test :Zst in names(df_long)
        
        # Test dimensions
        n_buses = nrow(df_long) ÷ 20
        @test nrow(df_long) == n_buses * 20
        @test size(Xstate) == (n_buses, 20)
        @test length(Zstate) == n_buses
        @test length(Branded) == n_buses
        
        # Test time variable
        @test minimum(df_long.time) == 1
        @test maximum(df_long.time) == 20
        
        # Test bus_id consistency
        @test length(unique(df_long.bus_id)) == n_buses
        
        # Test Y is binary
        @test all(y -> y in [0, 1], df_long.Y)
        
        # Test no missing values in key columns
        @test !any(ismissing, df_long.Y)
        @test !any(ismissing, df_long.Odometer)
        @test !any(ismissing, df_long.Xstate)
    end
    
    @testset "Flexible Logit Estimation" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, _, _, _ = load_and_reshape_data(url)
        
        @test_nowarn flex_model = estimate_flexible_logit(df_long)
        
        flex_model = estimate_flexible_logit(df_long)
        
        # Test model type and convergence
        @test isa(flex_model, GeneralizedLinearModel)
        @test GLM.converged(flex_model)
        
        # Test predictions are probabilities
        preds = predict(flex_model, df_long)
        @test all(0 .<= preds .<= 1)
        
        # Test model has coefficients
        @test length(coef(flex_model)) > 0
        @test !any(isnan, coef(flex_model))
    end
    
    @testset "State Space Construction" begin
        zval, zbin, xval, xbin, xtran = create_grids()
        
        @test_nowarn state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        # Test DataFrame structure
        @test isa(state_df, DataFrame)
        @test :Odometer in names(state_df)
        @test :RouteUsage in names(state_df)
        @test :Branded in names(state_df)
        @test :time in names(state_df)
        
        # Test dimensions
        @test nrow(state_df) == xbin * zbin
        
        # Test grid construction
        @test length(unique(state_df.Odometer)) == xbin
        @test length(unique(state_df.RouteUsage)) == zbin
        
        # Test initial values
        @test all(state_df.Branded .== 0)
        @test all(state_df.time .== 0)
    end
    
    @testset "Future Values Computation" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, _, _, _ = load_and_reshape_data(url)
        flex_model = estimate_flexible_logit(df_long)
        
        zval, zbin, xval, xbin, xtran = create_grids()
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        β = 0.9
        T = 20
        
        @test_nowarn FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, T, β)
        
        FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, T, β)
        
        # Test array dimensions
        @test size(FV) == (xbin * zbin, 2, T + 1)
        
        # Test boundary conditions
        @test all(FV[:, :, T+1] .== 0)  # Terminal period
        
        # Test non-negativity (since FV = -β * log(p0) and 0 < p0 < 1)
        @test all(FV[:, :, 1:T] .>= 0)
        
        # Test no NaN or Inf values
        @test !any(isnan, FV)
        @test !any(isinf, FV)
    end
    
    @testset "Future Value Mapping" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        flex_model = estimate_flexible_logit(df_long)
        
        zval, zbin, xval, xbin, xtran = create_grids()
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        β = 0.9
        T = 20
        FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, T, β)
        
        @test_nowarn fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
        
        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
        
        # Test dimensions
        @test length(fvt1) == nrow(df_long)
        
        # Test no NaN or Inf values
        @test !any(isnan, fvt1)
        @test !any(isinf, fvt1)
        
        # Test finite values
        @test all(isfinite, fvt1)
    end
    
    @testset "Structural Parameter Estimation" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        flex_model = estimate_flexible_logit(df_long)
        
        zval, zbin, xval, xbin, xtran = create_grids()
        state_df = construct_state_space(xbin, zbin, xval, zval, xtran)
        
        β = 0.9
        T = 20
        FV = compute_future_values(state_df, flex_model, xtran, xbin, zbin, T, β)
        fvt1 = compute_fvt1(df_long, FV, xtran, Xstate, Zstate, xbin, Branded)
        
        @test_nowarn theta_hat = estimate_structural_params(df_long, fvt1)
        
        theta_hat = estimate_structural_params(df_long, fvt1)
        
        # Test model type and convergence
        @test isa(theta_hat, GeneralizedLinearModel)
        @test GLM.converged(theta_hat)
        
        # Test coefficient structure
        coeffs = coef(theta_hat)
        @test length(coeffs) >= 2  # At least intercept + Odometer + Branded
        @test !any(isnan, coeffs)
        
        # Test standard errors exist
        @test !any(isnan, stderror(theta_hat))
    end
    
    @testset "Integration Test - Main Function" begin
        # Capture output to test main function
        @test_nowarn main()
    end
    
    @testset "Edge Cases and Error Handling" begin
        # Test with empty DataFrame
        empty_df = DataFrame()
        @test_throws Exception estimate_flexible_logit(empty_df)
        
        # Test future values with invalid dimensions
        zval, zbin, xval, xbin, xtran = create_grids()
        @test_throws BoundsError compute_future_values(DataFrame(), nothing, xtran, 0, 0, 20, 0.9)
        
        # Test state space construction with invalid parameters
        @test_throws Exception construct_state_space(0, 1, [1.0], [1.0], zeros(1,1))
    end
    
    @testset "Mathematical Properties" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS5-ddc/busdata.csv"
        df_long, Xstate, Zstate, Branded = load_and_reshape_data(url)
        
        # Test that transition probabilities sum to 1
        zval, zbin, xval, xbin, xtran = create_grids()
        row_sums = sum(xtran, dims=2)
        @test all(abs.(row_sums .- 1.0) .< 1e-10)  # Should sum to 1 within numerical precision
        
        # Test that state indices are valid
        @test all(1 .<= Xstate .<= xbin)
        @test all(1 .<= Zstate .<= zbin)
        @test all(Branded .∈ Ref([0, 1]))
    end
end
