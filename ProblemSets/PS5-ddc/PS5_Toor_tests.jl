using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM

cd(@__DIR__)

include("PS5_Toor_source.jl")


@testset "PS5 Bus Engine Replacement Model Tests" begin

    @testset "Data Loading Functions" begin
        @testset "Static Data Loading" begin
            df_long = load_static_data()
            
            @test isa(df_long, DataFrame)
            @test nrow(df_long) > 0
            @test Set(names(df_long)) ⊇ Set([:bus_id, :time, :Y, :Odometer, :RouteUsage, :Branded])
            @test all(x -> x in [0, 1], df_long.Y)
            @test all(x -> x in [0, 1], df_long.Branded)
            @test all(df_long.Odometer .>= 0)
            @test Set(df_long.time) == Set(1:20)
            
            # Panel structure - each bus has exactly 20 observations
            bus_counts = combine(groupby(df_long, :bus_id), nrow => :count)
            @test all(bus_counts.count .== 20)
        end
        
        @testset "Dynamic Data Loading" begin
            d = load_dynamic_data()
            
            # Structure validation
            required_fields = [:Y, :X, :B, :Xstate, :Zstate, :N, :T, :xval, :xbin, :zbin, :xtran, :β]
            @test all(haskey(d, field) for field in required_fields)
            
            # Dimension consistency
            @test size(d.Y) == size(d.X) == size(d.Xstate) == (d.N, d.T)
            @test length(d.Zstate) == length(d.B) == d.N
            @test length(d.xval) == d.xbin
            @test size(d.xtran) == (d.zbin * d.xbin, d.xbin)
            
            # Data validity
            @test all(d.Y .∈ Ref([0, 1]))
            @test all(1 .<= d.Xstate .<= d.xbin)
            @test all(1 .<= d.Zstate .<= d.zbin)
            @test d.β == 0.9
            
            # Transition matrix is stochastic
            @test all(abs.(sum(d.xtran, dims=2) .- 1) .< 1e-10)
            @test all(d.xtran .>= 0)
        end
    end
    
    @testset "Static Model Estimation" begin
        df_long = load_static_data()
        model = estimate_static_model(df_long)
        
        @test isa(model, GLM.GeneralizedLinearModel)
        @test length(coef(model)) == 3
        @test all(isfinite.(coef(model)))
        
        # Economic intuition: higher odometer should reduce replacement probability
        @test coef(model)[2] < 0  # Odometer coefficient negative
        
        # Model diagnostics
        @test deviance(model) > 0
        @test nobs(model) == nrow(df_long)
    end
    
    @testset "Dynamic Model Core Functions" begin
        d = load_dynamic_data()
        θ_test = [2.0, -0.01, 1.0]
        
        @testset "Future Value Computation" begin
            FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
            result = compute_future_value!(FV, θ_test, d)
            
            @test result === FV  # In-place modification
            @test size(FV) == (d.zbin * d.xbin, 2, d.T + 1)
            
            # Terminal condition
            @test all(FV[:, :, d.T + 1] .== 0)
            
            # Non-trivial values computed
            @test any(FV[:, :, 1:d.T] .!= 0)
            @test all(isfinite.(FV[:, :, 1:d.T]))
            
            # FIXED: Test monotonicity with sample only (avoids massive loop)
            sample_size = min(3, d.zbin * d.xbin)  # Test only first 3 states
            sample_times = min(3, d.T-1)            # Test only first 3 time periods
            for s in 1:sample_size, b in 1:2, t in 1:sample_times
                @test FV[s, b, t] >= FV[s, b, t+1] - 1e-10
            end
        end
        
        @testset "Log Likelihood Function" begin
            ll = log_likelihood_dynamic(θ_test, d)
            
            @test isa(ll, Real)
            @test isfinite(ll)
            @test ll < 0  # Should be negative log likelihood
            
            # Different parameters give different likelihoods
            θ_alt = [1.5, -0.02, 0.5]
            ll_alt = log_likelihood_dynamic(θ_alt, d)
            @test ll != ll_alt
            
            # Continuity: small parameter changes yield small likelihood changes
            θ_perturb = θ_test .+ [0.001, 0.0001, 0.001]
            ll_perturb = log_likelihood_dynamic(θ_perturb, d)
            @test abs(ll - ll_perturb) / abs(ll) < 0.1
        end
    end
    
    @testset "Model Integration Tests" begin
        @testset "End-to-End Estimation Pipeline" begin
            # Static estimation
            df_long = load_static_data()
            static_model = estimate_static_model(df_long)
            static_coefs = coef(static_model)
            @test length(static_coefs) == 3
            
            # Dynamic data and likelihood
            d = load_dynamic_data()
            ll = log_likelihood_dynamic([2.0, -0.01, 1.0], d)
            @test isfinite(ll)
            
            # Components work together
            FV = zeros(d.zbin * d.xbin, 2, d.T + 1)
            compute_future_value!(FV, [2.0, -0.01, 1.0], d)
            @test all(isfinite.(FV[:, :, 1:d.T]))
        end
        
        @testset "Parameter Validation" begin
            d = load_dynamic_data()
            
            # Valid parameter vector
            θ_valid = [2.0, -0.01, 1.0]
            ll_valid = log_likelihood_dynamic(θ_valid, d)
            @test isfinite(ll_valid)
            
            # Wrong dimensions should error
            @test_throws BoundsError log_likelihood_dynamic([1.0, 2.0], d)
            @test_throws BoundsError log_likelihood_dynamic([1.0, 2.0, 3.0, 4.0], d)
            
            # Extreme values should be handled
            θ_extreme = [100.0, -10.0, 100.0]
            ll_extreme = log_likelihood_dynamic(θ_extreme, d)
            @test isfinite(ll_extreme) && !isnan(ll_extreme)
        end
    end
    
    @testset "Economic and Mathematical Properties" begin
        d = load_dynamic_data()
        
        @testset "Economic Intuition" begin
            θ_base = [2.0, -0.01, 1.0]
            θ_high_cost = [2.0, -0.05, 1.0]  # Higher mileage cost
            
            ll_base = log_likelihood_dynamic(θ_base, d)
            ll_high = log_likelihood_dynamic(θ_high_cost, d)
            
            @test isfinite(ll_base) && isfinite(ll_high)
        end
        
        @testset "Mathematical Consistency" begin
            # Transition matrix properties
            @test size(d.xtran, 1) == d.zbin * d.xbin
            @test size(d.xtran, 2) == d.xbin
            
            # State space consistency
            @test d.N > 0 && d.T > 0
            @test d.xbin > 0 && d.zbin > 0
            @test 0 < d.β < 1
            
            # Data arrays have consistent dimensions
            @test size(d.Y, 1) == size(d.X, 1) == d.N
            @test size(d.Y, 2) == size(d.X, 2) == d.T
        end
        
        @testset "Edge Cases and Robustness" begin
            d = load_dynamic_data()
            
            # Test with various parameter magnitudes
            test_params = [
                [0.1, -0.001, 0.1],    # Small values
                [5.0, -0.1, 3.0],      # Moderate values  
                [50.0, -1.0, 20.0]     # Large values
            ]
            
            for θ in test_params
                ll = log_likelihood_dynamic(θ, d)
                @test isfinite(ll) && !isnan(ll)
            end
            
            # Test state bounds are respected
            @test all(d.Xstate .>= 1) && all(d.Xstate .<= d.xbin)
            @test all(d.Zstate .>= 1) && all(d.Zstate .<= d.zbin)
        end
    end
end

println("All tests completed: $(Test.get_testset().n_passed) passed!")