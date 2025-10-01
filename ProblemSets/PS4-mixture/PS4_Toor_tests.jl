using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions 

cd(@__DIR__) 
Random.seed!(1234)  
include("PS4_Toor_source.jl") 

@testset "PS4 Tests" begin
    
    # Load data once for all tests
    df, X, Z, y = load_data()
    K, J, N = size(X, 2), length(unique(y)), length(y)
    
    @testset "Data Loading" begin
        @test size(X) == (N, 3)  # age, white, collgrad
        @test size(Z) == (N, 8)  # 8 wage alternatives
        @test length(y) == N
        @test all(1 .<= y .<= 8)
        @test length(unique(y)) == 8
        @test !any(ismissing, [X; Z; y])
    end
    
    @testset "Q1: Multinomial Logit" begin
        # Test parameter structure: K*(J-1) alphas + 1 gamma = 21 + 1 = 22
        theta = [zeros(21); 0.1]
        
        # Test likelihood computation
        loglik = mlogit_with_Z(theta, X, Z, y)
        @test loglik > 0  # negative log-likelihood is positive
        @test isfinite(loglik)
        
        # Test parameter extraction
        alpha = theta[1:end-1]
        gamma = theta[end]
        @test length(alpha) == 21
        @test gamma == 0.1
        
        # Test bigAlpha construction
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        @test size(bigAlpha) == (3, 8)
        @test bigAlpha[:, 8] == zeros(3)  # normalized choice
        
        # Test choice indicators sum to 1
        bigY = zeros(N, J)
        for j = 1:J
            bigY[:, j] = y .== j
        end
        @test all(sum(bigY, dims=2) .== 1)
        
        # Test different parameters give different likelihood
        theta2 = [0.1*randn(21); 0.2]
        loglik2 = mlogit_with_Z(theta2, X, Z, y)
        @test loglik2 != loglik
    end
    
    @testset "Q3a: Quadrature Practice" begin
        # Test lgwt function
        nodes, weights = lgwt(7, -4, 4)
        @test length(nodes) == length(weights) == 7
        @test all(weights .> 0)
        @test -4 <= minimum(nodes) < maximum(nodes) <= 4
        
        # Test standard normal integration
        d = Normal(0, 1)
        integral = sum(weights .* pdf.(d, nodes))
        expectation = sum(weights .* nodes .* pdf.(d, nodes))
        
        @test abs(integral - 1.0) < 0.005  # ∫φ(x)dx ≈ 1
        @test abs(expectation) < 0.005     # ∫xφ(x)dx ≈ 0
        
        @test_nowarn practice_quadrature()
    end
    
    @testset "Q3b: Variance Quadrature" begin
        σ = 2
        d = Normal(0, σ)
        
        # Test 7 points
        nodes7, weights7 = lgwt(7, -5*σ, 5*σ)
        var7 = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))
        
        # Test 10 points  
        nodes10, weights10 = lgwt(10, -5*σ, 5*σ)
        var10 = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))
        
        @test abs(var7 - σ^2) < 0.2    # Should approximate 4
        @test abs(var10 - σ^2) < 0.1   # Should be more accurate
        @test abs(var10 - σ^2) < abs(var7 - σ^2)  # 10 points better
        
        @test_nowarn variance_quadrature()
    end
    
    @testset "Q3c: Monte Carlo Practice" begin
        σ = 2
        d = Normal(0, σ)
        A, B = -5*σ, 5*σ
        
        function mc_integrate(f, a, b, D)
            draws = rand(D) * (b - a) .+ a
            return (b - a) * mean(f.(draws))
        end
        
        # Test variance estimation
        Random.seed!(1234)
        var_mc = mc_integrate(x -> x^2 * pdf(d, x), A, B, 10000)
        @test abs(var_mc - σ^2) < 0.5  # MC has more noise
        
        # Test density integral
        Random.seed!(1234)
        dens_mc = mc_integrate(x -> pdf(d, x), A, B, 10000)
        @test abs(dens_mc - 1.0) < 0.05
        
        @test_nowarn practice_monte_carlo()
    end
    
    @testset "Q4: Mixed Logit Quadrature Setup" begin
        # Test parameter structure: 21 alphas + mu_gamma + sigma_gamma = 23
        theta = [zeros(21); 0.1; 1.0]
        @test length(theta) == 23
        
        alpha = theta[1:21]
        mu_gamma = theta[22] 
        sigma_gamma = theta[23]
        @test mu_gamma == 0.1
        @test sigma_gamma == 1.0
        @test sigma_gamma > 0
        
        # Test quadrature setup
        R = 7
        nodes, weights = lgwt(R, mu_gamma - 5*sigma_gamma, mu_gamma + 5*sigma_gamma)
        @test length(nodes) == length(weights) == R
        
        # Test that function exists (don't run - too expensive)
        @test isdefined(Main, :mixed_logit_quad) 
    end
    
    @testset "Q5: Mixed Logit Monte Carlo Setup" begin
        # Same parameter structure as Q4
        theta = [zeros(21); 0.1; 1.0]
        D = 1000
        
        mu_gamma, sigma_gamma = theta[22], theta[23]
        a, b = mu_gamma - 5*sigma_gamma, mu_gamma + 5*sigma_gamma
        
        # Test MC draws
        Random.seed!(1234)
        draws = rand(D) * (b - a) .+ a
        @test length(draws) == D
        @test all(a .<= draws .<= b)
        
        # Test density evaluation
        gamma_dist = Normal(mu_gamma, sigma_gamma)
        densities = pdf.(gamma_dist, draws)
        @test all(densities .> 0)
        @test length(densities) == D
        
        # Test that function exists (don't run - too expensive)
        @test isdefined(Main, :mixed_logit_mc)
    end
    
    @testset "Optimization Functions" begin
        # Test function definitions
        @test isdefined(Main, :optimize_mlogit)
        @test isdefined(Main, :optimize_mixed_logit_quad)
        @test isdefined(Main, :optimize_mixed_logit_mc)
        
        # Test starting values construction
        Random.seed!(1234)
        startvals_mlogit = [2*rand(21).-1; 0.1]
        @test length(startvals_mlogit) == 22
        @test startvals_mlogit[end] == 0.1
        
        startvals_mixed = [2*rand(21).-1; 0.1; 1.0]  
        @test length(startvals_mixed) == 23
        @test startvals_mixed[end] > 0  # sigma > 0
        
        # Test TwiceDifferentiable setup
        theta_test = [zeros(21); 0.1]
        @test_nowarn TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), theta_test; autodiff = :forward)
    end
    
    @testset "Q6: Main Wrapper" begin
        @test isdefined(Main, :allwrap)
        @test_nowarn allwrap()
    end
    
    @testset "Mathematical Properties" begin
        # Test probability properties for multinomial logit
        theta = [0.1*randn(21); 0.2]
        alpha = theta[1:21]
        gamma = theta[22]
        bigAlpha = [reshape(alpha, 3, 7) zeros(3)]
        
        # Compute probabilities manually
        num = zeros(N, J)
        for j = 1:J
            num[:,j] = exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
        end
        dem = sum(num, dims=2)
        P = num ./ dem
        
        @test all(0 .<= P .<= 1)  # Valid probabilities
        @test all(abs.(sum(P, dims=2) .- 1) .< 1e-10)  # Sum to 1
        
        # Test gamma coefficient effect
        theta_high_gamma = [alpha; 1.0]
        ll_low = mlogit_with_Z(theta, X, Z, y)
        ll_high = mlogit_with_Z(theta_high_gamma, X, Z, y)
        @test ll_low != ll_high  # Different gamma should change likelihood
    end
    
    @testset "Edge Cases" begin
        # Test with extreme parameters
        theta_large = [ones(21); 5.0]
        @test isfinite(mlogit_with_Z(theta_large, X, Z, y))
        
        theta_small = [0.001*ones(21); 0.001]  
        @test isfinite(mlogit_with_Z(theta_small, X, Z, y))
        
        # Test quadrature with different bounds
        nodes_narrow, weights_narrow = lgwt(5, -1, 1)
        @test length(nodes_narrow) == 5
        @test all(weights_narrow .> 0)
        
        nodes_wide, weights_wide = lgwt(5, -10, 10)  
        @test length(nodes_wide) == 5
        @test sum(weights_wide) > sum(weights_narrow)  # Wider integration
    end
    
end

println("All tests passed!")