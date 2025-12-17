# ==============================================================================
# HEALTH INSURANCE AND SANITATION INVESTMENT STRUCTURAL MODEL
# Complete Source File - All Functions
# Following Mike Keane's Structural Modeling Methodology
# ==============================================================================

# ==============================================================================
# 1. PARAMETER STRUCTURE
# ==============================================================================

mutable struct ModelParams
    # Period 2 parameters (7)
    α_c::Float64           # Consumption taste coefficient
    α_1::Float64           # Basic care alternative-specific constant
    α_2::Float64           # Intensive care ASC (α_0 = 0 normalized)
    γ::Float64             # Health recovery value parameter
    r_1::Float64           # Basic care treatment effectiveness
    r_2::Float64           # Intensive care effectiveness (r_0 = 0)
    δ::Float64             # Sickness disutility parameter
    
    # Period 1 parameters (2)
    k::Float64             # Sanitation installation cost
    β::Float64             # Discount factor
    
    # Health production parameters (2)
    p_0::Float64           # Individual illness probability without sanitation
    p_1::Float64           # Individual illness probability with sanitation
    
    # Exogenous parameters
    W_1::Float64           # Period 1 income
    W_2::Float64           # Period 2 income
    M_1::Float64           # Basic care medical cost
    M_2::Float64           # Intensive care medical cost
    n::Int                 # Household size (number of members)
    θ::Float64             # Insurance generosity (coinsurance rate)
    T_slope::Float64       # Transfer function slope: T(θ) = T_slope * θ
end

"""
Constructor with default calibrated values matching low-income settings

Key calibration notes:
- δ = 6.0: Sickness disutility calibrated so net effect of best treatment 
  when sick is -1.5 (worse than healthy). Calculation: -δ + γ*r_2 = -6.0 + 5.0*0.9 = -1.5
- This ensures sick people seek treatment while maintaining health > sickness hierarchy
"""
function ModelParams(;
    α_c = 1.0,
    α_1 = -2.5,
    α_2 = -3.5,
    γ = 5.0,
    r_1 = 0.5,
    r_2 = 0.9,
    δ = 6.0,        # Sickness disutility (see calibration note above)
    k = 500.0,
    β = 0.9,
    p_0 = 0.30,
    p_1 = 0.10,
    W_1 = 5000.0,
    W_2 = 5000.0,
    M_1 = 1000.0,
    M_2 = 3000.0,
    n = 4,
    θ = 0.7,
    T_slope = 500.0  # Increased from 200 to test budget relief effect
)
    return ModelParams(α_c, α_1, α_2, γ, r_1, r_2, δ, k, β, p_0, p_1, 
                      W_1, W_2, M_1, M_2, n, θ, T_slope)
end

"""
Print parameter summary
"""
function print_parameters(p::ModelParams)
    println("\n" * "="^70)
    println("MODEL PARAMETERS")
    println("="^70)
    
    println("\nPeriod 2 Preferences:")
    println("  α_c (consumption taste):        $(p.α_c)")
    println("  α_1 (basic care ASC):           $(p.α_1)")
    println("  α_2 (intensive care ASC):       $(p.α_2)")
    println("  γ (health recovery value):      $(p.γ)")
    println("  r_1 (basic effectiveness):      $(p.r_1)")
    println("  r_2 (intensive effectiveness):  $(p.r_2)")
    println("  δ (sickness disutility):        $(p.δ)")
    
    println("\nPeriod 1 Preferences:")
    println("  k (sanitation cost):            $(p.k)")
    println("  β (discount factor):            $(p.β)")
    
    println("\nHealth Production:")
    println("  p_0 (illness risk, no sanit.):  $(p.p_0)")
    println("  p_1 (illness risk, w/ sanit.):  $(p.p_1)")
    println("  n (household size):             $(p.n)")
    
    println("\nExogenous Variables:")
    println("  W_1 (period 1 income):          $(p.W_1)")
    println("  W_2 (period 2 income):          $(p.W_2)")
    println("  M_1 (basic care cost):          $(p.M_1)")
    println("  M_2 (intensive care cost):      $(p.M_2)")
    println("  θ (insurance generosity):       $(p.θ)")
    println("  T_slope (transfer slope):       $(p.T_slope)")
    
    println("\nNormalizations:")
    println("  α_0 = 0 (no treatment ASC)")
    println("  r_0 = 0 (no treatment effectiveness)")
    println("  M_0 = 0 (no treatment cost)")
    println("="^70)
end

# ==============================================================================
# 2. HEALTH PRODUCTION FUNCTIONS
# ==============================================================================

"""
Transfer function: T(θ) = T_slope * θ
Government provides direct cash transfer based on insurance generosity
"""
function transfer(θ::Float64, p::ModelParams)
    return p.T_slope * θ
end

"""
Individual illness probability given sanitation choice
Returns p_0 if no sanitation (S_1=0), p_1 if sanitation (S_1=1)
"""
function individual_illness_prob(S_1::Int, p::ModelParams)
    return S_1 == 0 ? p.p_0 : p.p_1
end

"""
Household-level illness probability: Pr(H_2 = 1 | S_1)
At least one of n members gets sick: 1 - (1 - p_{S_1})^n
"""
function household_illness_prob(S_1::Int, p::ModelParams)
    p_individual = individual_illness_prob(S_1, p)
    return 1.0 - (1.0 - p_individual)^p.n
end

# ==============================================================================
# 3. PERIOD 2 COMPONENTS
# ==============================================================================

"""
Medical costs by treatment level Q_2 ∈ {0,1,2}
M(0) = 0, M(1) = M_1, M(2) = M_2
"""
function medical_cost(Q_2::Int, p::ModelParams)
    if Q_2 == 0
        return 0.0
    elseif Q_2 == 1
        return p.M_1
    elseif Q_2 == 2
        return p.M_2
    else
        error("Q_2 must be in {0,1,2}")
    end
end

"""
Treatment effectiveness by level Q_2
r(0) = 0, r(1) = r_1, r(2) = r_2
"""
function recovery_effect(Q_2::Int, p::ModelParams)
    if Q_2 == 0
        return 0.0
    elseif Q_2 == 1
        return p.r_1
    elseif Q_2 == 2
        return p.r_2
    else
        error("Q_2 must be in {0,1,2}")
    end
end

"""
Alternative-specific constant by healthcare level Q_2
α_0 = 0 (normalized), α_1, α_2
"""
function healthcare_asc(Q_2::Int, p::ModelParams)
    if Q_2 == 0
        return 0.0  # Normalized
    elseif Q_2 == 1
        return p.α_1
    elseif Q_2 == 2
        return p.α_2
    else
        error("Q_2 must be in {0,1,2}")
    end
end

"""
Period 2 consumption: c_2 = W_2 - H_2 * (1-θ) * M(Q_2)
Out-of-pocket medical costs only paid if sick (H_2=1)
Insurance covers θ fraction, household pays (1-θ) fraction
"""
function period2_consumption(Q_2::Int, H_2::Int, θ::Float64, p::ModelParams)
    M_Q = medical_cost(Q_2, p)
    oop_cost = H_2 * (1.0 - θ) * M_Q  # Out-of-pocket only if sick
    c_2 = p.W_2 - oop_cost
    
    # Ensure positive consumption
    if c_2 <= 0.0
        c_2 = 1e-6  # Small positive number to avoid log(0)
    end
    
    return c_2
end

"""
Period 2 systematic utility: v̄_2(Q_2 | S_1, H_2, θ)
= α_c * ln(c_2) + α_{Q_2} - δH_2 + γ * r(Q_2) * H_2

Components:
- α_c * ln(c_2): Consumption utility
- α_{Q_2}: Alternative-specific constant (baseline preference)
- -δH_2: Sickness disutility (only when sick)
- γ * r(Q_2) * H_2: Health recovery benefit (only when sick)
"""
function period2_systematic_utility(Q_2::Int, S_1::Int, H_2::Int, θ::Float64, p::ModelParams)
    # Consumption utility
    c_2 = period2_consumption(Q_2, H_2, θ, p)
    u_consumption = p.α_c * log(c_2)
    
    # Alternative-specific constant
    u_asc = healthcare_asc(Q_2, p)
    
    # Sickness disutility
    u_sickness = -p.δ * H_2
    
    # Health recovery benefit (only matters if sick)
    u_health = p.γ * recovery_effect(Q_2, p) * H_2
    
    return u_consumption + u_asc + u_sickness + u_health
end

"""
Period 2 continuation value (Emax): V_2(S_1, H_2, θ)
= log(Σ_{q=0}^{2} exp(v̄_2(q | S_1, H_2, θ)))

Type I Extreme Value integration formula
"""
function period2_continuation_value(S_1::Int, H_2::Int, θ::Float64, p::ModelParams)
    # Compute systematic utilities for all Q_2 ∈ {0,1,2}
    v_0 = period2_systematic_utility(0, S_1, H_2, θ, p)
    v_1 = period2_systematic_utility(1, S_1, H_2, θ, p)
    v_2 = period2_systematic_utility(2, S_1, H_2, θ, p)
    
    # Type I EV Emax formula (log-sum-exp)
    max_v = maximum([v_0, v_1, v_2])  # For numerical stability
    V2 = max_v + log(exp(v_0 - max_v) + exp(v_1 - max_v) + exp(v_2 - max_v))
    
    return V2
end

"""
Period 2 expected value before H_2 realizes: V̄_2(S_1, θ)
= Σ_{H_2 ∈ {0,1}} Pr(H_2 | S_1) * V_2(S_1, H_2, θ)

Integrates over health uncertainty
"""
function period2_expected_value(S_1::Int, θ::Float64, p::ModelParams)
    # Probability of illness given sanitation choice
    pr_sick = household_illness_prob(S_1, p)
    pr_healthy = 1.0 - pr_sick
    
    # Continuation values for each health state
    V_healthy = period2_continuation_value(S_1, 0, θ, p)
    V_sick = period2_continuation_value(S_1, 1, θ, p)
    
    # Expected value
    return pr_healthy * V_healthy + pr_sick * V_sick
end

"""
Period 2 choice probabilities: P(Q_2 = q | S_1, H_2, θ)
Logit formula from Type I EV shocks
Returns vector [P(Q_2=0), P(Q_2=1), P(Q_2=2)]
"""
function period2_choice_probs(S_1::Int, H_2::Int, θ::Float64, p::ModelParams)
    v_0 = period2_systematic_utility(0, S_1, H_2, θ, p)
    v_1 = period2_systematic_utility(1, S_1, H_2, θ, p)
    v_2 = period2_systematic_utility(2, S_1, H_2, θ, p)
    
    # Numerical stability: subtract max before exponentiating
    max_v = maximum([v_0, v_1, v_2])
    exp_v0 = exp(v_0 - max_v)
    exp_v1 = exp(v_1 - max_v)
    exp_v2 = exp(v_2 - max_v)
    
    denom = exp_v0 + exp_v1 + exp_v2
    
    return [exp_v0/denom, exp_v1/denom, exp_v2/denom]
end

# ==============================================================================
# 4. PERIOD 1 COMPONENTS
# ==============================================================================

"""
Period 1 consumption: c_1 = W_1 + T(θ) - k*S_1
Includes government transfer T(θ) and sanitation cost k if S_1=1
"""
function period1_consumption(S_1::Int, θ::Float64, p::ModelParams)
    T_θ = transfer(θ, p)
    c_1 = p.W_1 + T_θ - p.k * S_1
    
    # Ensure positive consumption
    if c_1 <= 0.0
        c_1 = 1e-6
    end
    
    return c_1
end

"""
Period 1 systematic utility: v̄_1(S_1, θ)
= ln(c_1) + β * V̄_2(S_1, θ)

Components:
- ln(c_1): Current consumption utility
- β * V̄_2(S_1, θ): Discounted continuation value
"""
function period1_systematic_utility(S_1::Int, θ::Float64, p::ModelParams)
    c_1 = period1_consumption(S_1, θ, p)
    
    # Current consumption utility
    u_consumption = log(c_1)
    
    # Discounted continuation value
    V2_bar = period2_expected_value(S_1, θ, p)
    u_continuation = p.β * V2_bar
    
    return u_consumption + u_continuation
end

"""
Net gain from sanitation: Δ(θ) = v̄_1(1, θ) - v̄_1(0, θ)

Decomposes into:
- Liquidity effect: ln[(W_1 + T(θ) - k)/(W_1 + T(θ))]
- Insurance effect: β[V̄_2(1, θ) - V̄_2(0, θ)]
"""
function sanitation_net_gain(θ::Float64, p::ModelParams)
    v_1 = period1_systematic_utility(1, θ, p)
    v_0 = period1_systematic_utility(0, θ, p)
    return v_1 - v_0
end

"""
Period 1 choice probability: P(S_1 = 1 | θ)
Logit formula: 1 / (1 + exp(-Δ(θ)))
"""
function period1_choice_prob(θ::Float64, p::ModelParams)
    Δ = sanitation_net_gain(θ, p)
    return 1.0 / (1.0 + exp(-Δ))
end

"""
Decompose sanitation net gain into channels
Returns (total, liquidity_effect, insurance_effect)
"""
function decompose_sanitation_gain(θ::Float64, p::ModelParams)
    # Total net gain
    Δ_total = sanitation_net_gain(θ, p)
    
    # Liquidity effect (period 1 consumption cost)
    T_θ = transfer(θ, p)
    liquidity = log((p.W_1 + T_θ - p.k) / (p.W_1 + T_θ))
    
    # Insurance effect (continuation value difference)
    V2_with = period2_expected_value(1, θ, p)
    V2_without = period2_expected_value(0, θ, p)
    insurance = p.β * (V2_with - V2_without)
    
    return (total = Δ_total, liquidity = liquidity, insurance = insurance)
end

# ==============================================================================
# 5. SIMULATION FUNCTIONS
# ==============================================================================

"""
Simulate N households under given parameters
Returns DataFrame with columns: S_1, H_2, Q_2, c_1, c_2, ν_S1, ε_Q2, utilities
"""
function simulate_data(N::Int, p::ModelParams; seed::Int=123)
    Random.seed!(seed)
    
    # Initialize storage
    S_1_vec = zeros(Int, N)
    H_2_vec = zeros(Int, N)
    Q_2_vec = zeros(Int, N)
    c_1_vec = zeros(Float64, N)
    c_2_vec = zeros(Float64, N)
    
    # Store shocks for diagnostics
    ν_0_vec = zeros(Float64, N)
    ν_1_vec = zeros(Float64, N)
    ε_0_vec = zeros(Float64, N)
    ε_1_vec = zeros(Float64, N)
    ε_2_vec = zeros(Float64, N)
    
    # Get current insurance generosity
    θ = p.θ
    
    for i in 1:N
        # ======================================================================
        # PERIOD 1: SANITATION CHOICE
        # ======================================================================
        
        # Step 1: Draw period 1 preference shocks (Gumbel distribution)
        ν_0 = rand(Gumbel(0, 1))
        ν_1 = rand(Gumbel(0, 1))
        ν_0_vec[i] = ν_0
        ν_1_vec[i] = ν_1
        
        # Step 2: Compute period 1 utilities
        v̄_0 = period1_systematic_utility(0, θ, p)
        v̄_1 = period1_systematic_utility(1, θ, p)
        
        U_0 = v̄_0 + ν_0
        U_1 = v̄_1 + ν_1
        
        # Step 3: Choose optimal sanitation
        S_1 = U_1 > U_0 ? 1 : 0
        S_1_vec[i] = S_1
        
        # Record period 1 consumption
        c_1_vec[i] = period1_consumption(S_1, θ, p)
        
        # ======================================================================
        # PERIOD 2: HEALTH REALIZATION AND HEALTHCARE CHOICE
        # ======================================================================
        
        # Step 4: Draw health realization
        pr_sick = household_illness_prob(S_1, p)
        H_2 = rand() < pr_sick ? 1 : 0
        H_2_vec[i] = H_2
        
        # Step 5: Draw period 2 preference shocks
        ε_0 = rand(Gumbel(0, 1))
        ε_1 = rand(Gumbel(0, 1))
        ε_2 = rand(Gumbel(0, 1))
        ε_0_vec[i] = ε_0
        ε_1_vec[i] = ε_1
        ε_2_vec[i] = ε_2
        
        # Step 6: Compute period 2 utilities
        v̄_Q0 = period2_systematic_utility(0, S_1, H_2, θ, p)
        v̄_Q1 = period2_systematic_utility(1, S_1, H_2, θ, p)
        v̄_Q2 = period2_systematic_utility(2, S_1, H_2, θ, p)
        
        U_Q0 = v̄_Q0 + ε_0
        U_Q1 = v̄_Q1 + ε_1
        U_Q2 = v̄_Q2 + ε_2
        
        # Step 7: Choose optimal healthcare
        utilities = [U_Q0, U_Q1, U_Q2]
        Q_2 = argmax(utilities) - 1  # Convert to 0,1,2
        Q_2_vec[i] = Q_2
        
        # Record period 2 consumption
        c_2_vec[i] = period2_consumption(Q_2, H_2, θ, p)
    end
    
    # Return comprehensive DataFrame
    return DataFrame(
        id = 1:N,
        S_1 = S_1_vec,
        H_2 = H_2_vec,
        Q_2 = Q_2_vec,
        c_1 = c_1_vec,
        c_2 = c_2_vec,
        ν_0 = ν_0_vec,
        ν_1 = ν_1_vec,
        ε_0 = ε_0_vec,
        ε_1 = ε_1_vec,
        ε_2 = ε_2_vec
    )
end

# ==============================================================================
# 6. VALIDATION FUNCTIONS
# ==============================================================================

"""
Compute comprehensive summary statistics from simulated data
"""
function compute_summary_stats(df::DataFrame, p::ModelParams)
    stats = Dict{String, Float64}()
    
    # Overall statistics
    stats["N"] = nrow(df)
    
    # Sanitation adoption
    stats["sanitation_rate"] = mean(df.S_1)
    
    # Health outcomes
    stats["overall_illness_rate"] = mean(df.H_2)
    
    # Health outcomes by sanitation
    df_no_san = df[df.S_1 .== 0, :]
    df_yes_san = df[df.S_1 .== 1, :]
    
    stats["N_no_sanitation"] = nrow(df_no_san)
    stats["N_yes_sanitation"] = nrow(df_yes_san)
    
    stats["illness_rate_no_san"] = isempty(df_no_san) ? NaN : mean(df_no_san.H_2)
    stats["illness_rate_yes_san"] = isempty(df_yes_san) ? NaN : mean(df_yes_san.H_2)
    
    # Healthcare choices overall
    stats["no_treatment_rate"] = mean(df.Q_2 .== 0)
    stats["basic_care_rate"] = mean(df.Q_2 .== 1)
    stats["intensive_care_rate"] = mean(df.Q_2 .== 2)
    
    # Healthcare choices by health status
    df_healthy = df[df.H_2 .== 0, :]
    df_sick = df[df.H_2 .== 1, :]
    
    stats["N_healthy"] = nrow(df_healthy)
    stats["N_sick"] = nrow(df_sick)
    
    if !isempty(df_healthy)
        stats["no_treatment_healthy"] = mean(df_healthy.Q_2 .== 0)
        stats["basic_care_healthy"] = mean(df_healthy.Q_2 .== 1)
        stats["intensive_care_healthy"] = mean(df_healthy.Q_2 .== 2)
    else
        stats["no_treatment_healthy"] = NaN
        stats["basic_care_healthy"] = NaN
        stats["intensive_care_healthy"] = NaN
    end
    
    if !isempty(df_sick)
        stats["no_treatment_sick"] = mean(df_sick.Q_2 .== 0)
        stats["basic_care_sick"] = mean(df_sick.Q_2 .== 1)
        stats["intensive_care_sick"] = mean(df_sick.Q_2 .== 2)
        stats["any_treatment_sick"] = mean(df_sick.Q_2 .> 0)
    else
        stats["no_treatment_sick"] = NaN
        stats["basic_care_sick"] = NaN
        stats["intensive_care_sick"] = NaN
        stats["any_treatment_sick"] = NaN
    end
    
    # Consumption
    stats["mean_c1"] = mean(df.c_1)
    stats["std_c1"] = std(df.c_1)
    stats["mean_c2"] = mean(df.c_2)
    stats["std_c2"] = std(df.c_2)
    
    # Out-of-pocket medical spending
    medical_spending = zeros(nrow(df))
    for i in 1:nrow(df)
        if df.H_2[i] == 1
            medical_spending[i] = (1.0 - p.θ) * medical_cost(df.Q_2[i], p)
        end
    end
    stats["mean_oop_medical"] = mean(medical_spending)
    stats["mean_oop_medical_sick"] = isempty(df_sick) ? NaN : mean(medical_spending[df.H_2 .== 1])
    
    return stats
end

"""
Print summary statistics in organized format
"""
function print_summary_stats(stats::Dict{String, Float64})
    println("\n" * "="^70)
    println("SUMMARY STATISTICS")
    println("="^70)
    
    println("\nSample Size:")
    @printf("  N = %.0f\n", stats["N"])
    @printf("  No sanitation: %.0f (%.1f%%)\n", stats["N_no_sanitation"], 
            100 * stats["N_no_sanitation"] / stats["N"])
    @printf("  With sanitation: %.0f (%.1f%%)\n", stats["N_yes_sanitation"],
            100 * stats["N_yes_sanitation"] / stats["N"])
    
    println("\nSanitation and Health:")
    @printf("  Sanitation adoption rate: %.3f\n", stats["sanitation_rate"])
    @printf("  Overall illness rate: %.3f\n", stats["overall_illness_rate"])
    @printf("  Illness rate (no sanit.): %.3f\n", stats["illness_rate_no_san"])
    @printf("  Illness rate (w/ sanit.): %.3f\n", stats["illness_rate_yes_san"])
    
    println("\nHealthcare Utilization (Overall):")
    @printf("  No treatment: %.3f\n", stats["no_treatment_rate"])
    @printf("  Basic care: %.3f\n", stats["basic_care_rate"])
    @printf("  Intensive care: %.3f\n", stats["intensive_care_rate"])
    
    println("\nHealthcare Utilization (Healthy, N=$(stats["N_healthy"])):")
    @printf("  No treatment: %.3f\n", stats["no_treatment_healthy"])
    @printf("  Basic care: %.3f\n", stats["basic_care_healthy"])
    @printf("  Intensive care: %.3f\n", stats["intensive_care_healthy"])
    
    println("\nHealthcare Utilization (Sick, N=$(stats["N_sick"])):")
    @printf("  No treatment: %.3f\n", stats["no_treatment_sick"])
    @printf("  Basic care: %.3f\n", stats["basic_care_sick"])
    @printf("  Intensive care: %.3f\n", stats["intensive_care_sick"])
    @printf("  Any treatment: %.3f\n", stats["any_treatment_sick"])
    
    println("\nConsumption:")
    @printf("  Period 1: mean = %.2f, sd = %.2f\n", stats["mean_c1"], stats["std_c1"])
    @printf("  Period 2: mean = %.2f, sd = %.2f\n", stats["mean_c2"], stats["std_c2"])
    
    println("\nOut-of-Pocket Medical Spending:")
    @printf("  Overall mean: %.2f\n", stats["mean_oop_medical"])
    @printf("  Mean (if sick): %.2f\n", stats["mean_oop_medical_sick"])
    
    println("="^70)
end

"""
Validate model predictions against theoretical values
"""
function validate_model(df::DataFrame, p::ModelParams; verbose::Bool=true)
    if verbose
        println("\n" * "="^70)
        println("MODEL VALIDATION")
        println("="^70)
    end
    
    validation_results = Dict{String, Any}()
    
    # =========================================================================
    # 1. HEALTH PRODUCTION VALIDATION
    # =========================================================================
    
    # Theoretical predictions
    pr_sick_no_san_theory = household_illness_prob(0, p)
    pr_sick_yes_san_theory = household_illness_prob(1, p)
    
    # Empirical estimates
    df_no_san = df[df.S_1 .== 0, :]
    df_yes_san = df[df.S_1 .== 1, :]
    
    pr_sick_no_san_emp = isempty(df_no_san) ? NaN : mean(df_no_san.H_2)
    pr_sick_yes_san_emp = isempty(df_yes_san) ? NaN : mean(df_yes_san.H_2)
    
    validation_results["health_production"] = Dict(
        "theory_no_san" => pr_sick_no_san_theory,
        "empirical_no_san" => pr_sick_no_san_emp,
        "diff_no_san" => abs(pr_sick_no_san_theory - pr_sick_no_san_emp),
        "theory_yes_san" => pr_sick_yes_san_theory,
        "empirical_yes_san" => pr_sick_yes_san_emp,
        "diff_yes_san" => abs(pr_sick_yes_san_theory - pr_sick_yes_san_emp)
    )
    
    if verbose
        println("\n1. Health Production Validation:")
        @printf("  Pr(H_2=1 | S_1=0):\n")
        @printf("    Theory    = %.4f\n", pr_sick_no_san_theory)
        @printf("    Empirical = %.4f\n", pr_sick_no_san_emp)
        @printf("    Difference = %.4f\n", abs(pr_sick_no_san_theory - pr_sick_no_san_emp))
        
        @printf("  Pr(H_2=1 | S_1=1):\n")
        @printf("    Theory    = %.4f\n", pr_sick_yes_san_theory)
        @printf("    Empirical = %.4f\n", pr_sick_yes_san_emp)
        @printf("    Difference = %.4f\n", abs(pr_sick_yes_san_theory - pr_sick_yes_san_emp))
    end
    
    # =========================================================================
    # 2. SANITATION CHOICE VALIDATION
    # =========================================================================
    
    san_rate_emp = mean(df.S_1)
    san_prob_theory = period1_choice_prob(p.θ, p)
    
    validation_results["sanitation"] = Dict(
        "theory" => san_prob_theory,
        "empirical" => san_rate_emp,
        "diff" => abs(san_prob_theory - san_rate_emp)
    )
    
    if verbose
        println("\n2. Sanitation Choice Validation:")
        @printf("  P(S_1 = 1):\n")
        @printf("    Theory    = %.4f\n", san_prob_theory)
        @printf("    Empirical = %.4f\n", san_rate_emp)
        @printf("    Difference = %.4f\n", abs(san_prob_theory - san_rate_emp))
    end
    
    # =========================================================================
    # 3. HEALTHCARE CHOICE VALIDATION
    # =========================================================================
    
    if verbose
        println("\n3. Healthcare Choice Patterns:")
        println("  Overall:")
        @printf("    No treatment:   %.3f\n", mean(df.Q_2 .== 0))
        @printf("    Basic care:     %.3f\n", mean(df.Q_2 .== 1))
        @printf("    Intensive care: %.3f\n", mean(df.Q_2 .== 2))
    end
    
    # By health status
    df_healthy = df[df.H_2 .== 0, :]
    df_sick = df[df.H_2 .== 1, :]
    
    if verbose && !isempty(df_healthy)
        println("\n  When Healthy (N=$(nrow(df_healthy))):")
        @printf("    No treatment:   %.3f\n", mean(df_healthy.Q_2 .== 0))
        @printf("    Basic care:     %.3f\n", mean(df_healthy.Q_2 .== 1))
        @printf("    Intensive care: %.3f\n", mean(df_healthy.Q_2 .== 2))
    end
    
    if verbose && !isempty(df_sick)
        println("\n  When Sick (N=$(nrow(df_sick))):")
        @printf("    No treatment:   %.3f\n", mean(df_sick.Q_2 .== 0))
        @printf("    Basic care:     %.3f\n", mean(df_sick.Q_2 .== 1))
        @printf("    Intensive care: %.3f\n", mean(df_sick.Q_2 .== 2))
    end
    
    # Compare with theoretical choice probabilities for key states
    if verbose
        println("\n  Theoretical Choice Probabilities:")
        for S_1 in [0, 1], H_2 in [0, 1]
            probs = period2_choice_probs(S_1, H_2, p.θ, p)
            println("    P(Q_2 | S_1=$S_1, H_2=$H_2):")
            @printf("      Q_2=0: %.3f, Q_2=1: %.3f, Q_2=2: %.3f\n", probs[1], probs[2], probs[3])
        end
    end
    
    # =========================================================================
    # 4. VALIDATION STATUS
    # =========================================================================
    
    health_valid = validation_results["health_production"]["diff_no_san"] < 0.02 &&
                   validation_results["health_production"]["diff_yes_san"] < 0.02
    
    sanitation_valid = validation_results["sanitation"]["diff"] < 0.02
    
    validation_results["overall_valid"] = health_valid && sanitation_valid
    
    if verbose
        println("\n" * "="^70)
        println("VALIDATION STATUS:")
        println("  Health production: ", health_valid ? "✓ PASS" : "✗ FAIL")
        println("  Sanitation choice: ", sanitation_valid ? "✓ PASS" : "✗ FAIL")
        println("  Overall: ", validation_results["overall_valid"] ? "✓ PASS" : "✗ FAIL")
        println("="^70)
    end
    
    return validation_results
end

# ==============================================================================
# 7. POLICY EXPERIMENT FUNCTIONS
# ==============================================================================

"""
Experiment 1: Vary insurance generosity θ
Tests budget relief vs. moral hazard channels
"""
function experiment_insurance_generosity(
    N::Int, 
    p::ModelParams; 
    θ_values=[0.0, 0.3, 0.5, 0.7, 0.9],
    verbose::Bool=true
)
    if verbose
        println("\n" * "="^70)
        println("EXPERIMENT 1: VARYING INSURANCE GENEROSITY")
        println("="^70)
        println("\nTesting θ values: ", θ_values)
    end
    
    results = DataFrame(
        theta = Float64[],
        transfer = Float64[],
        sanitation_rate = Float64[],
        illness_rate = Float64[],
        treatment_rate_sick = Float64[],
        intensive_rate_sick = Float64[],
        mean_c1 = Float64[],
        mean_c2 = Float64[],
        mean_oop_medical = Float64[]
    )
    
    for θ in θ_values
        if verbose
            println("\nSimulating θ = $θ...")
        end
        
        # Update insurance generosity
        p_temp = deepcopy(p)
        p_temp.θ = θ
        
        # Simulate
        df = simulate_data(N, p_temp, seed=123)
        
        # Compute statistics
        san_rate = mean(df.S_1)
        illness_rate = mean(df.H_2)
        
        df_sick = df[df.H_2 .== 1, :]
        treatment_rate = isempty(df_sick) ? NaN : mean(df_sick.Q_2 .> 0)
        intensive_rate = isempty(df_sick) ? NaN : mean(df_sick.Q_2 .== 2)
        
        # Medical spending
        medical_spending = zeros(nrow(df))
        for i in 1:nrow(df)
            if df.H_2[i] == 1
                medical_spending[i] = (1.0 - θ) * medical_cost(df.Q_2[i], p_temp)
            end
        end
        
        T_θ = transfer(θ, p_temp)
        
        push!(results, (
            θ, 
            T_θ,
            san_rate, 
            illness_rate, 
            treatment_rate, 
            intensive_rate,
            mean(df.c_1), 
            mean(df.c_2),
            mean(medical_spending)
        ))
    end
    
    if verbose
        println("\n" * "="^70)
        println("RESULTS:")
        println("="^70)
        println(results)
        
        # Analyze pattern
        println("\n" * "="^70)
        println("INTERPRETATION:")
        println("="^70)
        
        # Show both effects separately
        println("\nMechanism Analysis:")
        println("  Transfer effect (liquidity):")
        @printf("    θ=%.1f: Transfer=\$%.0f\n", θ_values[1], results.transfer[1])
        @printf("    θ=%.1f: Transfer=\$%.0f\n", θ_values[end], results.transfer[end])
        @printf("    → Liquidity improvement: helps sanitation\n")
        
        println("\n  Insurance effect (moral hazard):")
        # Compute continuation value differences
        decomp_low = decompose_sanitation_gain(θ_values[1], ModelParams(θ=θ_values[1], T_slope=p.T_slope))
        decomp_high = decompose_sanitation_gain(θ_values[end], ModelParams(θ=θ_values[end], T_slope=p.T_slope))
        @printf("    θ=%.1f: Insurance effect = %.4f\n", θ_values[1], decomp_low.insurance)
        @printf("    θ=%.1f: Insurance effect = %.4f\n", θ_values[end], decomp_high.insurance)
        @printf("    → Change = %.4f (more negative = stronger moral hazard)\n", 
                decomp_high.insurance - decomp_low.insurance)
        
        Δ_san = results.sanitation_rate[end] - results.sanitation_rate[1]
        
        println("\n  NET EFFECT:")
        if Δ_san > 0.01
            println("  Finding: BUDGET RELIEF DOMINATES")
            println("  Higher insurance → Higher sanitation")
            println("  Mechanism: Transfers relax liquidity constraints more than")
            println("             moral hazard reduces prevention incentive")
        elseif Δ_san < -0.01
            println("  Finding: MORAL HAZARD DOMINATES")
            println("  Higher insurance → Lower sanitation")
            println("  Mechanism: Lower illness cost reduces prevention incentive more than")
            println("             transfers relax liquidity constraint")
        else
            println("  Finding: OFFSETTING EFFECTS")
            println("  Budget relief ≈ Moral hazard")
        end
        
        @printf("\n  Change in sanitation (θ=%.1f vs θ=%.1f): %.3f\n", 
                θ_values[end], θ_values[1], Δ_san)
    end
    
    return results
end

"""
Experiment 2: Myopic households (β = 0)
Tests value of forward-looking behavior
"""
function experiment_myopic(N::Int, p::ModelParams; verbose::Bool=true)
    if verbose
        println("\n" * "="^70)
        println("EXPERIMENT 2: MYOPIC HOUSEHOLDS")
        println("="^70)
    end
    
    # Baseline (forward-looking)
    if verbose
        println("\nSimulating baseline (β = $(p.β))...")
    end
    df_baseline = simulate_data(N, p, seed=123)
    stats_baseline = compute_summary_stats(df_baseline, p)
    
    # Myopic (β = 0)
    if verbose
        println("Simulating myopic (β = 0)...")
    end
    p_myopic = deepcopy(p)
    p_myopic.β = 0.0
    df_myopic = simulate_data(N, p_myopic, seed=123)
    stats_myopic = compute_summary_stats(df_myopic, p_myopic)
    
    # Compare
    if verbose
        println("\n" * "="^70)
        println("RESULTS:")
        println("="^70)
        
        println("\nBaseline (β = $(p.β)):")
        @printf("  Sanitation rate: %.3f\n", stats_baseline["sanitation_rate"])
        @printf("  Illness rate: %.3f\n", mean(df_baseline.H_2))
        @printf("  Mean c1: %.2f\n", stats_baseline["mean_c1"])
        @printf("  Mean c2: %.2f\n", stats_baseline["mean_c2"])
        @printf("  OOP medical: %.2f\n", stats_baseline["mean_oop_medical"])
        
        println("\nMyopic (β = 0):")
        @printf("  Sanitation rate: %.3f\n", stats_myopic["sanitation_rate"])
        @printf("  Illness rate: %.3f\n", mean(df_myopic.H_2))
        @printf("  Mean c1: %.2f\n", stats_myopic["mean_c1"])
        @printf("  Mean c2: %.2f\n", stats_myopic["mean_c2"])
        @printf("  OOP medical: %.2f\n", stats_myopic["mean_oop_medical"])
        
        Δ_san = stats_baseline["sanitation_rate"] - stats_myopic["sanitation_rate"]
        Δ_illness = mean(df_baseline.H_2) - mean(df_myopic.H_2)
        
        println("\n" * "="^70)
        println("CHANGES (Baseline - Myopic):")
        println("="^70)
        @printf("  Sanitation: %+.3f (%.1f%% increase)\n", Δ_san, 
                100 * Δ_san / stats_myopic["sanitation_rate"])
        @printf("  Illness: %+.3f\n", Δ_illness)
        
        println("\nINTERPRETATION:")
        if Δ_san > 0
            println("  Myopic households invest MORE in prevention")
            println("  Reason: Insurance creates NEGATIVE continuation value")
            println("    → V̄₂(with sanit) < V̄₂(without sanit) due to generous insurance")
            println("    → Forward-looking agents discouraged by this")
            println("    → Myopic agents ignore future, only see liquidity cost")
        else
            println("  Forward-looking households invest more in prevention")
            println("  They anticipate future health benefits of sanitation")
            println("  Myopic households under-invest → higher illness → higher costs")
        end
    end
    
    results = DataFrame(
        scenario = ["Baseline (β=$(p.β))", "Myopic (β=0)"],
        beta = [p.β, 0.0],
        sanitation_rate = [stats_baseline["sanitation_rate"], stats_myopic["sanitation_rate"]],
        illness_rate = [mean(df_baseline.H_2), mean(df_myopic.H_2)],
        treatment_rate_sick = [stats_baseline["any_treatment_sick"], stats_myopic["any_treatment_sick"]],
        mean_c1 = [stats_baseline["mean_c1"], stats_myopic["mean_c1"]],
        mean_c2 = [stats_baseline["mean_c2"], stats_myopic["mean_c2"]],
        oop_medical = [stats_baseline["mean_oop_medical"], stats_myopic["mean_oop_medical"]]
    )
    
    return results
end

"""
Experiment 3: High illness environment
Tests how disease burden affects prevention investment
"""
function experiment_high_illness(
    N::Int, 
    p::ModelParams; 
    p_0_high=0.50,
    verbose::Bool=true
)
    if verbose
        println("\n" * "="^70)
        println("EXPERIMENT 3: HIGH ILLNESS ENVIRONMENT")
        println("="^70)
    end
    
    # Baseline
    if verbose
        println("\nSimulating baseline (p_0 = $(p.p_0))...")
    end
    df_baseline = simulate_data(N, p, seed=123)
    stats_baseline = compute_summary_stats(df_baseline, p)
    
    # High illness
    if verbose
        println("Simulating high illness (p_0 = $p_0_high)...")
    end
    p_high = deepcopy(p)
    p_high.p_0 = p_0_high  # Keep p_1 same, increase p_0
    df_high = simulate_data(N, p_high, seed=123)
    stats_high = compute_summary_stats(df_high, p_high)
    
    # Compare
    if verbose
        println("\n" * "="^70)
        println("RESULTS:")
        println("="^70)
        
        println("\nBaseline (p_0 = $(p.p_0)):")
        @printf("  Pr(H_2=1 | S_1=0) = %.3f\n", household_illness_prob(0, p))
        @printf("  Pr(H_2=1 | S_1=1) = %.3f\n", household_illness_prob(1, p))
        @printf("  Sanitation rate: %.3f\n", stats_baseline["sanitation_rate"])
        @printf("  Illness rate: %.3f\n", mean(df_baseline.H_2))
        @printf("  OOP medical: %.2f\n", stats_baseline["mean_oop_medical"])
        
        println("\nHigh Illness (p_0 = $(p_high.p_0)):")
        @printf("  Pr(H_2=1 | S_1=0) = %.3f\n", household_illness_prob(0, p_high))
        @printf("  Pr(H_2=1 | S_1=1) = %.3f\n", household_illness_prob(1, p_high))
        @printf("  Sanitation rate: %.3f\n", stats_high["sanitation_rate"])
        @printf("  Illness rate: %.3f\n", mean(df_high.H_2))
        @printf("  OOP medical: %.2f\n", stats_high["mean_oop_medical"])
        
        Δ_san = stats_high["sanitation_rate"] - stats_baseline["sanitation_rate"]
        
        println("\n" * "="^70)
        println("CHANGES (High - Baseline):")
        println("="^70)
        @printf("  Sanitation: %+.3f (%.1f%% increase)\n", Δ_san,
                100 * Δ_san / stats_baseline["sanitation_rate"])
        
        println("\nINTERPRETATION:")
        println("  Higher disease burden → stronger prevention incentive")
        println("  Sanitation becomes more attractive when illness risk is high")
        println("  Policy implication: Target sanitation subsidies to high-burden areas")
    end
    
    results = DataFrame(
        scenario = ["Baseline", "High Illness"],
        p_0 = [p.p_0, p_high.p_0],
        Pr_H2_no_san = [household_illness_prob(0, p), household_illness_prob(0, p_high)],
        Pr_H2_yes_san = [household_illness_prob(1, p), household_illness_prob(1, p_high)],
        sanitation_rate = [stats_baseline["sanitation_rate"], stats_high["sanitation_rate"]],
        illness_rate = [mean(df_baseline.H_2), mean(df_high.H_2)],
        treatment_rate_sick = [stats_baseline["any_treatment_sick"], stats_high["any_treatment_sick"]],
        mean_c2 = [stats_baseline["mean_c2"], stats_high["mean_c2"]],
        oop_medical = [stats_baseline["mean_oop_medical"], stats_high["mean_oop_medical"]]
    )
    
    return results
end

# ==============================================================================
# 8. BACKWARD RECURSION VERIFICATION
# ==============================================================================

"""
Print complete backward recursion for verification
Shows all continuation values and choice probabilities
"""
function print_backward_recursion(p::ModelParams)
    println("\n" * "="^70)
    println("BACKWARD RECURSION VERIFICATION")
    println("="^70)
    
    θ = p.θ
    
    # PERIOD 2
    println("\n" * "─"^70)
    println("PERIOD 2: Healthcare Choice")
    println("─"^70)
    
    for S_1 in [0, 1]
        for H_2 in [0, 1]
            println("\nState: S_1=$S_1, H_2=$H_2")
            
            # Systematic utilities
            println("  Systematic utilities:")
            for Q_2 in 0:2
                v̄ = period2_systematic_utility(Q_2, S_1, H_2, θ, p)
                c_2 = period2_consumption(Q_2, H_2, θ, p)
                @printf("    v̄_2(Q_2=%d) = %.4f  [c_2 = %.2f]\n", Q_2, v̄, c_2)
            end
            
            # Continuation value
            V2 = period2_continuation_value(S_1, H_2, θ, p)
            @printf("  V_2(S_1=%d, H_2=%d) = %.4f\n", S_1, H_2, V2)
            
            # Choice probabilities
            probs = period2_choice_probs(S_1, H_2, θ, p)
            println("  Choice probabilities:")
            @printf("    P(Q_2=0) = %.3f\n", probs[1])
            @printf("    P(Q_2=1) = %.3f\n", probs[2])
            @printf("    P(Q_2=2) = %.3f\n", probs[3])
        end
    end
    
    # INTEGRATION OVER H_2
    println("\n" * "─"^70)
    println("INTEGRATION OVER HEALTH UNCERTAINTY")
    println("─"^70)
    
    for S_1 in [0, 1]
        pr_sick = household_illness_prob(S_1, p)
        println("\nS_1=$S_1:")
        @printf("  Pr(H_2=1 | S_1=%d) = %.4f\n", S_1, pr_sick)
        @printf("  Pr(H_2=0 | S_1=%d) = %.4f\n", S_1, 1.0 - pr_sick)
        
        V2_healthy = period2_continuation_value(S_1, 0, θ, p)
        V2_sick = period2_continuation_value(S_1, 1, θ, p)
        V2_bar = period2_expected_value(S_1, θ, p)
        
        @printf("  V_2(S_1=%d, H_2=0) = %.4f\n", S_1, V2_healthy)
        @printf("  V_2(S_1=%d, H_2=1) = %.4f\n", S_1, V2_sick)
        @printf("  V̄_2(S_1=%d) = %.4f\n", S_1, V2_bar)
    end
    
    # PERIOD 1
    println("\n" * "─"^70)
    println("PERIOD 1: Sanitation Choice")
    println("─"^70)
    
    for S_1 in [0, 1]
        c_1 = period1_consumption(S_1, θ, p)
        V2_bar = period2_expected_value(S_1, θ, p)
        v̄_1 = period1_systematic_utility(S_1, θ, p)
        
        println("\nS_1=$S_1:")
        @printf("  c_1 = %.2f\n", c_1)
        @printf("  V̄_2(S_1=%d) = %.4f\n", S_1, V2_bar)
        @printf("  v̄_1(S_1=%d) = %.4f\n", S_1, v̄_1)
    end
    
    # NET GAIN AND CHOICE PROBABILITY
    println("\n" * "─"^70)
    println("SANITATION DECISION")
    println("─"^70)
    
    Δ = sanitation_net_gain(θ, p)
    decomp = decompose_sanitation_gain(θ, p)
    P_san = period1_choice_prob(θ, p)
    
    @printf("\nNet gain from sanitation: Δ = %.4f\n", Δ)
    println("\nDecomposition:")
    @printf("  Liquidity effect:  %.4f\n", decomp.liquidity)
    @printf("  Insurance effect:  %.4f\n", decomp.insurance)
    @printf("  Total:             %.4f\n", decomp.total)
    
    @printf("\nP(S_1 = 1) = %.4f\n", P_san)
    
    println("\n" * "="^70)
end

# ==============================================================================
# 9. RESULTS TABLE GENERATION
# ==============================================================================

"""
Create formatted table for paper
"""
function create_results_table(
    baseline_stats::Dict{String, Float64},
    exp1_results::DataFrame,
    exp2_results::DataFrame,
    exp3_results::DataFrame
)
    println("\n" * "="^70)
    println("SUMMARY TABLE: ALL EXPERIMENTS")
    println("="^70)
    
    # Select key scenarios from Experiment 1
    exp1_low = exp1_results[exp1_results.theta .== 0.3, :]
    exp1_high = exp1_results[exp1_results.theta .== 0.9, :]
    
    summary_table = DataFrame(
        Experiment = String[],
        Parameter = String[],
        Sanitation = Float64[],
        Illness = Float64[],
        Treatment_When_Sick = Float64[],
        Mean_C2 = Float64[]
    )
    
    # Baseline
    push!(summary_table, (
        "Baseline",
        "θ=0.7, β=0.9, p₀=0.30",
        baseline_stats["sanitation_rate"],
        baseline_stats["overall_illness_rate"],
        baseline_stats["any_treatment_sick"],
        baseline_stats["mean_c2"]
    ))
    
    # Experiment 1: Low insurance
    push!(summary_table, (
        "Exp 1: Low Insurance",
        "θ=0.3",
        exp1_low.sanitation_rate[1],
        exp1_low.illness_rate[1],
        exp1_low.treatment_rate_sick[1],
        exp1_low.mean_c2[1]
    ))
    
    # Experiment 1: High insurance
    push!(summary_table, (
        "Exp 1: High Insurance",
        "θ=0.9",
        exp1_high.sanitation_rate[1],
        exp1_high.illness_rate[1],
        exp1_high.treatment_rate_sick[1],
        exp1_high.mean_c2[1]
    ))
    
    # Experiment 2: Myopic
    myopic_row = exp2_results[exp2_results.scenario .== "Myopic (β=0)", :]
    push!(summary_table, (
        "Exp 2: Myopic",
        "β=0",
        myopic_row.sanitation_rate[1],
        myopic_row.illness_rate[1],
        myopic_row.treatment_rate_sick[1],
        myopic_row.mean_c2[1]
    ))
    
    # Experiment 3: High illness
    high_ill_row = exp3_results[exp3_results.scenario .== "High Illness", :]
    push!(summary_table, (
        "Exp 3: High Illness",
        "p₀=0.50",
        high_ill_row.sanitation_rate[1],
        high_ill_row.illness_rate[1],
        high_ill_row.treatment_rate_sick[1],
        high_ill_row.mean_c2[1]
    ))
    
    println("\n")
    show(summary_table, allrows=true, allcols=true)
    println("\n")
    
    # Percentage changes from baseline
    println("\n" * "="^70)
    println("PERCENTAGE CHANGES FROM BASELINE")
    println("="^70)
    
    base_san = baseline_stats["sanitation_rate"]
    base_ill = baseline_stats["overall_illness_rate"]
    
    println("\nExperiment 1 (Insurance Generosity):")
    @printf("  θ=0.3: Sanitation %+.1f%%, Illness %+.1f%%\n",
            100 * (exp1_low.sanitation_rate[1] - base_san) / base_san,
            100 * (exp1_low.illness_rate[1] - base_ill) / base_ill)
    @printf("  θ=0.9: Sanitation %+.1f%%, Illness %+.1f%%\n",
            100 * (exp1_high.sanitation_rate[1] - base_san) / base_san,
            100 * (exp1_high.illness_rate[1] - base_ill) / base_ill)
    
    println("\nExperiment 2 (Myopic):")
    @printf("  Sanitation %+.1f%%, Illness %+.1f%%\n",
            100 * (myopic_row.sanitation_rate[1] - base_san) / base_san,
            100 * (myopic_row.illness_rate[1] - base_ill) / base_ill)
    
    println("\nExperiment 3 (High Illness):")
    @printf("  Sanitation %+.1f%%, Illness %+.1f%%\n",
            100 * (high_ill_row.sanitation_rate[1] - base_san) / base_san,
            100 * (high_ill_row.illness_rate[1] - base_ill) / base_ill)
    
    println("="^70)
    
    return summary_table
end

# ==============================================================================
# 10. LATEX EXPORT FUNCTIONS
# ==============================================================================

"""
Export DataFrame to LaTeX table
"""
function export_to_latex(df::DataFrame, filename::String, caption::String, label::String)
    io = open(filename, "w")
    
    # Table header
    println(io, "\\begin{table}[htbp]")
    println(io, "\\centering")
    println(io, "\\caption{$caption}")
    println(io, "\\label{tab:$label}")
    
    # Determine number of columns
    ncols = ncol(df)
    col_align = "l" * "r"^(ncols-1)
    
    println(io, "\\begin{tabular}{$col_align}")
    println(io, "\\hline\\hline")
    
    # Column headers
    headers = names(df)
    header_line = join(headers, " & ") * " \\\\"
    println(io, header_line)
    println(io, "\\hline")
    
    # Data rows
    for i in 1:nrow(df)
        row_data = []
        for j in 1:ncols
            val = df[i, j]
            if ismissing(val) || (val isa Float64 && isnan(val))
                push!(row_data, "--")
            elseif val isa Float64
                # Format numbers with appropriate precision
                if abs(val) >= 100
                    push!(row_data, @sprintf("%.2f", val))
                elseif abs(val) >= 1
                    push!(row_data, @sprintf("%.3f", val))
                else
                    push!(row_data, @sprintf("%.4f", val))
                end
            else
                push!(row_data, string(val))
            end
        end
        row_line = join(row_data, " & ") * " \\\\"
        println(io, row_line)
    end
    
    # Table footer
    println(io, "\\hline\\hline")
    println(io, "\\end{tabular}")
    println(io, "\\end{table}")
    
    close(io)
    println("LaTeX table saved to: $filename")
end

"""
Export all results to LaTeX files
"""
function export_all_to_latex(results; output_dir=".")
    println("\n" * "="^70)
    println("EXPORTING RESULTS TO LATEX")
    println("="^70)
    
    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkdir(output_dir)
    end
    
    # Export Experiment 1: Insurance Generosity
    export_to_latex(
        results.exp1,
        joinpath(output_dir, "table_exp1_insurance.tex"),
        "Effect of Insurance Generosity on Sanitation Investment and Health Outcomes",
        "exp1_insurance"
    )
    
    # Export Experiment 2: Myopic Households
    export_to_latex(
        results.exp2,
        joinpath(output_dir, "table_exp2_myopic.tex"),
        "Comparison of Forward-Looking vs. Myopic Households",
        "exp2_myopic"
    )
    
    # Export Experiment 3: High Illness Environment
    export_to_latex(
        results.exp3,
        joinpath(output_dir, "table_exp3_high_illness.tex"),
        "Effect of Disease Burden on Sanitation Investment",
        "exp3_high_illness"
    )
    
    # Export Summary Table
    export_to_latex(
        results.summary,
        joinpath(output_dir, "table_summary.tex"),
        "Summary of All Policy Experiments",
        "summary"
    )
    
    # Create a comprehensive statistics table from baseline
    stats = results.baseline_stats
    baseline_table = DataFrame(
        Statistic = [
            "Sample Size",
            "Sanitation Adoption Rate",
            "Illness Rate (No Sanitation)",
            "Illness Rate (With Sanitation)",
            "Overall Illness Rate",
            "Treatment Rate (When Sick)",
            "Mean Period 1 Consumption",
            "Mean Period 2 Consumption",
            "Mean OOP Medical Spending"
        ],
        Value = [
            stats["N"],
            stats["sanitation_rate"],
            stats["illness_rate_no_san"],
            stats["illness_rate_yes_san"],
            stats["overall_illness_rate"],
            stats["any_treatment_sick"],
            stats["mean_c1"],
            stats["mean_c2"],
            stats["mean_oop_medical"]
        ]
    )
    
    export_to_latex(
        baseline_table,
        joinpath(output_dir, "table_baseline_stats.tex"),
        "Baseline Summary Statistics",
        "baseline_stats"
    )
    
    println("\n" * "="^70)
    println("All LaTeX tables exported successfully!")
    println("Files saved in: $output_dir")
    println("="^70)
end

# ==============================================================================
# 11. MAIN EXECUTION FUNCTION
# ==============================================================================

"""
Main function to run complete analysis
Runs all experiments and generates comprehensive output
"""
function main()
    println("\n" * "="^70)
    println("HEALTH INSURANCE AND SANITATION STRUCTURAL MODEL")
    println("Complete Analysis")
    println("="^70)
    
    # =========================================================================
    # SETUP
    # =========================================================================
    println("\n[1/8] Initializing parameters...")
    p = ModelParams()
    print_parameters(p)
    
    # =========================================================================
    # BASELINE SIMULATION
    # =========================================================================
    println("\n[2/8] Simulating baseline scenario...")
    N = 10000
    df_baseline = simulate_data(N, p, seed=123)
    
    println("\n[3/8] Computing summary statistics...")
    stats_baseline = compute_summary_stats(df_baseline, p)
    print_summary_stats(stats_baseline)
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    println("\n[4/8] Validating model...")
    validation_results = validate_model(df_baseline, p, verbose=true)
    
    # =========================================================================
    # BACKWARD RECURSION
    # =========================================================================
    println("\n[5/8] Printing backward recursion...")
    print_backward_recursion(p)
    
    # =========================================================================
    # POLICY EXPERIMENTS
    # =========================================================================
    println("\n[6/8] Running policy experiments...")
    
    # Experiment 1: Insurance generosity
    exp1_results = experiment_insurance_generosity(N, p, verbose=true)
    
    # Experiment 2: Myopic households
    exp2_results = experiment_myopic(N, p, verbose=true)
    
    # Experiment 3: High illness environment
    exp3_results = experiment_high_illness(N, p, verbose=true)
    
    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    println("\n[7/8] Creating summary table...")
    summary_table = create_results_table(
        stats_baseline,
        exp1_results,
        exp2_results,
        exp3_results
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    println("\n[8/8] Analysis complete!")
    println("\n" * "="^70)
    println("KEY FINDINGS")
    println("="^70)
    
    println("\n1. BASELINE BEHAVIOR:")
    @printf("   - %.1f%% of households adopt sanitation\n", 100 * stats_baseline["sanitation_rate"])
    @printf("   - Illness rate: %.1f%% (no sanit.) vs %.1f%% (with sanit.)\n",
            100 * stats_baseline["illness_rate_no_san"],
            100 * stats_baseline["illness_rate_yes_san"])
    
    println("\n2. INSURANCE EFFECTS:")
    Δ_san = exp1_results.sanitation_rate[end] - exp1_results.sanitation_rate[1]
    if Δ_san > 0.01
        println("   - Budget relief DOMINATES: Higher insurance → More sanitation")
    elseif Δ_san < -0.01
        println("   - Moral hazard DOMINATES: Higher insurance → Less sanitation")
    else
        println("   - Offsetting effects: Budget relief ≈ Moral hazard")
    end
    
    println("\n3. FORWARD-LOOKING BEHAVIOR:")
    base_san = stats_baseline["sanitation_rate"]
    myopic_san = exp2_results.sanitation_rate[2]
    @printf("   - Forward-looking households: %.1f%% adopt sanitation\n", 100 * base_san)
    @printf("   - Myopic households: %.1f%% adopt sanitation\n", 100 * myopic_san)
    if myopic_san > base_san
        @printf("   - SURPRISING: Myopic adopt MORE (%.1f pp)\n", 100 * (myopic_san - base_san))
        println("   - Explanation: With generous insurance (θ=0.7), sanitation reduces")
        println("     CURRENT consumption but has NEGATIVE continuation value")
        println("     (insurance makes future illness less costly)")
        println("   - Forward-looking agents see this and invest LESS")
        println("   - Myopic agents ignore future, focus only on liquidity cost")
    else
        @printf("   - Myopic adopt LESS: %.1f percentage points lower\n", 100 * (base_san - myopic_san))
        println("   - Forward-looking households anticipate future health benefits")
    end
    
    println("\n4. DISEASE BURDEN:")
    high_san = exp3_results.sanitation_rate[2]
    @printf("   - High illness environment increases sanitation by %.1f pp\n",
            100 * (high_san - base_san))
    
    println("\n" * "="^70)
    println("ANALYSIS COMPLETE - ALL RESULTS SAVED")
    println("="^70)
    
    # =========================================================================
    # EXPORT TO LATEX
    # =========================================================================
    results_tuple = (
        baseline_stats = stats_baseline,
        baseline_data = df_baseline,
        validation = validation_results,
        exp1 = exp1_results,
        exp2 = exp2_results,
        exp3 = exp3_results,
        summary = summary_table
    )
    
    export_all_to_latex(results_tuple)
    
    return results_tuple
end

# ==============================================================================
# END OF SOURCE FILE
# ==============================================================================

println("\n" * "="^70)
println("SOURCE FILE LOADED SUCCESSFULLY")
println("="^70)
println("\nAvailable functions:")
println("  - ModelParams() - Create parameter structure")
println("  - simulate_data() - Simulate N households")
println("  - validate_model() - Validate against theory")
println("  - experiment_insurance_generosity() - Policy Exp 1")
println("  - experiment_myopic() - Policy Exp 2")
println("  - experiment_high_illness() - Policy Exp 3")
println("  - print_backward_recursion() - Verify backward recursion")
println("  - create_results_table() - Generate summary table")
println("  - main() - Run complete analysis")
println("="^70) 
