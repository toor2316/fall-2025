using Optim, HTTP, GLM, LinearAlgebra, DataFrames, CSV, StatsBase, Random, Statistics, FreqTables

#:::::::::::::::::::::::::::::::::::::::::::::::::::

cd(@__DIR__) 

function allwrap()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 1
#:::::::::::::::::::::::::::::::::::::::::::::::::::
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   # random starting value
result = optimize(minusf, startval, BFGS())
println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result)) 
result_better= optimize(minusf, [-7.0], BFGS()) 
println(result_better) 


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 2
#::::::::::::::::::::::::::::::::::::::::::::::::::: 
 
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2025/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1 

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y 
#standard errors 
N= size(X,1) 
K= size(X,2)
MSE= sum(((y-X*bols).^2)/(N-K))
VCOV= MSE*inv(X'*X) 
se_bols= sqrt.(diag(VCOV))  

println("OLS estimates ",bols) 

df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)


#:::::::::::::::::::::::::::::::::::::::::::::::::::
# question 3
#::::::::::::::::::::::::::::::::::::::::::::::::::: 
function logit(alpha, X, d) 
    pll = exp.(X * alpha) ./ (1 .+ exp.(X * alpha))
    ll  = -sum((d .== 1) .* log.(pll) .+ (d .== 0) .* log.(1 .- pll))
    return ll
end

alpha_hat_logit = optimize(b -> logit(b, X, y),
                           rand(size(X, 2)), LBFGS(),
                           Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true)) 
                           
println(alpha_hat_logit.minimizer) 

# ...................
#Question 4 
#................ 

alpha_hat_glm=glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println(alpha_hat_glm) 


# ...................
#Question 5
#..................

freqtable(df, :occupation) 
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7 
freqtable(df, :occupation) # problem solved 
#redefining X and y
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, y) 
K=size(X,2)
J=length(unique(y))
N=length(y)
bigY=zeros(N,J) 
for j=1:J 
 bigY[:,j]=(y.==j) 
 end
 bigalpha=[reshape(alpha,K,J-1) zeros(K)]

num=zeros(N,J) 
denom= zeros(N) 
for j=1:J 
 num[:,j]= exp.(X*bigalpha[:,j]) 
 denom .+= num[:,j] 
end 
P=num./repeat(denom,1,J)

loglike= -sum(sum(bigY.*log.(P))) 
return loglike
end 


 alpha_zero=zeros(6*size(X,2))
  alpha_rand= rand(6*size(X,2)) 
  #for the alpha_true values I am using the values pooled from estimation in stata from given code#

alpha_true = [
    # 1) Professional_Technical
     0.1910213,  -0.0335262,   0.5963968,   0.4165052,
    # 2) Managers_Admin
    -0.1698368,  -0.0359784,   1.3068400,  -0.4309970,
    # 3) Sales
     0.6894727,  -0.0104578,   0.5231634,  -1.4924750,
    # 4) Clerical_Unskilled
    -2.2674800,  -0.0053001,   1.3914020,  -0.9849661,
    # 5) Craftsmen
    -1.3984680,  -0.0142969,  -0.0176531,  -1.4951230,
    # 6) Operatives
     0.2454891,  -0.0067267,  -0.5382892,  -3.7897500
] 

alpha_start= alpha_true.*rand(size(alpha_true)) 
print(size(alpha_true)) 
alpha_hat_optim= optimize(a -> mlogit(a, X, y),
                           alpha_start, LBFGS(),
                           Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true)) 
  alpha_hat_mle= alpha_hat_optim.minimizer
  println(alpha_hat_mle) 
end 

 #6. Wrapping all of codes and calling the function 

allwrap() 

#7.Separately submitting the unit test file 
# Note: Initially had problems with commiting, so doing another commit to submit my work 