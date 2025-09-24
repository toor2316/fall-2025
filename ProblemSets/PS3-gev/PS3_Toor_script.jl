using Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables
cd(@__DIR__) 

include("PS3_Toor_source.jl") 

allwrap() 

# Answer 2. Interpret the estimated coefficient γ
#estimated gamma is -0.094 
#Gamma represents the change in latent utility 
#with a 1-unit change in relative E(log wage) 
#in occupation j (relative to Other) 
#Gamma being negative is counter intuitive, as following to economic theory, we would expect higher wages to increase utility.
