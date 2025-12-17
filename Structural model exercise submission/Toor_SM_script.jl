using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, Distributions, Printf

cd(@__DIR__)

Random.seed!(123)

include("Toor_SM_source.jl")

# Run analysis and store results
results = main()

