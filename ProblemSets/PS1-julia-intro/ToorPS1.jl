using Test, JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
#.................
#Question 1. part A
#.................
#set the seed 
Random.seed!(1234)
# Draw Uniform random numbers 
A= rand(10,7)
# modifying and expanding this 
A= -5 .+ 15*rand(10,7) 
A=rand(Uniform(-5,10),10,7) 

# Draw normal rand 
B=rand(Normal(-2,15),10,7) 
#translation and expansion to transfrom Distributions
#Indexing
C=[A[1:5,1:5] B[1:5,6:7]]
# Bit array/ Dummy variable 
D=A.*(A.<=0) 
#............. 
#Question 1. part B
#....................
size(A)
size(A,1)*size(A,2)
length(A)
size(A[:]) 
#............. 
#Question 1. part C
#....................
length(D)
length(unique(D))
#............. 
#Question 1. part D
#....................
E=reshape(B,70,1)
E=reshape(B,length(B),1)
E=B[:] 
#............. 
#Question 1. part E
#Creaiting 3 dimentsional array 
#....................
F=cat(A,B; dims=3) 

#............. 
#Question 1. part F 
#................
F=permutedims(F, (3,1,2))
#Creaiting 3 dimentsional array 
#.................... 
#Question 1. part G
#................ 
G=kron(B,C) 
#G=kron(C,F) # this doesnot work 
#..................
#Question 1. part H
#..................
# saving the matrices to jld file
# saving matrices A, B, C, D, E, F, G
save("matrixpractice.jld",
     "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)
#.............
#QUestion 1, part I    
#.............     
#saving matrices A, B, C, D
save("firstmatrix.jld",
     "A", A, "B", B, "C", C, "D", D) 
#.............
#QUestion 1, part J   
#.............
# using the code CSV.write("Cmatrix.csv",C)showing error , so putting it in the commenet  
#convert matrix A to data frames  
DataFrame(C, :auto)
CSV.write("Cmatrix.csv", DataFrame(C, :auto)) 

#.............
#Question 1, part K
#.............
#Exporting D matrix to tab delimited file 
 df_D = DataFrame(D, :auto)
    CSV.write("Dmatrix.dat", df_D, delim='\t')
#piping in Julia
DataFrame(D, :auto) |> df -> CSV.write("Dmatrix.dat", df; delim='\t')

#........
# Question 1.L
#........
function q1()
    Random.seed!(1234)           
    A = -5 .+ 15 .* rand(10,7)   
    B = rand(Normal(-2,15), 10,7)
    C = [A[1:5,1:5]  B[1:5,6:7]] 
    D = A .<= 0             
    return A, B, C, D
end
 
function q2(A, B, C)
#....................            
#Question 2 Part (a)
#....................

AB = zeros(size(A))
  for r in axes(A,1)
    for c in axes(A,2)
      AB[r, c] = A[r, c] * B[r, c]
    end
  end
  AB=A .*B 
  #....................
  #Question 2 , part (b)
  #....................
  #find indices of c where value of c is between -5 and 5
   Cprime = Float64[]
    for c in axes(C, 2)
       for r in axes(C, 1)
            if C[r, c] >= -5 && C[r, c] <= 5
               push!(Cprime, C[r, c])
            end
   end
    end
  Cprime2 = C[ (C.>= -5) .& (C.<= 5)] 
# compare the two vectors 
Cprime == Cprime2
    if Cprime != Cprime2
       @show Cprime
       @show Cprime2
       @show Cprime .== Cprime2
        error("Cprime and Cprime2 are not the same")
    end

 #...................
 #Question 2 part (c)
  #...................
   N=15_169
   K=6  
   T=5
   X= zeros(N,K,T) 
#ordering of 2nd dimentsional 
#intercept 
#dummyvariable 
#continuous variable (normal) 
#binomial ("discrete" normal)
#another binomial 
for i in axes(X,1) 
X[i, 1, :] .= 1.0 
X[i, 5, :] .=rand(Binomial(20, 0.6)) 
X[i, 6, :] .=rand(Binomial(20,0.5))
 for t in axes(X,3) 
  X[i, 2, t]= rand() <=.75 * (6 - t) /5
  X[i, 3, t]= rand(Normal(15 + t - 1, 5*(t-1)))
  X[i, 4, t]= rand(Normal(π*(6-t), 1/e)) 
  end 
  end 
  #..................
#Question 2 part d # 
  #..................
β = zeros(K, T)
 β[1, :] = [1+0.25*(t-1) for t in 1:T]
 β[2, :] = [log(t) for t in 1:T]
 β[3, :] = [-sqrt(t) for t in 1:T]
 β[4, :] = [exp(t) - exp(t+1) for t in 1:T]
 β[5, :] = [t for t in 1:T]
 β[6, :] = [t/3 for t in 1:T]

 #.........................
  #Question 2, part (e) 
  #.........................
 Y = [X[:, :, t] * β[:, t] .+ rand(Normal(0, 0.36), N) for t in 1:T]
 
 return nothing 
end 
function q3() 
#.......................  
# Question 3 , part (a) 
#.......................
df = DataFrame(CSV.File("nlsw88.csv"))
    @show df[1:5, :]
    @show typeof(df[:, :grade])
    # save as cleaned CSV file
    CSV.write("nlsw88_processed.csv", df)

#......................... 
# Question 3 , part (b)
#part b , percentage never married 
 @show mean(df[:, :never_married])
#..........................
# Question 3, part (c)
@show freqtable(df[:, :race])
#using comprehension
#.................
#Question 3, part (d)
#.................
vars = names(df)
@show describe(df) 
summarystats= describe(df) 
@show summarystats 
# ...............................
#Question 3, part (e)
#.........................
#Cross tabulation of industry and occupation
@show freqtable(df[:, :industry], df[:, :occupation])
#....................
#question 3, part (f) 
#.....................
df_sub = df[:, [:industry, :occupation, :wage]]
grouped = groupby(df_sub, [:industry, :occupation])
mean_wage = combine(grouped, :wage => mean => :mean_wage)
@show mean_wage 

return nothing
end
# part b and (c) of question 4 
"""
matrixops(A,B) 
Performs the following operations on matricies A and B: 
1.Computes the elementwise product of A and B 
2.Computes the matrix product of the transopose of A and B
3. Computes the sum of all elements of the sum of A and B
""" 

function matrixops(A::Array{Float64}, B::Array{Float64})
  # ..................
 #QUestion4, part(e)
  #...................
  # part (e) of question 4: check dimension compatibility
    if size(A) != size(B)
        error("Matrices A and B must have the same dimensions for element-wise operations.")
    end
  # (i)  elementewise product of A and B 
  out1= A .* B
  # (ii) matrix product of A' and B  
  out2= A' * B
  # (iii) sum of all elements of sum of A and B 
  out3= sum(A + B) 
  return out1, out2, out3
 end 
function q4()
#question 4, part (a) 
#three way to load the jld file 
#(i) load all variables
#@load "matrixpractice.jld" A B C D E F G 
@load "matrixpractice.jld"  

#...................
    # part (d) of question 4 
    #....................
    matrixops(A, B)
   #....................
   # part (f) of question 4
    try 
        matrixops(C, D) 
    catch err
        println("Trying matrixops(C, D):")
        println(err)
    end
#part g of question 4
#read in processed CSV
nlsw88 = DataFrame(CSV.File("nlsw88_processed.csv"))
ttl_exp = convert(Array, nlsw88.ttl_exp)
 wage = convert(Array, nlsw88.wage)
  matrixops(ttl_exp, wage)
return nothing 
end 
A, B, C, D = q1() 

q2(A, B, C)

q3()

q4()

# Test file: ToorPS1_test.jl
