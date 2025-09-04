using Test, JLD, Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions
#set the seed 
Random.seed!(1234)
# Draw Uniform random numbers 
A= rand(10,7)
# modify this 
A= -5 .+ rand(10,7)
#expand 
A= -5 .+ rand(10,7) 
rand(Uniform(-5,10),10,7)
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
permutedimsF = permutedims(F, (3,1,2))
F=permutedims(F, (3,1,2))
#Creaiting 3 dimentsional array 
#.................... 
G=kron(B,C) 
#G=kron(C,F) # this doesnot work 
#Question 1. part H
# save JLD file 
save("matrixpractice.jld",
     "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

save("firstmatrix.jld",
     "A", A, "B", B, "C", C, "D", D)
    
CSV.write("Cmatrix.csv",C) 
#showing error 
#convert matrix A to data frames 
CSV.write("Cmatrix.csv",C) 
DataFrame(C, :auto)
CSV.write("Cmatrix.csv", DataFrame(C, :auto)) 

#piping in JUlia 

DataFrame(D, :auto) |>
    df -> CSV.write("Dmatrix.dat", df; delim='\t')

#........
# Question 1.L

function q1()
    Random.seed!(1234)           
    A = -5 .+ 15 .* rand(10,7)   
    B = rand(Normal(-2,15), 10,7)
    C = [A[1:5,1:5]  B[1:5,6:7]] 
    D = A .<= 0             
    return A, B, C, D
end

A, B, C, D = q1()              
#Question 2 Part (b)#
function q2(A,B)
#question 2 (a)
AB=zeros(size(A))
for r in 1:eachindex(A[:,1])
 for c in 1:eachindex(A[1,:])
  AB[row,col]= A[row,col] * B[row,col]
  end 
  AB=A .*B 
  #Question 2 , part c 
  #find indices of c where value of c is between -5 and 5 
  Cprime=[] 

  for c in 1:eachindex(C[1,:])
  for r in 1:eachindex(C[:,1])
  #Julia is column major 
  if C[r,c] >=-5 && C[r,c] <=5 
   push!(Cprime, C[r,c])
   end 
   end
   end 
# compare the two vectors 
Cprime = Float64[]
Cprime == Cprime2 
if Cprime != Cprime2 
 @show Cprime 
 @show Cprime2 
 @show Cprime .=
 
   #Question 2 part c 
   #N=
   X= zeros(15_169,6,5)
#ordering of 2nd dimentsional 
#intercept 
#dummyvariable 
#conrinious variable (normal) 
#binomial ("discrete" normal)
#another binomial 
for i in axes(X,1) 
X[i, 1, :] .= 1.0 
X[i, 5, :] .=rand(binomial(20, 0.6)) 
X[i, 6, :] .=rand(bionmial(20,0.5))
 for t in axes(X,3) 
  X[i, 2, t]= rand() <=.75 * (6 - t) /5
  X[i, 3, t]= rand(Normal(15 + t - 1, 5*(t-1)))
  X[i, 4, t]= rand(Normal(pi*(6-t), 1/e)) 
  end 
  end 
#Qeustion 2 part d # 
\beta 
\beta= zeros (K,T)
\beta[1, :] = [1+o.25*(t-1) for t in 1:T]
\beta[2, :] = [log(t) for t in 1:T] 
\beta[3, :] = [-sqrt(t) for t in 1:T] 
\beta[5, :] = [t for t in 1:T]
\beta[5, :] = [t/3 for t in 1:T]
#Question 2, part (e) 
Y= zeros (N,T) 
Y[:, t]= [X[:, :, t] * beta [:,t] . + rand (Normal (0,0.36), N) for t in 1:T] 

return nothing 

#call the function from q1 
A, B, C, D = q1() 

#call the function from q2 
function q2 (A, B, C) 
return nothing 
end 


function q3() 
# Question 3 , part (a) 

df= DataFrame(CSV.File("nlsw88.csv")) 
csv.write("nlsw88.dat", df; delim='\t')
@show df[1:5, :]
@show typeof (df[:, :grade])
 return nothing 
 end 

 # call the function from q1 
 A, B, C, D = q1()
#part b , percentage never married 
@show mean (df[:, : never_married]) 
#..........................
# Quesion 3, part (c)  
........................ 
@show FreqTables (df[:, :race]) 
#using comprehension 
#Question 3, part (d) 

vars=names(df) 
@show vars 

return nothing 
@show describe(df) 
summarystats= describe(df) 
@show summarystats 
# ...............................
#QUestion 3, part (e) 
#................................. 
cross tabulation of industry and occupation 
@show freqtable(df[:, :industry], df[:, :occupation]) 

#question 3, part (f) 
return nothing 
end 
ds_sub = df[: , [industry, :occupation, :wage]] 
grouped= groupby(df_sub, [:industry, :occupation]) 
mean_wage= combine(grouped, :wage=> mean=> : mean_wage) 
@show mean_wage 

return nothing 
end 
# part b and (c) of question 4 
"""
matrixops(A,B) 
Performs the following operations on matricies A and B: 
1.Computes the elementwise product of A and B 
2.Computes the matrix product of the transopose of A and B
2. Computes the sum of all elements of the sum of A and B
""" 

function matrixops(A::Array{Float64}, B::Array{Float64})
  # part e 
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
# part (e) of question 4 
#part (f) of question 4 
try 
    matrixops (C, D) 
catch e  
  println("Trying matrixops (C,D)")
   println(e) 
end
return nothing 
end  
#part g of question 4 
#read in processed CSV 
nlsw88 =  DataFrame (CSV.File("nlsw88.csv") 
ttl_exp = convert(Array, nlsw88.ttl_exp) 
wage    = convert(Array, nlsw88.wage)  
# in unit test use approxi sign sometimes 
#check matrix setup and operation right for unit test 
# the cprime thing in unit test 
#earlier include N,K,T=15_169,6,5 
#for uniit test , use less observations 
#N,K,T=100,6,5 
#creating mock data for unit test 
# using @test_throws operator as unit test for catching error 





