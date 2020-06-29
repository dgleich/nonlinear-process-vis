## Let's look at a nonlinear process on a graph.
# We are going to use the "LocalFlowImprove" routine due to
# Orecchia and Zhu (2014, SODA) and implementations based on
# Veldt, Mahoney and Gleich (2014, ICML) and
# Veldt, Gleich, Klymko, 2019 (SIAM SDM) that look at various
# improvements.

## To get the code, I just downloaded "FlowSeed-1.0.jl" from Veldt's github page.
# https://github.com/nveldt/FlowSeed
# and specifically the file
# https://raw.githubusercontent.com/nveldt/FlowSeed/master/algorithms/FlowSeed-1.0.jl
# We found a few small typos in some small pieces of this (likely due to
# JuliaLang changes... but maybe ours too.)
# FlowSeed is a generalized idea, but we are just going to use the special case.

## To implement local_flow_improve with this, we just have to make a specific call.
#=
function local_flow_improve(A::SparseMatrixCSC,
        R::Vector{Int},delta::Real)

     S,cond = FlowSeed(A,R,delta,zeros(length(R)), zeros(length(R)))
 end
 =#
 # We added that function to the following file.
include("FlowSeed-1.0.jl")

## (Include other files too)
include("diffusion-tools.jl")
using MatrixNetworks, Plots, LinearAlgebra, Statistics, DelimitedFiles

## Recompute Pset so this file is independent.
# You can check, because the length should be 506.
x = personalized_pagerank(G, 0.99, 4)
Pset = findall(x .> 4e-4)
@assert length(Pset) == 506

## The way local flow improve works, is that we give it a reference set R and
# then it returns an improved set S.
# The value of "delta" controls how far from the reference the algorithm is
# willing to go. If delta -> Infty, then the algorithm will always return
# a subset of Pset. If delta -> 0, then the algorithm may look further, but
# there is a limit to how far it will look.
S,cond = local_flow_improve(G, Pset, 0.1)
## Results
#=
The full seed set has conductance 0.020188720649550143
-------------------------------------------------------
Improvement found: R-Conductance = 0.0004, Size = 705
------------------------------------------------------
Final Answer: Conductance = 0.0003677428885583499, Size = 705
=#
## These results mean that we started with a conductance 0.02 set
# and LocalFlowImprove found a set with conductance 0.00036, which is much better!
## If you want a simple way to think about local flow improve, it's a 1-norm
# version of PageRank.
DiffusionTools.draw_graph(G,xy; size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
scatter!(xy[S,1], xy[S,2], framestyle=:none, legend=false, markerstrokewidth=0)
scatter!(xy[Pset,1], xy[Pset,2], color=1) # show the reference set too
## This is a fairly nonlinear process.
# Let's look at two different inputs with disjoint support.
# This will take a minute or so to find. (It also helps when you use
# intersect instead of union!)
R1 = neighborhood(G, [Pset[5]], 1)
R2 = neighborhood(G, [Pset[2]], 1)
@show length(intersect(R1,R2))
scatter(xy[R1,1],xy[R1,2],alpha=0.25)
scatter!(xy[R2,1],xy[R2,2],alpha=0.25)
## So we have the sets R1 and R2 at opposite ends.
# Let's see that PageRank is linear
x1 = personalized_pagerank(G, 0.85, Set(R1))
x2 = personalized_pagerank(G, 0.85, Set(R2))
x3 = personalized_pagerank(G, 0.85, Set(union(R1,R2)))
## An easy way to test is with the SVD/rank
@show rank([x1 x2 x3])
## For PageRank, the difference is related to the size of the groups.
norm(x3 - length(R1)/(length(R1)+length(R2))*x1 - length(R2)/(length(R1)+length(R2))*x2)

## Let's try with Local Flow Improve now
S1 = local_flow_improve(G, R1, 0.25)[1]# I don't want to find the huge set
S2 = local_flow_improve(G, R2, 0.25)[1]#
S3 = local_flow_improve(G, union(R1,R2), 0.25)[1]#
## In this case, the function gives us a rank 3 output from a linear combo of inputs.
function _set_to_indicator(n,S)
 i=zeros(n); i[S] .= 1; return i
end
rank([_set_to_indicator(size(G,1), S1) _set_to_indicator(size(G,1), S2) _set_to_indicator(size(G,1), S3) ])
## Show the nonlinearity
DiffusionTools.draw_graph(G,xy; size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
scatter!(xy[S1,1], xy[S1,2], framestyle=:none, legend=false, markerstrokewidth=0)
scatter!(xy[R1,1], xy[R1,2], framestyle=:none, legend=false, markerstrokewidth=0)
##
DiffusionTools.draw_graph(G,xy; size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
scatter!(xy[S2,1], xy[S2,2], framestyle=:none, legend=false, markerstrokewidth=0)
scatter!(xy[R2,1], xy[R2,2], framestyle=:none, legend=false, markerstrokewidth=0)

##
DiffusionTools.draw_graph(G,xy; size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
scatter!(xy[S3,1], xy[S3,2], framestyle=:none, legend=false, markerstrokewidth=0)
scatter!(xy[union(R1,R2),1], xy[union(R1,R2),2], framestyle=:none, legend=false, markerstrokewidth=0)

#scatter!(xy[R2,1], xy[R2,2], framestyle=:none, legend=false, markerstrokewidth=0)

#scatter!(xy[S2,1], xy[S2,2], framestyle=:none, legend=false, markerstrokewidth=0)
#scatter!(xy[S3,1], xy[S3,2], framestyle=:none, legend=false, markerstrokewidth=0)
