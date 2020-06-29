## Why is this discrete nature of flow-based methods a problem?
# Think about using multiple seeds to define a local basis.

## Repeat our setup
include("FlowSeed-1.0.jl")
include("diffusion-tools.jl")
using MatrixNetworks, Plots, LinearAlgebra, Statistics, DelimitedFiles
G = MatrixNetworks.readSMAT("data/nonlin-text-graph.smat")
xy = readdlm("data/nonlin-text-graph.xy")
# Recompute Pset so this file is independent.
# You can check, because the length should be 506.
x = personalized_pagerank(G, 0.99, 4)
Pset = findall(x .> 4e-4)
@assert length(Pset) == 506

## Build the neighborhoods
R1 = neighborhood(G, [Pset[5]], 1)
R2 = neighborhood(G, [Pset[2]], 1)

## Show that we could get interesting coordinates from PageRank
x1 = personalized_pagerank(G, 0.85, Set(R1))
x2 = personalized_pagerank(G, 0.85, Set(R2))
## (log-transformed PageRank values work nicely)
DiffusionTools.draw_graph(G[Pset,Pset], log10.([x1 x2][Pset,:]);
  size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
## This is not useful for Flow
S1 = local_flow_improve(G, R1, 0.25)[1]# I don't want to find the huge set
S2 = local_flow_improve(G, R2, 0.25)[1]#
function _set_to_indicator(n,S)
 i=zeros(n); i[S] .= 1; return i
end
Z = [_set_to_indicator(size(G,1), S1) _set_to_indicator(size(G,1), S1)]

## We get nothing very useful.
DiffusionTools.draw_graph(G[Pset,Pset], Z[Pset,:];
  size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
