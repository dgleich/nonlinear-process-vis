## Let's see local coordinates from PageRank and Flow vectors.

## Add our standard setup
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

##

""" `prlocal(A,R,N)` Return the matrix of local PageRank samples.

### Optional Parameters
- `expand::Int` the number of BFS steps to take around the sampled vertex.
"""
function prlocal(A, R, N::Int; alpha=0.85, expand::Int=0)
  X = zeros(size(A,1),N)
  for i=1:N
    Rs = [rand(R)] # randomly sample from set R
    if expand > 0
      Rs = neighborhood(A, Rs, expand)
    end
    X[:,i] = personalized_pagerank(A, alpha, Set(Rs))
  end
  return X
end

X = prlocal(G, Pset, 100)
U,S,V = svd(X)
## Compare mean to expected value compared to singular
xpset = personalized_pagerank(G, 0.85, Set(Pset))
Xmean = mean(X,dims=2) # row-average
u1 = U[:,1]*sign(U[1,1])
@show norm(xpset/norm(xpset)-u1/norm(u1))
@show norm(Xmean/norm(Xmean)-u1/norm(u1))
@show norm(xpset-Xmean)
## Not too close in this case.
scatter(u1/norm(u1), xpset/norm(xpset) ) # but lots of correlation among large entries.


##
U,S,V = svd(X)
##
scatter(U[Pset,2],U[Pset,3])
## We find log-transformed PageRank values give better results
U,S,V = svd(log10.(X)) # thankfully, PageRank is positive!

##
DiffusionTools.draw_graph(G[Pset,Pset], U[Pset,1:2];
  size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)

##
DiffusionTools.draw_graph(G[Pset,Pset], U[Pset,2:3];
  size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)

## Now let's try the same thing for flow.
function flowlocal(A, R, N::Int; delta=0.5, expand::Int=0)
  X = zeros(size(A,1),N)
  for i=1:N
    Rs = [rand(R)] # randomly sample from set R
    if expand > 0
      Rs = neighborhood(A, Rs, expand)
    end
    S = local_flow_improve(A, Rs, delta)[1]
    X[S,i] .= 1
  end
  return X
end
F = flowlocal(G, Pset, 100)
## This has a problem... flow has trouble growing from small sets.
# For this reason, we want to expand the sets by one or two steps.
F = flowlocal(G, Pset, 100; expand=1)
##
rank(F)
##
U,S,V = svd(F)

##
DiffusionTools.draw_graph(G[Pset,Pset], U[Pset,1:2];
  size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)

## Hmm... our last random sample was better :) let's try one more...
DiffusionTools.draw_graph(G[Pset,Pset], U[Pset,2:3];
    size=(2000,1000), linewidth=1.5,
    framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
    axis_buffer=0.02)
## For a variety of reasons, we find it handy to show structure by
# order instead of value.
byrank(x) = invperm(sortperm(x))
function mydraw_P(A,Pset,xy; rank::Bool=false)
  As = A[Pset,Pset]
  if rank
    xy = [byrank(xy[Pset,1]) byrank(xy[Pset,2])]
  else
    xy = xy[Pset,:]
  end
  DiffusionTools.draw_graph(As, xy;
      size=(2000,1000), linewidth=1.5,
      framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
      axis_buffer=0.02)
end
mydraw_P(G, Pset, U[:,1:2]; rank=true)
##
mydraw_P(G, Pset, U[:,2:3]; rank=true)

## Let's see the values on the nodes.
scatter(xy[Pset,1,],xy[Pset,2], marker_z=byrank(U[Pset,1]), colorbar=false, legend=false,
  framestyle=:none)
## Spectral vs. flow comparison
begin
  vnum = 1
  U,S,V = svd(X)
  p1 = scatter(xy[Pset,1,],xy[Pset,2],
    marker_z=byrank(U[Pset,vnum]), colorbar=false, legend=false,
    framestyle=:none, title="Spectral")
  U,S,V = svd(F)
  p2 = scatter(xy[Pset,1,],xy[Pset,2],
    marker_z=byrank(U[Pset,vnum]), colorbar=false, legend=false,
    framestyle=:none, title="Flow")
  plot(p1,p2)
end

##
begin
  vnum = 2
  U,S,V = svd(X)
  p1 = scatter(xy[Pset,1,],xy[Pset,2],
    marker_z=byrank(U[Pset,vnum]), colorbar=false, legend=false,
    framestyle=:none, title="Spectral")
  U,S,V = svd(F)
  p2 = scatter(xy[Pset,1,],xy[Pset,2],
    marker_z=byrank(U[Pset,vnum]), colorbar=false, legend=false,
    framestyle=:none, title="Flow")
  plot(p1,p2)
end



## Let's see if we can make this look a little better.
# We can use the expand option to make it
F = flowlocal(G, Pset, 100; expand=1)
##
U,S,V = svd(F)
mydraw_P(G, Pset, U[:,1:2]; rank=true)
## Didn't seem to do much.
