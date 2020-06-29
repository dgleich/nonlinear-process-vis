## Load the graph and draw the spectral visualization.

include("diffusion-tools.jl")
using MatrixNetworks
using Plots
using LinearAlgebra
using Statistics
using DelimitedFiles
##
G = MatrixNetworks.readSMAT("data/nonlin-text-graph.smat")
xy = readdlm("data/nonlin-text-graph.xy")
## Let's see the graph with it's native coordinates!
DiffusionTools.draw_graph(G,xy; size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3)
## A spectral embedding is a set of coordinates for each vertex.
# This is also called a Laplacian embedding.
X,lams = DiffusionTools.spectral_embedding(G, 4)
#= From the help of spectral embedding.
  Get a spectral embedding of a sparse matrix A that represents a graph.

  This handles small matrices by using LAPACK and large sparse matrices with ARPACK.
  Given a sparse matrix $ A $, this returns the smallest eigenspace of the
  generalized eigenvalue problem min x'Lx/x'Dx where L is the Laplacian and D
  is the degree matrix. The sign of the eigenspace is based on the vertex
  with maximum degree.
=#
## We usually plot on the second and third non-trivial eigenvectors.
DiffusionTools.draw_graph(G,X[:,2:3]; size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
## This often happens with spectral embeddings :(


## What we can do instead is to look at a local spectral embedding.
# Let's extract the set of vertices associated with the letter "P"
# I worked out an easy way to do this with a PageRank vector!
# Let's see if I remember how :)
x = personalized_pagerank(G, 0.99, 4) # solve a Personalized PageRank problem on node 4.
##
scatter(xy[:,1],xy[:,2],marker_z=x.^(1/4), # small powers or logs of PR values look good.
  markersize=4, colorbar=false, markerstrokewidth=0, legend=false, framestyle=:none)
## Let's show just the subset
scatter(xy[:,1],xy[:,2],marker_z=x .> 4e-4, # small powers or logs of PR values look good.
  markersize=2, colorbar=false, markerstrokewidth=0, legend=false, framestyle=:none)
Pset = findall(x .> 4e-4)
scatter!(xy[Pset,1],xy[Pset,2], markersize=4, size=(800,400))
## We can visualize this subset with a local spectral embedding.
# A local spectral embedding is something proposed by Fan Chung in a
# paper around early 2007
# @Article{Chung-2007-local-cuts,
#  author = {Fan Chung},
#  title = {Random walks and local cuts in graphs},
#  journal = {Linear Algebra and its Applications},
#  year = {2007},
#  volume = {423}, number = {1}, pages = {22 - 32}, doi = {10.1016/j.laa.2006.07.018}}
# The idea is that we can use "Dirichlet Eigenvectors" for local spectral analysis.
# This simply means we put a boundary condition of "0" on all the quantities
# outside our local set, so we look for eigenvectors where everything not
# in Pset is equal to zero.

# Pragmatically, this means that we can just extract a submatrix of the
# Laplacian and compute its eigenvectors.

X,lams = DiffusionTools.local_spectral_embedding(G, Pset, 3)

## Note that the smallest eigenvalue isn't zero for the local spectral
# embedding. This is because we are looking at a submatrix of the Laplacian.
# (Among other reasons).

##
DiffusionTools.draw_graph(G[Pset,Pset], X[:,1:2];
  size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
## The cool thing is that when we do this local analysis, we get a nice
# picture of the "P" .

## We can get other drawings and info with other eigenectors
DiffusionTools.draw_graph(G[Pset,Pset], X[:,2:3];
  size=(2000,1000), linewidth=1.5,
  framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
  axis_buffer=0.02)
