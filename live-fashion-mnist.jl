## Let's try these ideas again in Fashion MNIST
include("FlowSeed-1.0.jl")
include("diffusion-tools.jl")
using MatrixNetworks, Plots, LinearAlgebra, Statistics, DelimitedFiles
G = MatrixNetworks.readSMAT("data/fashion-mnist-5.smat")
labels = vec(readdlm("data/fashion-mnist.labels"))
## Local the raw images too
using MLDatasets
D = FashionMNIST.traindata()[1]
## For this, we need some scatter + images options
include("image-scatter.jl")
## Pick a fun class
# Class 0 is t-shirts
# Class 6 is sweaters
Pset = findall(labels .== 5)
##
X,lams = DiffusionTools.local_spectral_embedding(G, Pset, 3)
## Show the calss.
byrank(x) = invperm(sortperm(x))
function myplot_fashion(A,S,xy;images=nothing,vals=nothing,nimages::Int=50,
  rank::Bool=false, imgsize=12)
  if rank
    xy = [byrank(xy[:,1]) byrank(xy[:,2])]
  end
  plot(framestyle=:none, legend=false)
  DiffusionTools.draw_graph(A, xy,
    linewidth=0.75,
    framestyle=:none, linecolor=:black, linealpha=0.3, legend=false,
    axis_buffer=0.02)
  scatter!(xy[S,1],xy[S,2],markerstrokewidth=0, markersize=2,color=:black)
  if images !=nothing
    imgset = unique(rand(S,nimages))
    image_scatter!(xy[imgset,1],xy[imgset,2], images[:,:,imgset], imgsize,
      c=:grays, alphascale=0.75)
  end
  plot!()
end
myplot_fashion(G[Pset,Pset],1:length(Pset),
  X[:,1:2],images=1.0 .- D[:,:,Pset]; rank=true)
## Need to adjust image-scatter to make the pictures better...
# Done
## Compute the local flow
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
F = flowlocal(G, Pset, 250; expand=3,delta=0.1)
##
rank(F)
##
U,S,V = svd(F)
myplot_fashion(G[Pset,Pset],1:length(Pset),
  U[Pset,1:2],images=1.0 .- D[:,:,Pset]; rank=true)
##
Fbig = flowlocal(G, Pset, 250; expand=3,delta=0.1)
##
U,S,V = svd(Fbig)
myplot_fashion(G[Pset,Pset],1:length(Pset),
  U[Pset,1:2],images=1.0 .- D[:,:,Pset]; rank=true)
