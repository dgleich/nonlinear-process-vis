##
include("diffusion-tools.jl")
using MatrixNetworks

##
using DelimitedFiles
G = MatrixNetworks.readSMAT("data/nonlin-text-graph.smat")
xy = readdlm("data/nonlin-text-graph.xy")

##
seeds1 = ([3,6,9,21,12,54,16,18,2])
seeds2 = ([4,23,157,39,17,34,20,40,19])

## Show one of our nonlinear diffusion
function nonlinear_diffusion(M, h, niter, v, p)
  n = size(M,1)
  d = vec(sum(M,dims=2))
  u = zeros(n)
  u .+= v
  u ./= sum(u) # normalize
  for i=1:niter
    gu = u.^p
    u = u - h*(gu - M*gu./d)
    u = max.(u, 0) # truncate to positive
  end
  return u
end



function myplot(G,xy,pr)
  DiffusionTools.draw_graph(G,xy;
  markercolor=:black, markerstrokecolor=:white,
  markersize=2, linecolor=:lightgrey, linealpha=0.4, linewidth=0.6,
  markeralpha=0.5, framestyle=:none,
  axis_buffer=0.02, background=:black, size=(1920,1080))
  x,y = xy[:,1], xy[:,2]
  p = sortperm(pr,rev=false)
  scatter!(x[p],y[p],marker_z=pr[p].^(1/3),markersize=4.5,
      markerstrokewidth=0,legend=false,colorbar=false)
end

##
v = zeros(size(G,1))
v[seeds1] .= 1
v[seeds2] .= 1
u = nonlinear_diffusion(G, 0.01, 1000, v, 0.5)
myplot(G,xy,u)
##

#nsteps = 1:100:1000
nsteps = [1:9; 10:2:29; 30:5:100; 100:10:195; 200:20:390; 400:50:1000]
##

anim = @animate for j in nsteps
  v = zeros(size(G,1))
  v[seeds1] .= 1
  v[seeds2] .= 1

  u = nonlinear_diffusion(G, 0.01, j, v, 0.5)
  myplot(G,xy,u)
end
gif(anim, "figures/nonlin-movie.gif"; fps=30)
