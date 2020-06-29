## PageRank seeds to solutions video.

##
include("diffusion-tools.jl")
using MatrixNetworks

##
using DelimitedFiles
G = MatrixNetworks.readSMAT("data/nonlin-text-graph.smat")
xy = readdlm("data/nonlin-text-graph.xy")

##
using Statistics
function pr_seed_plot(G, xy, alpha, seed)
  pr = personalized_pagerank(G, alpha, seed)
  DiffusionTools.draw_graph(G,xy;
  markercolor=:black, markerstrokecolor=:white,
  markersize=2, linecolor=:black, linealpha=0.3, linewidth=1.5,
  markeralpha=0.5,
  axis_buffer=0.02, background=nothing, size=(2000,1000))
  x,y = xy[:,1], xy[:,2]
  p = sortperm(pr,rev=false)
  scatter!(x[p],y[p],marker_z=pr[p].^(1/3),markersize=8,
      markerstrokewidth=0,legend=false,colorbar=false)
  scatter!([x[seed]],[y[seed]], markersize=16, color=:yellow,
    markerstrokewidth=0)
  #annotate!(x[seed],y[seed], "seed", color=:yellow)
  mxy = mean(xy,dims=1)
  plot!([mxy[1],x[seed]], [mxy[2],y[seed]], arrow = :closed,
    color=:red)
  annotate!(mxy[1],mxy[2], ("seed", 36, :red))
end
##
seeds = [3,6,9,21,12,54,16,18,2,4,23,157,39,17,34,20,40,19]
pr_seed_plot(G,xy,0.99,seeds[end])
##
anim = @animate for seed in seeds
  pr_seed_plot(G,xy,0.99,seed)
  plot!(background=:white)
end
gif(anim, "figures/seeds-pr.gif",fps=3)
##
#=
scatter(xy[:,1],xy[:,2])
for i=1:50

  annotate!(xy[i,1],xy[i,2], ("$i", 24, :red))
end
plot!()
  =#
