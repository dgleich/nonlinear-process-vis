## PageRank seeds to solutions video.

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
function pr_seeds_plot(G, xy, alpha, seeds)
  pr = personalized_pagerank(G, alpha, Set(seeds))
  DiffusionTools.draw_graph(G,xy;
  markercolor=:black, markerstrokecolor=:white,
  markersize=2, linecolor=:black, linealpha=0.3, linewidth=1.5,
  markeralpha=0.5,
  axis_buffer=0.02, background=nothing, size=(2000,1000))
  x,y = xy[:,1], xy[:,2]
  p = sortperm(pr,rev=false)
  scatter!(x[p],y[p],marker_z=pr[p].^(1/3),markersize=8,
      markerstrokewidth=0,legend=false,colorbar=false)
  scatter!(x[seeds],y[seeds], markersize=16, color=:yellow,
    markerstrokewidth=0)
  #annotate!(x[seed],y[seed], "seed", color=:yellow)
  mxy = mean(xy,dims=1)
  for seed in seeds
    plot!([mxy[1],x[seed]], [mxy[2],y[seed]], arrow = :closed,
      color=:red)
  end
  annotate!(mxy[1],mxy[2], ("seeds", 36, :red))
end
pr_seeds_plot(G, xy, 0.99,seeds1)

##
pr_seeds_plot(G, xy, 0.99,seeds1)
savefig("figures/pr-linear-seeds1.png")
##
pr_seeds_plot(G, xy, 0.99,seeds2)
savefig("figures/pr-linear-seeds2.png")
##
pr_seeds_plot(G, xy, 0.99,union(seeds1,seeds2))
savefig("figures/pr-linear-seedsall.png")
