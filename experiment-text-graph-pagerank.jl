## Nonlinear graph figures

include("diffusion-tools.jl")
using MatrixNetworks

##
using DelimitedFiles
G = MatrixNetworks.readSMAT("data/nonlin-text-graph.smat")
xy = readdlm("data/nonlin-text-graph.xy")
##
using Plots
x = personalized_pagerank(G,0.99,4)
scatter(xy[:,1],xy[:,2], marker_z = sqrt.(x), markersize=10,
markerstrokewidth=0,size=(2000,1000))
##
function myplot!(x,y,v;kwargs...)
    p = sortperm(v,rev=false)
    scatter!(x[p],y[p],marker_z=v[p];kwargs...)
end
DiffusionTools.draw_graph(G,xy;
markercolor=:black, markerstrokecolor=:white,
markersize=2, linecolor=:black, linealpha=0.3, linewidth=1.5,
markeralpha=0.5,
axis_buffer=0.02, background=nothing, size=(2000,1000))
myplot!(xy[:,1],xy[:,2],(x).^(1/3),markersize=8,
    markerstrokewidth=0,legend=false,colorbar=false)
##
savefig("figures/text-graph-seed.png")
##
plot(sort(x,rev=true),yscale=:log10)
##
x = personalized_pagerank(G,0.99,4)
myplot!(xy[:,1],xy[:,2],  x.>4e-4, markersize=10,
markerstrokewidth=0, size=(2000,1000))
##
Pset = findall(x.>4e-4)
##
X,lams = DiffusionTools.local_spectral_embedding(G,Pset,3)
##
scatter(-X[:,1],-X[:,2])
## Randomly sample
