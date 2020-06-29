##
using Images
using FileIO
using Plots
##
img = load("data/logo-hires.png")
##

A = 2.0.*Float64.(alpha.(img))

##
AE = diff(A,dims=1)[1:end,1:end-1] + diff(A,dims=2)[1:end-1,1:end]
##
gr(size=size(A'))
heatmap(A)
##
using StatsBase
using Random
w = weights(A)
A
##
function sample_points(A,n;rho=0.0)
  B = copy(A)
  B .+= rho
  w = weights(B)
  map = CartesianIndices(size(A))
  x = zeros(0)
  y = zeros(0)
  for i=1:n
    pt = sample(w)
    push!(x, map[pt][1])
    push!(y, map[pt][2])
  end
  return x,y
end
Random.seed!(0)
#x,y = sample_points(A,2000;rho=0.03)
x,y = sample_points(3*abs.(AE)+A[1:end-1,1:end-1],4000;rho=0.0)
x .+= randn(size(x))
y .+= randn(size(y))
x1 = x # save the first samples...
y1 = y
#x2,y2 = sample_points(abs.(A),500;rho=0.0)
#x = [x;x2]
#y = [y;y2]
x2,y2 =  sample_points(A,1500;rho=0.5)
x = [x;x2]
y = [y;y2]
scatter(y,-x,alpha=0.1)
##
using NearestNeighbors
using GraphRecipes
function gnk(xy,k,r=15)
  T = BallTree(xy)
  idxs = knn(T, xy, k)[1]
  # form the edges for sparse
  ei = Int[]
  ej = Int[]
  for i=1:size(xy,2)
    for j=idxs[i]
      if i != j
        push!(ei,i)
        push!(ej,j)
      end
    end
  end
  idxs = inrange(T, xy, r)
  for i=1:size(xy,2)
    for j=idxs[i]
      if i > j
        push!(ei,i)
        push!(ej,j)
      end
    end
  end
  return xy, ei, ej
end
using SparseArrays
using LinearAlgebra
xy, ei, ej = gnk(copy([y -x]'),5)
ei2,ej2 = gnk(copy([y1 -x1]'), 8, 45)[2:3]  # form neighbors in the logo
ei = [ei;ei2]
ej = [ej;ej2]
G = sparse(ei,ej,1)
G = min.(G,1.0)
G = max.(G,G')
ei,ej = findnz(triu(G,1))
gr(size=size(A'))
##
function draw_graph(A::SparseMatrixCSC, xy; kwargs...)
    ei,ej = findnz(triu(A,1))[1:2]
    # find the line segments
    lx = zeros(0)
    ly = zeros(0)
    for nz=1:length(ei)
        src = ei[nz]
        dst = ej[nz]
        push!(lx, xy[src,1])
        push!(lx, xy[dst,1])
        push!(lx, Inf)

        push!(ly, xy[src,2])
        push!(ly, xy[dst,2])
        push!(ly, Inf)
    end
    plot(lx,ly;
        kwargs...)
end

draw_graph(G,xy';
markercolor=:black, markerstrokecolor=:white,
markersize=4, linecolor=1, linealpha=0.8, linewidth=0.7,
axis_buffer=0.02, background=nothing)

##
## Save Data
function writeSMAT(filename::AbstractString, A::SparseMatrixCSC{T,Int};
    values::Bool=true) where T
    open(filename, "w") do outfile
        write(outfile, join((size(A,1), size(A,2), nnz(A)), " "), "\n")

        rows = rowvals(A)
        vals = nonzeros(A)
        m, n = size(A)
        for j = 1:n
           for nzi in nzrange(A, j)
              row = rows[nzi]
              val = vals[nzi]
              if values
                write(outfile, join((row-1, j-1, val), " "), "\n")
              else
                write(outfile, join((row-1, j-1, 1), " "), "\n")
              end
           end
        end
    end
end
using DelimitedFiles
writeSMAT("data/nonlin-text-graph.smat",G)
writedlm("data/nonlin-text-graph.xy", xy')

##
draw_graph(G,xy',
  markercolor=:black, markerstrokecolor=:white,
  markersize=2, linecolor=:black, linealpha=0.3, linewidth=0.5,
  markeralpha=0.5,
  axis_buffer=0.02, background=nothing)

savefig("figures/text-graph.png")
