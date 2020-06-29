## Let's play around with this example
using Plots

## Now let's build this into an overall routine,
# The next thing to do would be to add a Plots.jl receipe.
function _make_centered_rect(i,j,m,n)
  #Note: The center is (0,0) and the size is expected to be rougly the area of the unit circle.
  #TODO: Add those offsets, we aren't using them yet!
  dx = 2/m
  dy = 2/n
  # make it in native coords, then
  x_coords = [(i-1)*dx-1,     i*dx-1, i*dx-1, (i-1)*dx-1, (i-1)*dx-1]
  y_coords = [(j-1)*dy-1, (j-1)*dy-1, j*dy-1,     j*dy-1, (j-1)*dy-1]
  return x_coords, y_coords
end
function _matrix_to_marker(X::AbstractArray{T,2},offset=(0,0),scale=1,
    flip::Bool=true) where T
  shapes = Shape[]
  zvals = Float64[] # the
  m,n = size(X)
  for I in CartesianIndices(X)
    i,j = I[1],I[2]
    jcoord = j
    if flip
      jcoord = m-j+1
    end
    coords = _make_centered_rect(i,jcoord,m,n)
    push!(shapes, Shape(scale.*coords[1].+offset[1],scale.*coords[2].+offset[2]))
    push!(zvals, X[i,j])
  end
  return shapes, zvals
end
function image_scatter!(x,y,X::AbstractArray{T,3},scale::Real=50;
    flip::Bool=true,alphascale::Real=0.5,kwargs...) where T <: Real
  # check sizes
  @assert length(x)==length(y)
  @assert size(X,3) == length(x)
  xmin,xmax = extrema(x) # wish I could use the markersize for this...
  ymin,ymax = extrema(y)
  bigsize=max(xmax-xmin,ymax-ymin)
  for i = 1:length(x)
    shapes,zvals = _matrix_to_marker(@view(X[:,:,i]),(x[i],y[i]),
                      scale*bigsize/500,flip)
    plot!(shapes, fill_z = zvals, fillalpha=alphascale*zvals.+(1 .-alphascale),
              linecolor=:transparent, linewidth=0; kwargs...)
  end
  plot!() # add this at the end to get plots
end
