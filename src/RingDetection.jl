module RingDetection

using ImageFeatures
using Images
using TiffImages
using StatsBase
using ParticleTracking
using GeometryBasics: Point

#== Tracking interface ==#
export Ring, radius, location, location_raw

struct Ring{T} <: ParticleTracking.AbstractBlob{T,Nothing,2}
    location::Point{2,T}
    location_raw::CartesianIndex{2}
    radius::T
end
function Ring(c::CartesianIndex{2}, r)
    Ring{Float64}(Point{2,Float64}(c.I), c, Float64(r))
end
function Base.show(io::IO, ring::Ring)
    x, y = location(ring)
    r = radius(ring)
    print(io, "Ring(x=$(x), y=$(y), r=$(r))")
end
ParticleTracking.radius(r::Ring) = r.radius
ParticleTracking.location_raw(r::Ring) = r.location_raw
for f in (zeroth_moment, second_moment, amplitude, intensity_map, scale)
    fs = nameof(f)
    @eval ParticleTracking.$fs(r::Ring{T}) where T = zero(T)
end

"""
    evaluate_maxcost(T, maxdist, dt, cost; kwargs...)
Evaluate the maximum allowed cost for a link if the maximum allowed
distance is `maxdist` pixels and the time difference `dt` frames.
"""
function ParticleTracking.evaluate_maxcost(::Type{<:Ring{T}},
    maxdist::Real, dt::Integer, cost::Function; kwargs...
) where {T}
    # define two dummy blobs maxdist apart and evaluate their linking cost
    posA = ntuple(_ -> 0, N)
    posB = ntuple(i -> i==1 ? ceil(Int, maxdist) : 0, N)
    A = Ring(CartesianIndex(posA), 0)
    B = Ring(CartesianIndex(posB), 0)
    return cost(A, B; kwargs...)
end


#== Core ==#
export imgread, sharpen, detect_rings

function imgread(fname::AbstractString; rotate=true, T=Float64)
    if rotate
        rotr90(T.(TiffImages.load(fname)))
    else
        T.(TiffImages.load(fname))
    end
end

function safeslice(M, i, j, r)
    si, sj = size(M)
    R = r+1
    imin = max(1, i-R)
    imax = min(si, i+R)
    jmin = max(1, j-R)
    jmax = min(sj, j+R)
    @view M[imin:imax, jmin:jmax]
end

function sharpen(img::AbstractMatrix; α::Real=1.05, σ::Real=1, w::Integer=3)
    med = median(img)
    proc = [u > α*med ? one(u) : zero(u) for u in img]
    proc = imfilter(proc, Kernel.gaussian(σ))
    proc = mapwindow(minimum, proc, (w,w))
    proc
end

function detect_rings(vid::AbstractVector, radii::AbstractVector; kwargs...)
    map(img -> detect_rings(img, radii; kwargs...), vid)
end
function detect_rings(img::AbstractMatrix, radii::AbstractVector;
    σ::Real=1.2, # smoothing size for Canny
    hi::Real=99, lo::Real=50, # percentile thresholds for Canny
    k=Kernel.ando5, # kernel for gradient evaluation
    min_dist::Real=minimum(radii), # minimal distance between rings
    vote_threshold::Integer=1, # Hough voting threshold
    R::Integer=0, β::Real=5, # filtering
)
    # canny edge detection
    img_edges = canny(img, (Percentile(hi), Percentile(lo)), σ)
    # phase of image gradient
    dx, dy = imgradients(img, k)
    img_phase = phase(dx, dy)
    # find rings
    hough = hough_circle_gradient(img_edges, img_phase, radii;
        min_dist, vote_threshold
    )
    # filter out rings without a central bright spot
    m = mean(img) #WARN:this may not be a good metric in general
    idx_save = findall(
        c -> mean(safeslice(img, c.I..., R)) > β*m,
        first(hough)
    )
    centers = first(hough)[idx_save]
    radii = last(hough)[idx_save]
    # centers, radii
    [Ring(c, r) for (c, r) in zip(centers, radii)]
end

#== Interactive GUI if GLMakie available ==#
using Requires
function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("gui.jl")
end

end # module
