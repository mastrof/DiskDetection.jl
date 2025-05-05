module RingDetection

using ImageFeatures
using Images
using TiffImages
using StatsBase

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
    centers, radii
end

#== Interactive GUI if GLMakie available ==#
using Requires
function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("gui.jl")
end

end # module
