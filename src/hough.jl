export detect_rings

"""
    detect_rings(img, radii; kwargs...)

Detect ring-shaped objects in `img` with given `radii`
using a variation of the Hough circle transform.

**Arguments**
- `img`: image (`AbstractMatrix`) to perform detection on
- `radii`: collection of integer values for desired ring radii

**Keywords**
- `σ::Real=1.2`: standard deviation of Gaussian filter for Canny edge detection
- `hi::Real=99`: upper percentile threshold for Canny
- `lo::Real=50`: lower percentile threshold for Canny
- `k=KernelFactors.ando5`: kernel for phase image evaluation
- `min_dist::Integer=minimum(radii)`: window size for search of local maxima
- `min_votes::Integer=1`: minimum number of votes for ring candidates
- `vote_threshold::Real=π/6`: minimum angular extent of rings (0, 2π)
"""
function detect_rings(
    img::AbstractMatrix, radii::AbstractVector{<:Integer};
    σ::Real=1.2, # smoothing size for Canny
    hi::Real=99, lo::Real=50, # percentile thresholds for Canny
    k=KernelFactors.ando5, # kernel for gradient evaluation
    min_dist::Integer=minimum(radii), # minimal distance between rings
    min_votes::Integer=1, # minimum votes for hotspots
    vote_threshold::Real=π/4, # fractional threshold (0,2π) for rings
)
    # evaluate edges and phase of input image
    img_edges = canny(img, Percentile.((hi, lo)), σ)
    img_phase = phase(imgradients(img, k)...)
    # accumulate votes in (x-y-r) space and identify hotspots
    accumulator = hough_accumulator(img_edges, img_phase, radii)
    # TODO:
    # find hotspots in hough_accumulator to optimize performance
    hotspots = findall(accumulator .> min_votes)
    # smooth and reweight hotspots (Afik 2015 Sci Rep)
    accumulator_proc = process_votes(hotspots, accumulator, radii)
    # sort and filter ring candidates
    detections = find_ring_candidates(accumulator, accumulator_proc, min_dist)
    filter!(p -> accumulator_proc[p] > vote_threshold, detections)
    rings = [
        Ring(CartesianIndex{2}(p[1],p[2]), radii[p[3]])
        for p in detections
    ]
    # WARN:
    # accumulator values are currently returned for
    # debugging and calibration purposes,
    # may be removed in the future
    rings, accumulator[detections], accumulator_proc[detections]
end

function hough_accumulator(
    img_edges::AbstractArray{Bool,2},
    img_phase::AbstractArray{<:Number,2},
    radii::AbstractRange{<:Integer}
)
    rows, cols = size(img_edges)
    nradii = length(radii)
    accumulator = zeros(Float64, rows+1, cols+1, nradii)
    for j in axes(img_edges, 2)
        for i in axes(img_edges, 1)
            if img_edges[i,j]
                sinθ = -cos(img_phase[i,j]);
                cosθ = sin(img_phase[i,j]);
                for (ridx, r) in enumerate(radii)
                    # vote in both directions from edge ij
                    x=(i+r*sinθ)
                    y=(j+r*cosθ)
                    vote!(accumulator, x, y, ridx)
                    x=(i-r*sinθ)
                    y=(j-r*cosθ)
                    vote!(accumulator, x, y, ridx)
                end
            end
        end
    end
    # HACK:
    # smoothing seems necessary to improve search of maxima
    # but this is likely suboptimal and may need better control
    return imfilter(accumulator, Kernel.gaussian((1,1,1)))
end

function vote!(acc, x, y, r)
    fx = Int(floor(x))
    fy = Int(floor(y))
    for i in fx:fx+1
        for j in fy:fy+1
            if checkbounds(Bool, acc, i, j, r)
                @inbounds acc[i, j, r] += 1
            end
        end
    end
end

function process_votes(hotspots, accumulator, radii)
    accumulator_proc = zero(accumulator)
    nx, ny = size(accumulator)
    for p in hotspots
        x0, y0, ridx = p.I
        r = radii[ridx]
        σ = 0.05*r + 0.25 # gaussian smoothing size (Afik 2015 Sci Rep)
        # σ = 0.25*r + 0.25 # gaussian smoothing size
        w = floor(Int, 3σ) + 1 # window size
        xrng = max(x0-w,1):min(x0+w,nx)
        yrng = max(y0-w,1):min(y0+w,ny)
        v = accumulator[xrng, yrng, ridx] # raw hotspot votes
        dx = xrng .- x0
        dy = yrng .- y0
        # gaussian weights around hotspot
        gx = @. exp(-dx^2 / (2*σ^2))
        gy = @. exp(-dy^2 / (2*σ^2))
        g = gx * gy'
        g ./= sum(g) # normalize weights to 1
        # WARN:
        # the 1/r division biases too strongly towards small r
        # but now this is not really normalized to (0,2π)
        accumulator_proc[p] = sum(g .* v)# / r
    end
    accumulator_proc
end

function find_ring_candidates(accumulator, accumulator_proc, min_dist)
    d = max(3, min_dist)
    # WARN:
    # this is not really a minimum distance
    # it is only a min distance between similar-radius (±2) objects
    window = (d, d, 5)
    sort(
        findlocalmaxima(accumulator_proc; window);
        # sort by vote and radius
        by=(p -> (accumulator[p], p[3])),
        rev=true # larger votes and radii first
    )
end
