export hough_accumulator

function hough_accumulator(
    vid::AbstractVector, radii::AbstractVector;
    kwargs...
)
    map(img -> hough_accumulator(img, radii; kwargs...), vid)
end
function hough_accumulator(img::AbstractMatrix, radii::AbstractVector;
    σ::Real=1.2, # smoothing size for Canny
    hi::Real=99, lo::Real=50, # percentile thresholds for Canny
    k=Kernel.ando5, # kernel for gradient evaluation
    min_dist::Real=minimum(radii), # minimal distance between rings
    vote_threshold::Integer=1, # Hough voting threshold
)
    # canny edge detection
    img_edges = canny(img, (Percentile(hi), Percentile(lo)), σ)
    # phase of image gradient
    dx, dy = imgradients(img, k)
    img_phase = phase(dx, dy)
    # find rings
    c, r, A = custom_hough(img_edges, img_phase, radii;
        min_dist, vote_threshold
    )
    c, r, A
end

function custom_hough(
        img_edges::AbstractArray{Bool,2},
        img_phase::AbstractArray{<:Number,2},
        radii::AbstractRange{<:Integer};
        scale::Number=1,
        min_dist::Number=minimum(radii),
        vote_threshold::Number=minimum(radii)*min(scale, length(radii)))

    rows,cols=size(img_edges)

    non_zeros=CartesianIndex{2}[]
    centers=CartesianIndex{2}[]
    circle_centers=CartesianIndex{2}[]
    circle_radius=Int[]
    accumulator_matrix=zeros(Int, Int(floor(rows/scale))+1, Int(floor(cols/scale))+1)

    function vote!(accumulator_matrix, x, y)
        fx = Int(floor(x))
        fy = Int(floor(y))

        for i in fx:fx+1
            for j in fy:fy+1
                if checkbounds(Bool, accumulator_matrix, i, j)
                    @inbounds accumulator_matrix[i, j] += 1
                end
            end
        end
    end

    for j in axes(img_edges, 2)
        for i in axes(img_edges, 1)
            if img_edges[i,j]
                sinθ = -cos(img_phase[i,j]);
                cosθ = sin(img_phase[i,j]);

                for r in radii
                    x=(i+r*sinθ)/scale
                    y=(j+r*cosθ)/scale
                    vote!(accumulator_matrix, x, y)

                    x=(i-r*sinθ)/scale
                    y=(j-r*cosθ)/scale
                    vote!(accumulator_matrix, x, y)
                end
                push!(non_zeros, CartesianIndex{2}(i,j));
            end
        end
    end

    for i in findlocalmaxima(accumulator_matrix)
        if accumulator_matrix[i]>vote_threshold
            push!(centers, i);
        end
    end

    @noinline sort_by_votes(centers, accumulator_matrix) = sort!(centers, lt=(a, b) -> accumulator_matrix[a]>accumulator_matrix[b])

    sort_by_votes(centers, accumulator_matrix)

    dist(a, b) = sqrt(sum(abs2, (a-b).I))

    f = CartesianIndex(map(r->first(r), axes(accumulator_matrix)))
    l = CartesianIndex(map(r->last(r), axes(accumulator_matrix)))
    radius_accumulator=Vector{Int}(undef, Int(floor(dist(f,l)/scale)+1))

    for center in centers
        center=(center-1*oneunit(center))*scale
        fill!(radius_accumulator, 0)

        too_close=false
        for circle_center in circle_centers
            if dist(center, circle_center)< min_dist
                too_close=true
                break
            end
        end
        if too_close
            continue;
        end

        for point in non_zeros
            r=Int(floor(dist(center, point)/scale))
            if radii.start/scale<=r<=radii.stop/scale
                radius_accumulator[r+1]+=1
            end
        end

        voters, radius = findmax(radius_accumulator)
        radius=(radius-1)*scale;

        if voters>vote_threshold
            push!(circle_centers, center)
            push!(circle_radius, radius)
        end
    end
    return circle_centers, circle_radius, accumulator_matrix
end

