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
    posA = ntuple(_ -> 0, 2)
    posB = ntuple(i -> i==1 ? ceil(Int, maxdist) : 0, 2)
    A = Ring(CartesianIndex(posA), 0)
    B = Ring(CartesianIndex(posB), 0)
    return cost(A, B; kwargs...)
end

