export Ring, radius, location, location_raw

@kwdef struct Ring <: ParticleTracking.AbstractBlob{Nothing,2}
    location::NTuple{2,Float64}
    location_raw::CartesianIndex{2} = CartesianIndex(
        any(isnan, location) ? ntuple(zero, N) : round.(Int, location)
    )
    radius::Float64 = 0
end
function Ring(c::CartesianIndex{2}, r)
    Ring(float.(c.I), c, float(r))
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
    @eval ParticleTracking.$fs(r::Ring) = 0
end
