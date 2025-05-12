module RingDetection

using ImageFeatures
using Images
using TiffImages
using StatsBase
using ParticleTracking
using GeometryBasics

#== Core ==#
include("tracking_interface.jl")
include("utils.jl")
include("hough.jl")

#== Interactive GUI if GLMakie available ==#
using Requires
function __init__()
    @require GLMakie="e9467ef8-e4e7-5192-8a1a-b1aee30e663a" include("gui.jl")
end

end # module
