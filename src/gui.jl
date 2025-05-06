export tune_ring_detection

using .GLMakie

function tune_ring_detection(
    img::AbstractMatrix;
    fout::AbstractString="",
    colormap=:bone,
    ring_strokecolor=:red,
    ring_strokewidth=1,
)
    # initialize
    channel = Channel{Bool}(1)
    fig = Figure(size=(900, 900), fontsize=24)
    nx, ny = size(img)
    ax = GLMakie.Axis(fig[1:4,1:2]; aspect=nx/ny)
    hidespines!(ax)
    hidedecorations!(ax)
    # controls
    sg1 = SliderGrid(fig[5,1],
        (label="α threshold", range=1.05:0.05:2, startvalue=1.05),
        (label="min radius", range=1:50, startvalue=2),
        (label="max radius", range=1:50, startvalue=10),
        (label="σ canny", range=1:0.1:20, startvalue=1)
    )
    α, rmin, rmax, σ = [s.value for s in sg1.sliders]
    sg2 = SliderGrid(fig[5,2],
        (label="min distance", range=1:50, startvalue=10),
        (label="vote threshold", range=1:50, startvalue=1),
        (label="β filter", range=0:0.1:10, startvalue=5),
    )
    min_dist, vote_threshold, β = [s.value for s in sg2.sliders]
    # get rings
    img_pro = @lift(sharpen(img; α=$α))
    hough = @lift(
        detect_rings(
            $img_pro, ($rmin):($rmax);
            σ=$σ,
            min_dist=$min_dist,
            vote_threshold=$vote_threshold,
            β=$β
        )
    )
    rings = @lift([
        Circle(location(ring), radius(ring)) for ring in $hough
    ])
    # display
    heatmap!(ax, img_pro; colormap)
    poly!(ax, rings;
        color=:transparent,
        strokewidth=ring_strokewidth,
        strokecolor=ring_strokecolor,
    )
    # output parameter values on exit
    on(events(fig.scene).keyboardbutton) do event
        if !isopen(fig.scene)
            io = isempty(fout) ? stdout : open(fout, "w")
            println(io, "α = $(α[])")
            println(io, "range_radii = $(rmin[]):$(rmax[])")
            println(io, "σ = $(σ[])")
            println(io, "min_dist = $(min_dist[])")
            println(io, "vote_threshold = $(vote_threshold[])")
            println(io, "β = $(β[])")
            !isempty(fout) && close(io)
            put!(channel, true)
        end
    end
    display(fig)
    take!(channel)
    return
end
