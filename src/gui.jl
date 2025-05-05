export tune_ring_detection

using .GLMakie

function tune_ring_detection(
    img::AbstractMatrix;
    preprocess=sharpen,
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
        (label="min radius", range=1:50, startvalue=2),
        (label="max radius", range=1:50, startvalue=10),
        (label="σ canny", range=1:0.1:20, startvalue=1)
    )
    rmin, rmax, σ = [s.value for s in sg1.sliders]
    sg2 = SliderGrid(fig[5,2],
        (label="min distance", range=1:50, startvalue=10),
        (label="vote threshold", range=1:50, startvalue=1),
        (label="β filter", range=0:0.1:10, startvalue=5),
    )
    min_dist, vote_threshold, β = [s.value for s in sg2.sliders]
    # get rings
    img_pro = sharpen(img)
    hough = @lift(
        detect_rings(
            img_pro, ($rmin):($rmax);
            σ=$σ,
            min_dist=$min_dist,
            vote_threshold=$vote_threshold,
            β=$β
        )
    )
    centers = @lift(first($hough))
    radii = @lift(last($hough))
    rings = @lift([
        Circle(Point2f(c.I), r) for (c, r) in zip($centers, $radii)
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
            println("range_radii = $(rmin[]):$(rmax[])")
            println("σ = $(σ[])")
            println("min_dist = $(min_dist[])")
            println("vote_threshold = $(vote_threshold[])")
            println("β = $(β[])")
            put!(channel, true)
        end
    end
    display(fig)
    take!(channel)
end
