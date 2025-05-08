export tune_preprocessing, tune_ring_detection

using .GLMakie

# TODO: this function is of much broader utility,
# should be moved to its own self-contained package
"""
    tune_preprocessing(img, preprocess, args; fout, colormap, layout)

Tune image preprocessing parameters via an interactive GUI.
Allows for tuning of up to 6 parameters simultaneously.
When the GUI is closed, the final parameter set is printed.

*Arguments*:

- `img`: the image to be used as a sample for tuning
- `preprocessing`: an arbitrary preprocessing function
 that must have a signature of the form `f(img; kwargs...)`.
- `args`: a `NamedTuple` of argument names and value ranges
 to be passed onto the `preprocessing` function.
 E.g. if `preprocessing` is `f(img; a=1, b=2)` then
 `args` can be of the form `(a=1:10, b=-5:5)`.
 `args` does not need to contain all the kwargs of
 `preprocessing`, but cannot contain elements that
 `preprocessing` would not accept as kwargs.

*Keyword arguments*:

- `fout` (defaults to `""`): path to a file where the
 parameter values are logged upon closing the GUI.
 If unspecified, the values are logged to `stdout`.
- `colormap` (defaults to `:bone`): colormap for the image.
- `layout` (defaults to `:h`): `:h` to show raw and processed
 image side by side horizontally, `:v` to align them vertically,
 `:p` to show only the processed image.
"""
function tune_preprocessing(
    img::AbstractMatrix, preprocess::Function, args::NamedTuple;
    fout="", colormap=:bone, layout=:h
)
    # initialize
    channel = Channel{Bool}(1)
    fig = Figure(size=(1200, 800), fontsize=24)
    nx, ny = size(img)
    if layout == :h
        ax0 = GLMakie.Axis(fig[1:4, 1]; aspect=nx/ny)
        ax = GLMakie.Axis(fig[1:4, 2]; aspect=nx/ny)
        hidespines!(ax0)
        hidedecorations!(ax0)
        heatmap!(ax0, img; colormap)
        linkaxes!(ax, ax0)
    elseif layout == :v
        ax0 = GLMakie.Axis(fig[1:2, 1:2]; aspect=nx/ny)
        ax = GLMakie.Axis(fig[3:4, 1:2]; aspect=nx/ny)
        hidespines!(ax0)
        hidedecorations!(ax0)
        heatmap!(ax0, img; colormap)
        linkaxes!(ax, ax0)
    elseif layout == :p
        ax = GLMakie.Axis(fig[1:4, 1:2]; aspect=nx/ny)
    else
        @error "Invalid layout value. Must be :h, :v or :p."
    end
    hidespines!(ax)
    hidedecorations!(ax)
    # display raw image
    # controls
    n_sliders = length(args)
    if n_sliders > 6
        @warn "Only up to 6 parameters can be tuned at a time"
    end
    arg_names = keys(args)
    arg_vals = values(args)
    # first 3 parameters
    sg1 = SliderGrid(fig[5,1],
        [
            (label=string(arg_names[i]), range=arg_vals[i])
            for i in 1:min(3, n_sliders)
        ]...
    )
    # next 3 parameters
    if n_sliders > 3
        sg2 = SliderGrid(fig[5,2],
            [
                (label=string(arg_names[i]), range=arg_vals[i])
                for i in 4:min(6, n_sliders)
            ]...
        )
    end
    # store observables by name in a dict
    current_vals = Dict(
        arg_names[n] =>
            n > 3 ? sg2.sliders[n-3].value : sg1.sliders[n].value
        for n in 1:min(n_sliders, 6)
    )
    # collect all parameters in a unique observable
    observer = if n_sliders > 3
        map((a...) -> [a...], vcat(
            [s.value for s in sg1.sliders],
            [s.value for s in sg2.sliders]
        )...)
    else
        map((a...) -> [a...], [s.value for s in sg1.sliders]...)
    end
    # show processed image
    img_pro = Observable(preprocess(img;
        [k => current_vals[k][] for k in arg_names]...
    ))
    heatmap!(ax, img_pro; colormap)
    # re-process image when observer is updated
    on(observer) do _
        img_pro[] = preprocess(img;
            [k => current_vals[k][] for k in arg_names]...
        )
    end
    # output parameters on exit
    on(events(fig.scene).keyboardbutton) do event
        if !isopen(fig.scene)
            io = isempty(fout) ? stdout : open(fout, "w")
            for key in keys(current_vals)
                println(io, "$key = $(current_vals[key][])")
            end
            !isempty(fout) && close(io)
            put!(channel, true)
        end
    end
    display(fig)
    take!(channel)
    return
end

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
        Circle(Point2f(location(ring)), radius(ring)) for ring in $hough
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
