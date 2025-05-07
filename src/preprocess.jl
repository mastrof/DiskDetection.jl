export sharpen

function sharpen(img::AbstractMatrix; α::Real=1.05, σ::Real=1, w::Integer=3)
    med = median(img)
    proc = [u > α*med ? one(u) : zero(u) for u in img]
    proc = imfilter(proc, Kernel.gaussian(σ))
    proc = mapwindow(minimum, proc, (w,w))
    proc
end
