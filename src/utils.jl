export imgread, safeslice

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
