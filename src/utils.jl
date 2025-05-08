export safeslice

function safeslice(M, i, j, r)
    si, sj = size(M)
    R = r+1
    imin = max(1, i-R)
    imax = min(si, i+R)
    jmin = max(1, j-R)
    jmax = min(sj, j+R)
    @view M[imin:imax, jmin:jmax]
end
