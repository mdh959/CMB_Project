
using CMBLensing: Map, m_rfft, m_irfft

"""
    grad_fft(field)

Compute the gradient of a CMB field via FFT.
Returns `(dTdx, dTdy, grad_mag)` as real-space arrays.
"""
function grad_fft(field)
    m = Map(field)
    proj = m.proj
    Ny, Nx = size(m.arr)
    g = m.arr

    F = m_rfft(g, (1, 2))
    NyF, NxF = size(F)

    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓx2D = repeat(ℓx', NyF, 1)
    ℓy2D = repeat(ℓy, 1, NxF)

    dTdx = m_irfft((im .* ℓx2D) .* F, Ny, (1, 2))
    dTdy = m_irfft((im .* ℓy2D) .* F, Ny, (1, 2))

    grad_mag = sqrt.(dTdx .^ 2 .+ dTdy .^ 2)

    return dTdx, dTdy, grad_mag
end