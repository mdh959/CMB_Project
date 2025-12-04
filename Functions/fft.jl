using FFTW
using CMBLensing: Cℓs, m_rfft, m_irfft, Map
using Statistics: mean

function grad_fft_local(T)
    m = Map(T)
    proj = m.proj
    Ny, Nx = size(m.arr)
    g = m.arr
    θpix = 0.74  # arcmin
    θpix_rad = θpix * (π/180) / 60

    F = m_rfft(g, (1,2))
    NyF, NxF = size(F)

    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓx2D = repeat(ℓx', NyF, 1)
    ℓy2D = repeat(ℓy , 1, NxF)

    dTdx = m_irfft((im .* ℓx2D) .* F, Ny, (1,2)) * θpix_rad
    dTdy = m_irfft((im .* ℓy2D) .* F, Ny, (1,2)) * θpix_rad

    grad_mag = sqrt.(dTdx.^2 .+ dTdy.^2)
    return dTdx, dTdy, grad_mag
end

"""
    get_Cℓ_fft
"""
function get_Cℓ_fft(ϕ1, ϕ2=ϕ1; nbins=300)
    m1 = Map(ϕ1)
    m2 = Map(ϕ2)

    a = m1.arr
    b = m2.arr

    Ny, Nx = size(a)
    proj = m1.proj
    θpix_rad = proj.θpix * (π/180)/60

    F1 = m_rfft(a, (1,2))
    F2 = m_rfft(b, (1,2))

    Pk = real.(F1 .* conj.(F2))
    Pk ./= Ny*Nx*θpix_rad^2

    NyF, NxF = size(F1)
    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓmap = @. sqrt((ℓx')^2 + ℓy^2)

    ℓmin, ℓmax = minimum(ℓmap), maximum(ℓmap)
    edges = range(ℓmin, ℓmax; length=nbins+1)

    Cs  = zeros(nbins)
    ℓc  = zeros(nbins)

    for i in 1:nbins
        mask = (edges[i] .<= ℓmap .< edges[i+1])
        if any(mask)
            Cs[i] = mean(Pk[mask])
            ℓc[i] = mean(ℓmap[mask])
        else
            ℓc[i] = (edges[i] + edges[i+1])/2
            Cs[i] = NaN
        end
    end

    return Cℓs(ℓc, Cs)
end

function bin_spectrum(ells, vals; ΔL=300)
    Lmin, Lmax = minimum(ells), maximum(ells)
    edges = collect(Lmin:ΔL:Lmax)
    centers = @. 0.5*(edges[1:end-1] + edges[2:end])

    binned = similar(centers)
    for i in eachindex(centers)
        mask = (ells .>= edges[i]) .& (ells .< edges[i+1])
        binned[i] = any(mask) ? mean(vals[mask]) : NaN
    end

    return centers, binned
end
