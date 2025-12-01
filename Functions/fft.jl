using FFTW
using CMBLensing: Cℓs, m_rfft, Map
using Statistics: mean


"""
    get_Cℓ_fft(ϕ1, ϕ2=ϕ1; nbins=300)

Compute the auto / cross power spectrum C_ℓ(ϕ1,ϕ2) on a flat-sky
Lambert map using FFTs. Returns a `Cℓs(ℓ, Cℓ)` object.
"""
function get_Cℓ_fft(ϕ1, ϕ2=ϕ1; nbins=300)

    # Real-space maps (Ny×Nx) and projection metadata
    m1 = Map(ϕ1)
    m2 = Map(ϕ2)

    a = m1.arr
    b = m2.arr
    Ny, Nx = size(a)
    @assert size(b) == (Ny, Nx)

    proj = m1.proj              # ProjLambert
    θpix_rad = proj.θpix * (π/180) / 60   # arcmin → radians

    # --- FFTs (rfft keeps only independent half-plane) ---
    F1 = m_rfft(a, (1,2))
    F2 = m_rfft(b, (1,2))
    NyF, NxF = size(F1)

    # Cross spectrum in Fourier space
    Pk = real.(F1 .* conj.(F2))

    # Normalise to power per steradian
    area_sr = Nx * Ny * θpix_rad^2
    Pk ./= area_sr

    # --- Build ℓ grid matching rfft layout ---
    # proj.ℓx, proj.ℓy are 1D arrays; keep the parts actually present in F1
    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓx2D = repeat(ℓx', NyF, 1)
    ℓy2D = repeat(ℓy , 1,  NxF)
    ℓmap = @. sqrt(ℓx2D^2 + ℓy2D^2)

    # --- Radial binning in ℓ ---
    ℓmin, ℓmax = minimum(ℓmap), maximum(ℓmap)
    edges = range(ℓmin, ℓmax; length=nbins+1)

    Cℓ  = zeros(Float64, nbins)
    ℓc  = zeros(Float64, nbins)

    for i in 1:nbins
        mask = (edges[i] .<= ℓmap .< edges[i+1])   # BitMatrix NyF×NxF
        if any(mask)
            vals = Pk[mask]
            Cℓ[i] = mean(vals)
            ℓc[i] = mean(ℓmap[mask])
        else
            Cℓ[i] = NaN
            ℓc[i] = (edges[i] + edges[i+1]) / 2
        end
    end

    return Cℓs(ℓc, Cℓ)
end

function bin_spectrum(ells, vals; ΔL=300)
    # Define bin edges
    Lmin = minimum(ells)
    Lmax = maximum(ells)
    edges = collect(Lmin:ΔL:Lmax)

    # Bin centres
    centers = 0.5 .* (edges[1:end-1] .+ edges[2:end])

    # Output array (same length as centres)
    binned = similar(centers)

    # Loop over bins
    for i in eachindex(centers)
        m = (ells .>= edges[i]) .& (ells .< edges[i+1])

        if any(m)
            binned[i] = mean(vals[m])
        else
            binned[i] = NaN
        end
    end

    return centers, binned
end


