
using CMBLensing: Cℓs, m_rfft, Map
using Statistics: mean, std

"""
    get_Cℓ_fft(ϕ1, ϕ2=ϕ1; nbins=300)

Compute the binned cross-power spectrum of two fields via FFT.
Returns a `Cℓs` object with fields `.ℓ` and `.Cℓ`.
"""
function get_Cℓ_fft(ϕ1, ϕ2=ϕ1; nbins=300)
    m1 = Map(ϕ1)
    m2 = Map(ϕ2)

    a = m1.arr
    b = m2.arr

    Ny, Nx = size(a)
    proj = m1.proj
    θpix_rad = proj.θpix * (π / 180) / 60

    F1 = m_rfft(a, (1, 2))
    F2 = m_rfft(b, (1, 2))

    Pk = real.(F1 .* conj.(F2))
    Pk ./= Ny * Nx * θpix_rad^2

    NyF, NxF = size(F1)
    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓmap = @. sqrt((ℓx')^2 + ℓy^2)

    ℓmin, ℓmax = minimum(ℓmap), maximum(ℓmap)
    edges = range(ℓmin, ℓmax; length=nbins + 1)

    Cs = zeros(nbins)
    ℓc = zeros(nbins)

    for i in 1:nbins
        mask = (edges[i] .<= ℓmap .< edges[i+1])
        if any(mask)
            Cs[i] = mean(Pk[mask])
            ℓc[i] = mean(ℓmap[mask])
        else
            ℓc[i] = (edges[i] + edges[i+1]) / 2
            Cs[i] = NaN
        end
    end

    return Cℓs(ℓc, Cs)
end

"""
    bin_spectrum(ells, vals; ΔL=300)

Bin a 1D spectrum into uniform ΔL-wide multipole bins.
Returns `(centres, binned_values)`.
"""
function bin_spectrum(ells, vals; ΔL=300)
    Lmin, Lmax = minimum(ells), maximum(ells)
    edges = collect(Lmin:ΔL:Lmax)
    centers = @. 0.5 * (edges[1:end-1] + edges[2:end])

    binned = similar(centers)
    for i in eachindex(centers)
        mask = (ells .>= edges[i]) .& (ells .< edges[i+1])
        binned[i] = any(mask) ? mean(vals[mask]) : NaN
    end

    return centers, binned
end

"""
    bin_stat(x, y; nbins=20)

Bin `y` values by `x` into `nbins` equal-width bins.
Returns `(bin_centres, mean_y_per_bin)`.
"""
function bin_stat(x, y; nbins=20)
    xvec = vec(x)
    yvec = vec(y)

    lo, hi = minimum(xvec), maximum(xvec)
    edges = range(lo, hi; length=nbins + 1)

    means = Float64[]
    centres = Float64[]

    for i in 1:nbins
        binmask = (edges[i] .<= xvec .< edges[i+1])
        push!(centres, 0.5 * (edges[i] + edges[i+1]))

        if any(binmask)
            push!(means, mean(yvec[binmask]))
        else
            push!(means, NaN)
        end
    end
    return centres, means
end

"""
    bin_stat_err(x, y; nbins=20)

Bin `y` values by `x` into `nbins` equal-width bins.
Returns `(bin_centres, mean_y, sem_y, counts)` where `sem_y` is the
standard error of the mean (σ/√N) per bin.
"""
function bin_stat_err(x, y; nbins=20)
    xvec = vec(x)
    yvec = vec(y)

    lo, hi = minimum(xvec), maximum(xvec)
    edges = range(lo, hi; length=nbins + 1)

    means   = Float64[]
    sems    = Float64[]
    centres = Float64[]
    counts  = Int[]

    for i in 1:nbins
        binmask = (edges[i] .<= xvec .< edges[i+1])
        push!(centres, 0.5 * (edges[i] + edges[i+1]))
        n = count(binmask)
        push!(counts, n)

        if n > 1
            vals = yvec[binmask]
            push!(means, mean(vals))
            push!(sems, std(vals) / sqrt(n))
        elseif n == 1
            push!(means, yvec[binmask][1])
            push!(sems, NaN)
        else
            push!(means, NaN)
            push!(sems, NaN)
        end
    end
    return centres, means, sems, counts
end