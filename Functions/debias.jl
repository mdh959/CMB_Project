using CMBLensing: Map, BaseField, Fourier

"""
    debias_phi_with_WL(ϕF, ℓc, W_L)

Debias a lensing potential φ in Fourier basis by dividing by the
empirical Wiener filter W_L (binned in ℓ).

Returns a real-space `Map`.
"""
function debias_phi_with_WL(ϕF::BaseField{Fourier}, ℓc::Vector{Float64}, W_L::Vector{Float64})
    proj = ϕF.proj
    φF   = ϕF.arr
    NyF, NxF = size(φF)

    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓx2D = repeat(ℓx', NyF, 1)
    ℓy2D = repeat(ℓy, 1, NxF)
    ℓmag = @. sqrt(ℓx2D^2 + ℓy2D^2)

    # Bin edges from centres
    nbins = length(ℓc)
    edges = similar(ℓc, nbins + 1)
    edges[2:nbins] .= 0.5 .* (ℓc[1:nbins-1] .+ ℓc[2:nbins])
    edges[1] = 0.0
    edges[end] = maximum(ℓmag) * 1.001

    # Work in Float64 to avoid truncation when input is Float32
    debiasF = zeros(Complex{Float64}, size(φF))
    for idx in eachindex(φF)
        L = ℓmag[idx]
        b = searchsortedfirst(edges, L) - 1
        if 1 ≤ b ≤ nbins && W_L[b] > 1e-6
            debiasF[idx] = Complex{Float64}(φF[idx]) / W_L[b]
        end
    end

    return Map(BaseField{Fourier}(debiasF, proj))
end