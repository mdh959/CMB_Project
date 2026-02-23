function debias_phi_with_WL(ϕ, WL_Cℓs; minW::Real=1e-3)
    ϕF   = Fourier(ϕ)
    ℓmag = Float64.(ϕF.metadata.ℓmag)

    # clamp ℓ to template support
    ℓmin = minimum(WL_Cℓs.ℓ)
    ℓmax = maximum(WL_Cℓs.ℓ)
    ℓcl  = clamp.(ℓmag, ℓmin, ℓmax)

    W = Float64.(WL_Cℓs.(ℓcl))

    # protect against NaN/neg/tiny W
    Wsafe = map(w -> (!isfinite(w) || w <= minW) ? 1.0 : w, W)

    return typeof(ϕF)(ϕF.arr ./ Wsafe, ϕF.metadata)
end