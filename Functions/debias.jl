# Debias a lensing potential estimate by dividing in Fourier space by the
# empirical transfer function W_L:
#
#   ϕ_deb(k) = ϕ_rec(k) / W_L(|k|)
#
# W_L = <C_L^{ϕ_true, ϕ_rec}> / <C_L^{ϕ_true, ϕ_true}> is the response of
# the estimator to the true signal, estimated from MC sims in normalization.jl.
# After debiasing: <C_L^{ϕ_deb, ϕ_true}> = C_L^{ϕ_true}.
#
# minW: floor on W_L below which modes are zeroed rather than amplified.
# Needed near and beyond lmax where W_L → 0 and division diverges.
# ℓmag is clamped to [ℓmin, ℓmax] of the WL grid before interpolating to
# avoid out-of-range extrapolation artefacts.
function debias_phi_with_WL(ϕ, WL_Cℓs; minW::Real=1e-8)
    ϕF   = Fourier(ϕ)
    ℓmag = Float64.(ϕF.metadata.ℓmag)

    ℓmin = minimum(WL_Cℓs.ℓ)
    ℓmax = maximum(WL_Cℓs.ℓ)
    ℓcl  = clamp.(ℓmag, ℓmin, ℓmax)   # stay within interpolation range

    W = Float64.(WL_Cℓs.(ℓcl))

    out = similar(ϕF.arr)
    @inbounds for i in eachindex(out)
        w = W[i]
        if !isfinite(w) || w <= minW
            out[i] = 0.0              # zero modes with negligible response
        else
            out[i] = ϕF.arr[i] / w
        end
    end

    return typeof(ϕF)(out, ϕF.metadata)
end