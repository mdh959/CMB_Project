module MAPNormalization

using CMBLensing
using LinearAlgebra

export compute_RL_iterative, compute_WL, rescale_phi_fourier

"""
    compute_RL_iterative(ds; niter=6)

Iterative calculation of the MAP normalization response R_L, following
Sec. II.B of Legrand & Carron (2022).

- Uses the QE *formula* for N_L^(0), but with partially–delensed
  lensing spectra C_L^{φφ,part} in the weights, as described in the
  three bullets under Eq. (2.11).

Returns `(RL, N0, rhoL, Cphi_part_fid)` where

- `RL`              - response R_L (vector on the φ multipole grid)
- `N0`              - final QE-like N_L^(0) with converged spectra
- `rhoL`            - correlation coefficient ρ_L
- `Cphi_part_fid`   - converged partially–delensed fiducial C_L^{φφ}
"""
function compute_RL_iterative(ds; niter::Int=6)

    # Fiducial φφ spectrum (vector on L-grid)
    Cphi_fid = diag(ds.Cϕ)
    Cphi_part_fid = copy(Cphi_fid)

    N0   = similar(Cphi_fid)
    rhoL = similar(Cphi_fid)

    for it in 1:niter
        # --- 1. Build dataset with current partially-delensed φ spectrum
        ds_tmp = ds(Cϕ = Diagonal(Cphi_part_fid))

        # --- 2. QE N^(0) using partially-delensed spectra as weights
        qe = quadratic_estimate(ds_tmp; weights=:lensed, wiener_filtered=false)
        N0 .= diag(qe.Nϕ)

        # ρ_L = C_L^{φφ,fid} / (C_L^{φφ,fid} + N_L^(0))
        @. rhoL = Cphi_fid / (Cphi_fid + N0 + eps(Float32))

        # --- 3. Update partially-delensed spectra: (1 - ρ_L^2) C_L^{φφ}
        @. Cphi_part_fid = (1 - rhoL^2) * Cphi_fid
    end

    # Final response: QE normalization with converged spectra
    ds_final = ds(Cϕ = Diagonal(Cphi_part_fid))
    qe_final = quadratic_estimate(ds_final; weights=:lensed, wiener_filtered=false)
    RL = 1 ./ diag(qe_final.Nϕ)

    return RL, N0, rhoL, Cphi_part_fid
end


"""
    compute_WL(RL, Cphi_fid)

Compute the MAP Wiener filter W_L from Eq. (2.8):

    W_L = Cphi_fid / (Cphi_fid + 1/RL)

Inputs:
- `RL`        – response vector from `compute_RL_iterative`
- `Cphi_fid`  – fiducial C_L^{φφ,fid} (same grid as RL)

Returns `W_L` (same length as RL).
"""
function compute_WL(RL, Cphi_fid)
    return Cphi_fid ./ (Cphi_fid .+ 1 ./ (RL .+ eps(eltype(RL))))
end

"""
    rescale_phi_fourier(phiF, RL_bins, L_edges, proj)

Approximate un-Wiener + un-response rescaling for Fourier-space φ_MAP.
This is ONLY for map-level comparisons (e.g. compare_gradT_phi_errors),
NOT the correct spectral debiasing.

Arguments:
- phiF       :: BaseField{Fourier}    # φ_MAP or φ_joint output
- RL_bins    :: Vector{Float32}       # binned response (length nbins)
- L_edges    :: Vector{Float32}       # bin edges (nbins+1)
- proj       :: ProjLambert           # geometry, carries ℓ-grid

Returns: new BaseField{Fourier} with φ(L)/R_L in each bin.
"""
function rescale_phi_fourier(phiF, RL_bins, L_edges, proj)

    φF = phiF.arr             # complex Fourier coefficients (Ny × Nx_rfft)
    ℓ  = proj.ℓmag            # ℓ for each Fourier pixel

    RLN = length(RL_bins)
    φF_new = similar(φF)

    for idx in eachindex(φF)
        L = ℓ[idx]
        b = searchsortedfirst(L_edges, L) - 1

        if 1 ≤ b ≤ RLN && RL_bins[b] > 0
            φF_new[idx] = φF[idx] / RL_bins[b]
        else
            φF_new[idx] = 0
        end
    end

    # return a correct BaseField in Fourier space
    return BaseField(phiF.proj, Fourier(), φF_new)
end

end
