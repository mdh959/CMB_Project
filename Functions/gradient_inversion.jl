using CMBLensing: Map, m_rfft, m_irfft, FlatMap, LowPass, HighPass
using Statistics: mean


"""
    gi_estimate(ds; Lgrad=2000, Lhp=4000, Lmax=18000)

Gradient-Inversion (GI) lensing estimator (Hadzhiyska et al. 2019).

Valid for L > Lhp, where primary CMB is Silk-damped and the small-scale
approximation T̃(x) ≈ ∇T_large(x) · ∇φ(x) holds.  The paper recommends
Lhp ≳ 4000 for this approximation to be reliable.

Algorithm (O(N log N)) — global matched-filter per Fourier mode:

  In the constant-gradient limit, T̃_hp(L) ≈ i(L·g) φ̂(L), so:
      φ̂(L) = −i · FFT[(L·g(x)) · T_hp(x)](L) / <(L·g(x))²>_x

  Expanding bilinearly:
      A(L) = FFT[gx · T_hp](L)          (unnormalised DFT sum over pixels)
      B(L) = FFT[gy · T_hp](L)
      φ̂(L) = −i(Lx·A + Ly·B) / (Lx² σxx + 2LxLy σxy + Ly² σyy)

  where σij = <gi(x)gj(x)>_x  (per-pixel mean, NOT sum).
  Using mean gives φ̂ → DFT(ϕ_true), so W_L → 1 analytically.
  Using sum would give φ̂ → DFT(ϕ_true)/N², W_L → 1/N² ≈ 4×10⁻⁶.

  This is algebraically equivalent to the paper's practical GI estimator using
  the Eq. (29) inverse-noise weighting.  For each target mode L, the scalar
  factor N_L^TT cancels in the normalisation, so the implementation can be
  vectorised into the form below without approximation.

Returns a CMBLensing FlatMap of the GI φ estimate.
"""
function gi_estimate(ds; Lgrad=2000, Lhp=4000, Lmax=18000,
                     denom_floor_frac=1e-6)

    m    = Map(ds.d)
    proj = m.proj
    Ny   = size(m.arr, 1)

    # ── 1. Large-scale gradient (low-pass filtered) ──────────────────────────
    dTdx, dTdy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))

    # ── 2. High-pass filtered T (small-scale lensing signal) ─────────────────
    T_hp = Map(HighPass(Lhp) * ds.d).arr

    # ── 3. Real-space products → FFT ─────────────────────────────────────────
    A_F = m_rfft(dTdx .* T_hp, (1,2))
    B_F = m_rfft(dTdy .* T_hp, (1,2))

    # Global gradient variance — must use mean (per-pixel average) so that
    # the denominator normalises the DFT correctly and W_L → 1.
    # Using sum gives denom ∝ N² too large → φ̂ ∝ 1/N², W_L → 1/N² ≈ 0.
    σ_xx = mean(dTdx .^ 2)
    σ_xy = mean(dTdx .* dTdy)
    σ_yy = mean(dTdy .^ 2)

    # ── 4. Fourier grid ───────────────────────────────────────────────────────
    NyF, NxF = size(A_F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2

    # ── 5. GI formula: φ̂(L) = −i(Lx·A + Ly·B) / (Lx²Σxx + 2LxLyΣxy + Ly²Σyy)
    numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
    denom = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy

    # Threshold: suppresses modes where L ⊥ gradT (denom ≈ 0)
    denom_threshold = denom_floor_frac * max(σ_xx, σ_yy)
    L_mask = @. (L2 > Lhp^2) & (L2 < Lmax^2)

    φ_F = @. ifelse(L_mask & (abs(denom) > denom_threshold),
                    numer / denom, complex(0.0))

    # ── 6. Back to real space ─────────────────────────────────────────────────
    return FlatMap(real.(m_irfft(φ_F, Ny, (1,2))), proj)
end
