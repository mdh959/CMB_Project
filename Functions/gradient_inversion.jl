using CMBLensing: Map, m_rfft, m_irfft, FlatMap, LowPass, HighPass
using Statistics: mean


"""
    gi_estimate(ds; Lgrad=2000, Lhp=4000, Lmax=18000)

Gradient-Inversion lensing estimator following Hadzhiyska et al. 2019.

Works in the small-scale regime where T̃(x) ≈ ∇T_large(x) · ∇φ(x), i.e.
the lensed temperature is well approximated by a gradient of the large-scale
field dotted into the lensing deflection. This holds for L > ~4000 where the
primary CMB is Silk-damped.

The estimator in Fourier space is:
    φ̂(L) = −i · FFT[(L·g) · T_hp](L) / <(L·g)²>

which expands to:
    A(L) = FFT[gx · T_hp](L)
    B(L) = FFT[gy · T_hp](L)
    φ̂(L) = −i(Lx·A + Ly·B) / (Lx² σxx + 2LxLy σxy + Ly² σyy)

σij = <gi gj> taken as a per-pixel mean over the map (not a sum) — this is
what makes W_L → 1. Using a sum instead gives a denominator ~N² too large
and W_L → 0.
"""
function gi_estimate(ds; Lgrad=2000, Lhp=4000, Lmax=18000,
                     denom_floor_frac=1e-8)

    m    = Map(ds.d)
    proj = m.proj
    Ny   = size(m.arr, 1)

    # large-scale gradient from low-passed map
    dTdx, dTdy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))

    # small-scale temperature — lensing signal lives here
    T_hp = Map(HighPass(Lhp) * ds.d).arr

    # form real-space products and FFT
    A_F = m_rfft(dTdx .* T_hp, (1,2))
    B_F = m_rfft(dTdy .* T_hp, (1,2))

    # gradient covariance as per-pixel mean — this gives φ_F = m_rfft(φ_true) → W_L = 1
    σ_xx = mean(dTdx .^ 2)
    σ_xy = mean(dTdx .* dTdy)
    σ_yy = mean(dTdy .^ 2)

    # Fourier grid
    NyF, NxF = size(A_F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2

    numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
    denom = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy

    # zero out modes where denom is tiny (L perpendicular to gradient)
    denom_threshold = denom_floor_frac * max(σ_xx, σ_yy)
    L_mask = @. (L2 > Lhp^2) & (L2 < Lmax^2)

    φ_F = @. ifelse(L_mask & (abs(denom) > denom_threshold),
                    numer / denom, complex(0.0))

    return FlatFourier(φ_F, proj)
end


"""
    gi_estimate_corrected(ds; Lgrad=2000, Lmax=20000, denom_floor_frac=1e-4)

Pixel-exact GI estimator matching Boryana's Python code (real_filt_subtr_method.py).

For each output Fourier mode L=(ℓx,ℓy), computes a per-pixel Wiener weight:

    gprod(x,L)  = ℓx·∂T/∂x(x) + ℓy·∂T/∂y(x)        [real-space map, per mode]
    N_pix(x,L)  = powersum(L) / gprod(x,L)²            [per-pixel noise in φ]
    W(x,L)      = Cϕ(L) / (N_pix(x,L) + Cϕ(L))        [per-pixel Wiener weight]
    φ̂(L)        = FFT[W·T_hp/gprod][L] / (mean(W)·i)

powersum(L) = (B²·C_l^{T,unlensed} + N_l)(L), matching Blake 2.

gi_estimate_boryana uses scalar mean(gprod²) in place of per-pixel gprod² —
exact only in the constant-gradient limit. This function does one FFT per output
mode and is therefore much slower.
"""
function gi_estimate_corrected(ds; Lgrad=2000, Lhp=4000, Lmax=20000, denom_floor_frac=1e-8)

    m    = Map(ds.d)
    proj = m.proj
    Ny, Nx = size(m.arr)
    NyF, NxF = Ny ÷ 2 + 1, Nx

    # ── Pre-compute gradient and high-pass T (once) ───────────────────────────
    # Gradient uses LowPass(Lgrad); T_hp uses HighPass(Lhp).
    # Using Lhp=4000 (not Lgrad=2000) avoids including ℓ=2000-4000 modes in T_hp
    # where the GI approximation T_hp ≈ (L·∇T)φ breaks down (CMB-dominated, not lensing).
    gx, gy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))   # plain arrays, Ny×Nx
    T_lp = Map(LowPass(Lhp) * ds.d).arr
    T_hp = m.arr .- T_lp

    # Per-mode power spectra — matches Python: C_unlensed + N_raw/B²
    # Use unit FlatFourier to extract diagonals: works for both plain Diagonal
    # and ParamDependentOp (ds.Cf, ds.Cϕ are parameterised operators).
    f_ones  = FlatFourier(ones(ComplexF64, NyF, NxF), proj)
    B_diag  = real.(ds.B̂.diag.arr)[1:NyF, 1:NxF]          # B̂ is plain Diagonal
    Cf_diag = real.((ds.Cf * f_ones).arr)[1:NyF, 1:NxF]
    Cn_diag = real.((ds.Cn̂ * f_ones).arr)[1:NyF, 1:NxF]
    PS_2D   = @. Cf_diag + Cn_diag / max(B_diag^2, 1e-30)
    Cϕ_2D   = real.((ds.Cϕ * f_ones).arr)[1:NyF, 1:NxF]

    grad_rms  = sqrt(mean(gx .^ 2) + mean(gy .^ 2))
    floor_val = denom_floor_frac * grad_rms

    ℓx_arr = proj.ℓx[1:NxF]
    ℓy_arr = proj.ℓy[1:NyF]

    phi_e     = zeros(Float64, Ny, Nx)
    gprod_buf = similar(gx)
    W_buf     = similar(gx)
    φ_F       = zeros(ComplexF64, NyF, NxF)

    # ── Per-mode loop ─────────────────────────────────────────────────────────
    for j in 1:NyF
        ly = ℓy_arr[j]
        for i in 1:NxF
            lx = ℓx_arr[i]
            L2 = lx^2 + ly^2
            (L2 == 0 || L2 > Lmax^2) && continue

            PS   = PS_2D[j, i]
            Cphi = Cϕ_2D[j, i]
            Cphi <= 0 && continue

            # Per-pixel gradient projection with floor
            @. gprod_buf = lx * gx + ly * gy
            @. gprod_buf = ifelse(abs(gprod_buf) < floor_val,
                                  ifelse(gprod_buf >= 0.0, floor_val, -floor_val),
                                  gprod_buf)

            # Per-pixel Wiener weight and estimator map
            @. W_buf  = Cphi / (PS / gprod_buf^2 + Cphi)
            @. phi_e  = W_buf * T_hp / gprod_buf

            W_mean = mean(W_buf)
            W_mean < 1e-30 && continue

            # FFT → pick mode [j,i] → normalise (÷ mean(W)·i matches Python norm_c *= 1j)
            φ_F[j, i] = m_rfft(phi_e, (1, 2))[j, i] / (W_mean * im)
        end
    end

    return FlatFourier(φ_F, proj)
end


"""
    gi_estimate_boryana(ds)

GI estimator matching Boryana's original implementation. Uses explicit
T_hp = T - LowPass(Lhp)*T subtraction (matching her Tlens - Tlens_filt),
no lower-L cut on reconstructed modes, and Lmax=20000.
"""
function gi_estimate_boryana(ds; denom_floor_frac=1e-8)
    Lgrad = 2000
    Lhp   = 4000
    Lmax  = 20000

    m    = Map(ds.d)
    proj = m.proj
    Ny   = size(m.arr, 1)

    dTdx, dTdy, _ = grad_fft(Map(LowPass(Lgrad) * ds.d))

    # exact match to her Tlens - Tlens_filt where Tlens_filt = LowPass(2000)*Tlens
    T_lp = Map(LowPass(Lhp) * ds.d).arr
    T_hp = Map(ds.d).arr .- T_lp

    A_F = m_rfft(dTdx .* T_hp, (1,2))
    B_F = m_rfft(dTdy .* T_hp, (1,2))

    σ_xx = mean(dTdx .^ 2)
    σ_xy = mean(dTdx .* dTdy)
    σ_yy = mean(dTdy .^ 2)

    NyF, NxF = size(A_F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2

    numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
    denom = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy

    denom_threshold = denom_floor_frac * max(σ_xx, σ_yy)
    L_mask = @. L2 < Lmax^2  # no lower cut, matches her l_beg≈0

    φ_F = @. ifelse(L_mask & (abs(denom) > denom_threshold),
                    numer / denom, complex(0.0))

    return FlatFourier(φ_F, proj)
end
