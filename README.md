# CMB Lensing Reconstruction: GI, QE, and MAP

This project implements and compares three CMB lensing estimators — the Quadratic Estimator (QE), the Gradient-Inversion (GI) estimator of Hadzhiyska et al. (2019), and Joint MAP reconstruction — reproducing the key statistical figures from that paper at two noise levels: S4-like (1 µK-arcmin) and ultra-low (UL, 0.1 µK-arcmin).

---

## Summary

The core finding is that GI outperforms QE in cross-spectrum σ at UL noise (3031 vs 1632 SNR), while MAP has comparable or lower SNR to QE when measured in debiased spectra. This apparent MAP underperformance is entirely explained by the Wiener transfer function W_L, which shrinks MAP amplitudes at high L by up to a factor of 3.5×. In raw (un-debiased) cross-spectra, MAP is competitive with or better than GI at each L bin — the debiasing step amplifies MAP sigma by 1/W_L. The key numbers:

| L | W_MAP | σ_MAP_raw / σ_GI_raw | After 1/W_L amplification | Debiased comparison |
|-------|-------|----------------------|--------------------------|---------------------|
| 5000 | 0.649 | 0.88× | 0.88 / 0.649 = 1.35× | MAP worse |
| 7000 | 0.602 | 0.76× | 0.76 / 0.602 = 1.26× | MAP worse |
| 9000 | 0.436 | 1.10× | 1.10 / 0.436 = 2.52× | MAP worse |
| 11000 | 0.284 | 0.98× | 0.98 / 0.284 = 3.45× | MAP much worse |

GI avoids Wiener shrinkage because it directly inverts the gradient equation without a prior, so W_GI ≈ 0.84 on average (mode-recovery loss only, not prior shrinkage). MAP's W_L → 0 at high L where C_φφ → 0 and the prior dominates, which is optimal Bayesian behaviour — not a bug.

**Gradient response analysis (N_sim = 473).** The GI denominator D(L) = Lx²σxx + 2LxLy σxy + Ly²σyy fluctuates between realisations. The ratio R = D_realized / D_fiducial sets how much gradient power is actually in a given CMB realisation. Pearson correlation r between D and each estimator's output shows:

| Estimator | r(D, output) | Fisher z | Significance |
|-----------|-------------|----------|--------------|
| QE | 0.958 | 41.6σ | Expected — QE denominator ∝ ⟨\|∇T\|²⟩ by construction |
| GI | 0.078 | 1.7σ | Not significant — GI successfully removes gradient dependence |
| MAP | 0.223 | 4.9σ | Real effect, not noise — MAP has residual gradient-response coupling |

MAP's r = 0.22 indicates that its preconditioner (which uses fiducial N0 = QE noise, not per-realization gradient power) leaves a 4.9σ residual correlation with D. The converged MAP solution has the correct phases (ρ_L is comparable to GI) but slightly wrong amplitude in realisations with anomalous gradient power. This is the motivation for **Phase 2**: a realization-dependent preconditioner where the per-mode Fisher weight D_D(Lx,Ly) replaces the fiducial D_fid(L).

**r_iso.** The scalar `r_iso = (σ_xx_r + σ_yy_r) / (σ_xx_f + σ_yy_f)` is the angle-averaged ratio of the realised gradient variance to the fiducial (MC-mean) gradient variance for a given simulation. It summarises in a single number how much stronger or weaker the CMB gradient was in that realisation relative to the ensemble average. r_iso > 1 means the realised CMB had an unusually strong gradient (above-average lensing detectability for GI); r_iso < 1 means below-average. Typical values scatter around 1 with r_iso ≈ 1.038 observed in diagnostic runs. The 2D anisotropic version D_D(Lx,Ly) = Lx²σ_xx_r + 2LxLyσ_xy_r + Ly²σ_yy_r resolves this into directional gradient power and is the quantity that enters the GI normalisation and the Cϕ-realised diagnostic.

---

## Background

Gravitational lensing by large-scale structure deflects CMB photons, imprinting a lensing potential φ on the observed temperature map. Reconstructing φ from the data is central to extracting cosmological information (neutrino masses, dark energy) from next-generation experiments. Three complementary estimators are studied here:

- **QE** (Okamoto & Hu 2003): the minimum-variance quadratic estimator; optimal at high noise but biased at low noise by the N^(0) lensing bias.
- **GI** (Hadzhiyska et al. 2019): exploits the gradient-approximation T̃ ≈ T + ∇T·∇φ on small scales (ℓ ≳ 4000 where the primary CMB is Silk-damped) to directly invert for φ without a noise bias.
- **MAP joint** (Carron & Lewis 2017): iterative maximum a posteriori reconstruction; unbiased and near-optimal at all noise levels, at the cost of substantial computation.

---

## 1. Initial Exploration — `notebooks/project.ipynb`

The notebook establishes the simulation framework and validates each estimator on a single realisation before running at scale.

**Simulation setup.** A flat-sky patch is generated via `load_sim` with CMB-S4 parameters: `θpix = 0.7438046267475303` arcmin (512 × 512 pixels, patch ≈ 6.35° × 6.35°), noise `μKarcminT = 1.0`, beam FWHM 1 arcmin, and bandpass `LowPass(12000)`. The lensed CMB, unlensed CMB, lensing potential φ_true, and data model `ds` are all stored from a single `load_sim` call.

**QE.** The quadratic estimate is computed with `quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true)`. The Wiener-filter output ϕ_qe is used as a warm start for MAP.

**GI.** `gi_estimate(ds; Lgrad=2000, Lhp=4000, Lmax=12000)` is called directly. The correlation coefficient ρ_L is plotted alongside QE and MAP to confirm that GI recovers lensing information at high L.

**MAP joint.** `MAP_joint(ds, FieldTuple(ϕ=ϕ_start); nsteps=40, ...)` is run with the QE WF as the warm start. The log-posterior trajectory is recorded to check convergence. 40 gradient-descent steps with conjugate-gradient tolerance 1e-3 are sufficient for the posterior to plateau at S4 noise (visible in the history plot).

**MAP marginal.** `MAP_marg` is also explored in the notebook, integrating over the unlensed CMB field. This requires a mean-field update to account for the non-Gaussianity of the posterior; 100 simulations are used per mean-field step.

**Power spectra and ρ_L.** Cross-spectra ⟨C(φ_true, φ̂)⟩ and correlation coefficients ρ_L are plotted for all three estimators. At S4 noise, MAP joint significantly outperforms QE at L ≳ 3000, as expected. GI shows non-zero ρ_L at high L where QE noise is high.

**Empirical Wiener filter.** The transfer function W_L = ⟨C(φ_true, φ̂) / C(φ_true, φ_true)⟩ is estimated from Monte Carlo simulations and compared to the analytical expression W_L = C^φφ_L / (C^φφ_L + N^(0)_L). Good agreement confirms the QE normalisation is correct. The MAP W_L is generally larger than the QE W_L at high L, reflecting the improved reconstruction.

---

## 2. GI Algorithm — `Functions/gradient_inversion.jl`

Three variants of the GI estimator are implemented:

**`gi_estimate` (production estimator).** Uses the scalar gradient covariance σij = ⟨gi gj⟩ as the denominator normalisation. This is a mean over pixels (not a sum), which is essential: CMBLensing's rfft carries an implicit 1/N_pix, so using a pixel sum would make the denominator ~N_pix too large and drive W_L → 0. The reconstruction is restricted to Lhp < L < Lmax where the gradient approximation holds. A floor on the denominator (`denom_floor_frac = 1e-8 × max(σxx, σyy)`) suppresses modes where L is nearly perpendicular to ∇T and the projection is near zero.

**`gi_estimate_corrected`.** A per-pixel Wiener-weighted version that matches Hadzhiyska's Python reference code more closely. For each output Fourier mode L, a per-pixel Wiener weight W(x, L) = C^φφ(L) / (N_pix(x, L) + C^φφ(L)) is computed, where N_pix is the per-pixel noise in φ given the local gradient amplitude. This is exact but requires one FFT per output mode and is O(N_modes) times slower; it is used for validation only.

**`gi_estimate_boryana`.** A direct translation of Hadzhiyska's original Python implementation, with no lower-L cut on the reconstructed modes (matching her `l_beg ≈ 0`). Used to verify that `gi_estimate` recovers the same W_L when configured identically.

**`gi_n0_fixed_gradient_mc` (per-sim fg-MC N0).** Computes a per-realization noise bias N^(0)_GI by holding the gradient fixed from the actual data and drawing `nmc=20` independent null T_hp realizations with φ=0. For each draw i:

```
f_null ~ C̃_f,   n_null ~ Ĉ_n
T_hp,null = HighPass(L_hp) × (B̂ M̂ f_null + n_null)
φ̂_null(L) = −i (Lx FFT[gx T_hp,null] + Ly FFT[gy T_hp,null]) / denom(L)
N^(0)_fgmc(L) = (1/nmc) Σ_i C_L(φ̂_null,i)
```

where gx, gy and denom are frozen from the data sim. Because T_hp,null is independent of the data T_hp, the correlation between the N0 estimate and the signal is fully broken — unlike the analytical formula which shares the same denom. Because the gradient amplitude is fixed, N^(0)_fgmc tracks per-sim fluctuations in gradient power, unlike the global MC N0 which is a single constant. Null draws use the **lensed** CMB covariance C̃_f (`ds.Cf̃`) rather than unlensed C_f: above the high-pass cut L_hp ≈ 4000 the observed temperature variance is set by the lensed damping tail, so using unlensed C_f would underestimate the variance of T_hp,null and bias N^(0)_fgmc low. Draws remain independent Gaussian (no shared φ, no trispectrum signal).

The result is stored per-sim as `N0_gi_fgmc` in the phi_maps JLD2 files and is the production GI debiasing method in all figures.

**Analytical N0 — removed.** `gi_n0_analytical` (Hadzhiyska+2019 Eq. A1–A2) computes

```
N^(0)_anal(L) = N^{TT}(L) / denom(L)²,   N^{TT}(L) = C^{TT,unl}_L + C^n_L / B²_L
```

per sim. This uses the same denom(L) as the reconstruction, introducing a correlation between N^(0)_anal and the signal that inflates the off-diagonal covariance matrix and *increases* variance after subtraction. It is removed from the pipeline and replaced by `gi_n0_fixed_gradient_mc`.

---

## 3. Error Analysis — `run_error_analysis.jl`, `plot_mse_analysis.jl`

Before the large production run, a targeted pixel-space error analysis checks whether MAP outperforms QE as a function of the local CMB temperature gradient.

**Motivation.** The GI estimator relies on a large local gradient to constrain φ. Pixels with large |∇T| should have low reconstruction error regardless of estimator. Pixels with small |∇T| are gradient-noise limited and MAP may offer less improvement there.

**Method.** `run_error_analysis_all_debiased` runs 100 simulations (at Lmax=8000, 1 µK-arcmin), computes the pixel-space reconstruction error Δφ = φ_true − φ̂ for QE, MAP joint, and MAP marginal, and bins ⟨|Δφ|²⟩ by the local gradient magnitude |∇T|. The gradient is computed via `grad_fft` from the unlensed CMB temperature.

**Statistics.** The per-bin MSE is averaged across simulations and then converted to RMS. The mean-first-then-sqrt order is used (rather than sqrt per sim then average) because mean(√x) < √(mean(x)) by Jensen's inequality, which would give a downward-biased RMS. SEMs are propagated through the sqrt via the delta method: σ(√μ) ≈ σ_μ / (2√μ).

**Plot (`plot_mse_analysis.jl`).** Three panels: RMS(Δφ) vs |∇T| for each estimator (log scale, shaded ±1 SEM); the ratio RMS_QE / RMS_MAP; and pixel counts per |∇T| bin to confirm the bins are well-populated. The ratio panel shows where MAP provides the largest improvement over QE.

---

## 4. Production Run — `run_qe_gi_wl12k.jl`

This script runs 1000 QE/GI simulations and up to 100 MAP simulations at two noise levels, accumulating transfer functions W_L and storing φ maps for subsequent analysis. Three top-level calls are made:

1. **S4-like QE/GI** — 1 µK-arcmin, 1′ beam, RDN0 enabled, MAP skipped.
2. **UL QE/GI** — 0.1 µK-arcmin, 0.3′ beam, RDN0 enabled, MAP skipped.
3. **UL MAP** — same UL noise, MAP only, writes to separate `_ul_hess` output files with improved settings (see below).

**Two noise levels.**
- **S4-like**: 1 µK-arcmin, 1′ beam, Lmax=12000. This matches the target specifications for CMB-S4.
- **Ultra-low (UL)**: 0.1 µK-arcmin, 0.3′ beam, Lmax=12000. A noise floor well below any planned experiment, used to study the estimator behaviour in the signal-dominated regime.

**Transfer function W_L.** For each simulation,
```
W_L[ℓ] = C(φ_true, φ̂)[ℓ] / C(φ_true, φ_true)[ℓ]
```
is accumulated as a running mean so the run can be interrupted and resumed from checkpoint. QE is normalised internally so W_QE ≈ 1; GI is not, so W_GI encodes the partial mode recovery (W_GI < 1 at all L, rising from 0 at L = Lhp = 4000).

**QE bandpass.** The QE and RDN0 use `LowPass(Lmax)` — a hard step-function cutoff. An earlier version used a wide cosine taper (`Δℓ=2000`) to smoothly roll off C^TT at the bandpass edge. In practice the taper moved both the beam B̂ and the noise Ĉ_n toward zero simultaneously in the transition zone, making the total signal+noise covariance Σ_tot = B̂²C̃_f + Ĉ_n → 0. The QE weights u = Σ_tot⁻¹(B̂·d) diverged on a single Fourier mode at the edge, producing a bright sinusoidal stripe in the φ̂_QE map (see §QE outlier analysis below). The hard cutoff prevents this: with white noise (ℓ_knee=0), Ĉ_n is non-zero everywhere within the bandpass, so Σ_tot ≥ Ĉ_n > 0 at all reconstructed modes.

**RDN0.** Per-simulation realization-dependent N^(0) estimates are computed via `N0_bias`. In the plotting stage these are averaged across simulations before subtracting from the QE auto-spectrum, since individual RDN0 estimates are very noisy at UL noise (CMB-dominated).

**MAP parameters.** At S4 noise, MAP warm-starts from the QE Wiener-filter solution with αmax=0.3 (stable line-search step). At UL noise the posterior gradient is much larger so αmax=0.05 is needed; 500 CG steps per MAP iteration are required to reach tolerance; and a QE Wiener-filter warm start is used.

**CG preconditioner.** A monkey-patch to CMBLensing replaces the default unlensed-Cf preconditioner with the lensed Cf̃ = L(ϕ)CfL(ϕ)†. This does not change the MAP solution but reduces the CG iteration count, particularly at low noise where Cf̃ ≫ Cn.

**Adaptive Hessian (UL Hess run).** `MAP_joint` supports a diagonal quasi-Newton secant update of the step-size matrix: for each Fourier mode L, `H⁻¹_L ≈ |Δϕ°_L| / |Δ∇_L|`. This update is enabled after `nburnin_update_hessian` steps. A bug caused the update to silently fail on every step: `ℓ^4 = 0` at the DC mode produced NaN in the log-log smoothing, and the `all(isfinite)` guard then rejected every update. Fixed by `FuncCℓs(ℓ -> max(ℓ^4, 1.0))`. The UL Hess run uses `nburnin_update_hessian=5`.

**Prior deprojection (UL Hess run).** `prior_deprojection_factor=0.5` projects 50% of the prior gradient out of the ϕ update step. This helps the optimizer focus on the data likelihood at prior-dominated high-L modes, potentially improving reconstruction beyond the MAP Wiener-filter fixed point.

**MAP convergence diagnostics.** A separate file `diag_map_12000{suffix}.jld2` records per-sim histories of total logpdf, loglike (data term), logprior, CG iteration count, and CG residual. These are never mixed with the MAP WL/phi output files. A per-sim summary table (status flags: `ok (data↑ prior↓)`, `⚠ prior-dominated`, `⚠ likelihood↓`, `OUTLIER`) is written to `diag_map_12000{suffix}_table.txt`.

**Checkpointing.** Both the W_L running sums and the φ maps are written after every simulation. If the run is interrupted, it resumes from where it left off by checking which sim indices are already stored.

---

## 5. Figures — `plot_qe_gi_sigma_12k.jl`

This script loads the checkpoint files produced by `run_qe_gi_wl12k.jl` and reproduces the main figures of Hadzhiyska et al. (2019).

**Debiasing.** For QE, the auto-spectrum is debiased at the spectrum level:
```
C_auto,deb = [C(φ̂_raw) − ⟨N^(0)⟩] / W_QE²(ℓ)
```
where ⟨N^(0)⟩ is the mean RDN0 across simulations. The cross-spectrum is debiased at the field level: φ̂_deb = φ̂ / W_QE, then C_cross = C(φ_true, φ̂_deb). For MAP, only field-level debiasing is used. W_L is smoothed with a 9-bin running mean before applying.

**GI noise bias.** GI has a noise bias N0_GI analogous to QE N0: even with φ=0, the estimator produces a non-zero auto-spectrum from the unlensed CMB and noise in T_hp. Three N0 estimators are implemented and saved per sim: (1) **global MC N0** (`gi_n0_mc`): 200 null (φ=0) simulations run once; the mean auto-spectrum gives ⟨N0_GI⟩; this global constant is stamped onto all sims and subtracted from each sim's auto-spectrum. (2) **analytical N0** (`gi_n0_analytical`): per-realization, using the formula `N^{TT}(L) / denom(L)` from Hadzhiyska+2019 Eq. A1–A2. (3) **per-sim RDN0** (`gi_twoleg`, saved as `N0_gi_rdn0`): two-leg cross-sim estimator analogous to QE RDN0.

The **global MC N0** correctly removes the systematic offset — subtracting a constant c gives Var[C − c] = Var[C], so it cannot reduce σ. The **analytical N0** was found to worsen off-diagonal covariances and increase variance when applied per-sim: the formula N^(0)(L) = N^{TT}(L) / denom(L) uses the same gradient denominator denom(L) as the reconstruction, so N^(0) is correlated with the signal for any given sim. This spurious correlation inflates the off-diagonal covariance matrix and increases variance after subtraction — the opposite of the intended effect. It is no longer applied and is removed from the pipeline. The **per-sim GI RDN0** (`gi_twoleg`) tracks per-realization noise fluctuations and is the closest GI analogue to QE RDN0; it is saved but not used in the current plot.

**N0 variants in the figures.** The plot script tracks and displays the following variants to make the effect of debiasing visible:

| Plot key | Colour | Description |
|----------|--------|-------------|
| `qe_no_rdn0` | orange | QE auto-spectrum with **no N0 subtraction**, only W_QE² division. Shown in fig3 auto-σ row to make the RDN0 improvement visible. |
| `qe` | red | QE auto-spectrum with **mean RDN0 subtracted** before W_QE² division (production estimator). |
| `gi` | blue | GI auto-spectrum with **global MC N0 subtracted** before W_GI² division (label: "GI (global MC N0)"). |
| `gi_fgmc` | cyan | GI auto-spectrum with **per-sim fixed-gradient MC N0 subtracted** (see §2). This is the production GI debiasing method. |
| `mj` | purple | MAP joint, field-level debiasing only (no N0 subtraction). |

The fig3 σ panels show `qe_no_rdn0`, `qe`, `gi`, `gi_fgmc`, and `mj` together so the effect of each N0 treatment can be read off directly from the change in error bar height. The covariance correlation figure (fig_covariance_correlation) uses the fg-MC N0 in its per-sim N0 subtraction row, replacing the global MC N0 which dilutes rather than decorrelates the off-diagonal structure.

**σ rescaling.** The simulation covers a patch of sky fraction f_sky_sim ≈ 7.6 × 10⁻⁴, while Hadzhiyska+2019 use a survey with f_sky = 0.4. Sample variance σ ∝ 1/√(f_sky ΔL (2L+1)), so σ is multiplied by √(f_sky_sim / f_sky_paper) to rescale to the paper's survey area.

**Figures produced.**
- **fig2** — mean cross- and auto-power spectra for QE, GI, and MAP, compared to the true C^κκ_L.
- **fig3** — error bars σ[C^κ̂κ̂] and σ[C^κκ̂] vs L (solid = simulation, dashed = Knox formula).
- **fig4** — effective reconstruction noise N_L,eff = σ × √(ΔL(2L+1)f_sky/2) − C^κκ_L (Eq. 33 of Hadzhiyska+2019).
- **fig5** — improvement ratio σ_QE / σ_GI and σ_QE / σ_MAP for auto and cross spectra.
- **fig6** — correlation coefficient ρ_L = ⟨C(κ_true, κ̂) / √(C^κκ C^κ̂κ̂)⟩ with ±1σ shading.
- **fig7** — combined panel: (C^κκ̂ / σ)² and ρ_L.
- **fig_WL** — transfer functions W_L for QE, GI, and MAP (raw and smoothed).
- **figA** — quarter-consistency check: σ computed on four subsets of the simulation suite, verifying convergence.
- **figB** — σ as a function of number of simulations, showing when the Monte Carlo has converged.
- **figC** — mean log-posterior, log-likelihood, and log-prior vs MAP step for S4, UL (original), and UL (Hess+deproj) runs, with CG iteration history and convergence diagnostic.
- **snr_table.txt** — integrated SNR for each estimator over L = 4000–12000, compared to the paper values.

---

## QE Outlier Analysis

**Mechanism.** The QE filter computes u = Σ_tot⁻¹ (B̂·d) where Σ_tot = B̂²C̃_f + Ĉ_n. If both B̂ and Ĉ_n go to zero at the same Fourier mode, Σ_tot → 0 and u diverges. Because the lensing signal in φ̂_QE is formed by convolving u with a second leg (the gradient), a single diverging mode (lx₀, ly₀) maps to a pure sinusoid e^{i(lx₀ x + ly₀ y)} in real space — a bright stripe across the full κ map.

**Observed artifacts.** Running `diag_phi_outlier.jl` across 2000 QE sims identifies ~5 sims with correlation ρ(κ_true, κ_QE) significantly below the bulk distribution (median − 4σ threshold). The worst (sim 1916, ρ = −0.18) shows two intersecting diagonal stripes in κ_QE — the ±(lx₀, ly₀) mode pair. GI and MAP reconstructions of the same sim are unaffected: GI's denominator is formed from gradient covariances σ_ij = ⟨∂_i T ∂_j T⟩ (bulk pixel statistics, bounded away from zero), and MAP optimises a full posterior without inverting Σ_tot directly.

**Cause (old taper code).** The earlier bandpass used a cosine taper over Δℓ = 2000 modes near Lmax. The noise spectrum Ĉ_n was also cut at the same ℓ_max. In the transition zone both B̂ and Ĉ_n decreased together, making Σ_tot approach zero. One Fourier mode sitting at the edge of the taper could have Σ_tot exactly zero (to floating-point precision), causing u → ∞.

**Fix.** `run_qe_gi_wl12k.jl` now uses a hard cutoff `LowPass(Lmax)` with no taper. With white noise (ℓ_knee = 0), Ĉ_n is a non-zero constant for all L ≤ Lmax, so Σ_tot ≥ Ĉ_n > 0 at every reconstructed mode. The pre-fix outlier maps are already stored in the JLD2 checkpoint files and cannot be regenerated without re-running those specific seeds. The plot script removes them via a 15×median RMS filter on per-sim auto power before computing σ statistics.

**Diagnostic scripts.** `diag_phi_outlier.jl` ranks sims by ρ(κ_true, κ_QE) and saves side-by-side κ maps for the worst outliers. `diag_qe_blowup.jl` confirms the mechanism: it plots Σ_tot per Fourier mode and the peak u amplitude per sim, showing that outlier sims have Σ_tot → 0 at a single mode and |u|_max orders of magnitude above the bulk.

---

## File Structure

```
CMB_Project/
├── notebooks/
│   └── project.ipynb            initial exploration (single sim)
├── Functions/
│   ├── gradient_inversion.jl    GI estimator (three variants)
│   ├── gradient.jl              grad_fft utility
│   ├── spectra.jl               get_Cℓ, binning
│   ├── debias.jl                debias_phi_with_WL
│   ├── normalization.jl         W_L computation
│   └── error_mean.jl            per-gradient error analysis
├── run_qe_gi_wl12k.jl           production run (QE + GI + MAP, 1000 sims)
├── plot_qe_gi_sigma_12k.jl      figures reproducing Hadzhiyska+2019
├── run_error_analysis.jl        pixel-space error analysis (100 sims, Lmax=8000)
├── plot_mse_analysis.jl         RMS(Δφ) vs |∇T| figure
├── utils.jl                     module re-exporting all Functions/
└── results/
    ├── Boryana's paper/          output figures and SNR table
    ├── WL_qe_gi_12000.jld2      W_L checkpoint (S4)
    ├── WL_qe_gi_12000_ul.jld2   W_L checkpoint (UL)
    ├── phi_maps_qe_gi_12000*    φ maps (QE + GI sims)
    ├── phi_maps_map_12000*      φ maps (MAP sims)
    ├── WL_map_12000*.jld2       MAP W_L checkpoints
    └── diag_map_12000*.jld2     MAP per-sim diagnostics (loglike/logprior/CG — separate from WL files)
```

---

## MAP Diagnostic Tests — `diag_map_convergence.jl`, `diag_map_s4_warmstart_test.jld2`

MAP joint reconstruction is run via `MAP_joint(ds, FieldTuple(ϕ=ϕ_start); nsteps=40, αmax=αmax, ...)`.  At S4 noise the log-posterior plateaus within 40 steps; at UL (0.1 µK-arcmin) noise convergence is slower.

### Experiments run

The diagnostic suite (`results/diagnose/diag_map_s4_warmstart_test.jld2`) tested the following at S4 noise:

| Test | Values |
|------|--------|
| Warm start | zero / QE Wiener-filter / GI |
| Step size αmax | 0.01 / 0.05 / 0.1 / 0.3 |
| Hessian burnin | off / 5 / 10 / 20 steps |

Each variant runs 40 gradient-descent steps with CG tolerance 1e-3. ρ_L is tracked at L = {4000, 6000, 8000, 10000} every step.

### Key results

All variants converge to the same fixed point regardless of warm start, step size, or Hessian schedule:

| L | ρ_L (MAP joint) |
|---|----------------|
| 4000 | 0.51 |
| 6000 | 0.36 |
| 8000 | 0.21 |
| 10000 | 0.10 |

Starting from the GI solution (ρ_L ≈ 0.76 at L = 6000) the MAP *regresses* to the same fixed point — it is being pulled away from the GI solution by the prior, not converging toward a better one.

### Why MAP underperforms GI at high L

MAP_joint applies a Bayesian Wiener filter W_L = C^φφ(L) / (C^φφ(L) + N_eff(L)).  At high L where C^φφ → 0 and N_eff is large, W_L → 0 and the prior dominates the reconstruction.  GI avoids this by not using a prior — it is a direct maximum-likelihood inversion restricted to scales where the gradient approximation holds.  This is expected behaviour, not a bug: at S4 noise the signal-to-noise in φ at L ≳ 6000 is very low and the MAP prior is correctly down-weighting those modes.

### Adaptive Hessian fix and improved UL run

The `nburnin_update_hessian` quasi-Newton update in `MAP_joint` was silently failing on every step due to an ℓ=0 NaN bug: `FuncCℓs(ℓ -> ℓ^4)` evaluates to 0 at the DC mode, causing log-log smoothing to produce NaN, which then caused `all(isfinite)` to reject every update. Fixed in `CMBLensing.jl/src/maximization.jl` by replacing `ℓ^4` with `FuncCℓs(ℓ -> max(ℓ^4, 1.0))`. A new UL MAP run (`_ul_hess` suffix) uses `nburnin_update_hessian=5` (adaptive Hessian active from step 5) and `prior_deprojection_factor=0.5`. Output goes to separate `WL_map_12000_ul_hess.jld2`, `phi_maps_map_12000_ul_hess.jld2`, and `diag_map_12000_ul_hess.jld2` files, leaving the original UL MAP files untouched. The plot script auto-selects the Hess run for UL figures if the file exists, and falls back to the original.

### Prior weakening test — result

MAP was re-run at S4 noise with the fiducial C^φφ prior multiplied by ×5, ×20, and ×100 at all L. The transfer function W_L did respond: with a 100× inflated prior the MAP W_L at high L increased toward W_GI, confirming the optimizer finds a different fixed point when the prior penalty is weaker. However, ρ_L did not improve. The MAP correlation coefficient remained at the same values as the fiducial run regardless of prior strength. This decoupling — W_L rises but ρ_L stays flat — shows that the prior Wiener filter is not the fundamental limitation on MAP performance at high L; the likelihood itself does not contain enough signal-to-noise in φ at L ≳ 6000 for the S4 noise level to improve ρ_L over QE. The prior sets the amplitude of the reconstruction, but the SNR in φ at these scales is genuinely low. GI sidesteps this because it is not a Bayesian estimator: it directly inverts the gradient equation without a prior penalty, so it avoids the Wiener suppression, but this also means it does not optimally weight modes.

### Zero-prior MAP test (`_ul_zero_a01`)

MAP was re-run at UL noise with the prior completely removed (C_φφ → ∞, zero prior gradient) and a smaller step size α = 0.1. Without the prior, MAP reduces to a maximum-likelihood estimator and the Wiener shrinkage W_L → 1 is no longer forced. Results are stored in `WL_map_12000_ul_zero_a01.jld2` and `phi_maps_map_12000_ul_zero_a01.jld2`. This run is overlaid in orange on the sigma, ρ_L, and W_L figures.

The Hessian (quasi-Newton secant) update is unreliable in this setting: once the log-posterior gradient becomes very small (near convergence or with no prior), the secant approximation |Δφ°| / |Δ∇| is 0/0, causing NaN and silently failing the `all(isfinite)` guard. The optimizer falls back to the fiducial preconditioner. This does not affect the converged MAP answer (fiducial vs realisation-dependent preconditioner change only the convergence path, not the fixed point) but does mean the adaptive Hessian offers no benefit in this run. Known-bad sims excluded: `{121, 242, 362, 453, 484, 661, 1916}`.

The zero-prior result is interesting precisely because it isolates the amplitude question: if MAP converges correctly without the prior, W_L should approach 1. If it does not, the explanation is either finite-iteration convergence failure or that the likelihood curvature at high L is so flat that many more steps are needed.

### Phase 2: Realization-dependent preconditioner

The 4.9σ gradient-response correlation in MAP (r = 0.223) suggests the fiducial preconditioner does not fully exploit per-realization information. The GI Fisher weighting D(Lx,Ly) = Lx²σxx + 2LxLy σxy + Ly²σyy directly captures the per-mode signal-to-noise available in each realisation. Replacing the fiducial Nϕ in `Hessian_logpdf_preconditioner` with a realisation-dependent version:

```
N_ϕ^{realized}(L) = N_ϕ^{fid}(L) × D_fid(L) / D_D(Lx,Ly)
```

would tighten the preconditioner in modes where the gradient is stronger than average (D_D > D_fid → smaller effective noise → larger step) and widen it where it is weaker. Two variants in order of difficulty:

- **Scalar version**: R_scalar = (σ_D_xx + σ_D_yy) / (σ_fid_xx + σ_fid_yy). Modify `ds.Nϕ /= R_scalar` before `MAP_joint`. Changes only convergence speed.
- **Per-mode version**: field-valued `Nϕ(Lx,Ly) = N_ϕ^{fid}(L) × D_fid(L) / D_D(Lx,Ly)`. Requires passing a `FlatFourier` field as `ds.Nϕ`, touching CMBLensing internals.

If MAP is genuinely converged (confirmed by log-posterior plateaus), preconditioner changes will not move the final answer — the effect would only be visible with a fixed iteration budget where better conditioning reaches the fixed point faster. Alternatively, modifying the effective prior Cϕ per realisation (not just the preconditioner) would change the converged answer, but this is a different estimator altogether.

---

## Current Results — UL (0.1 µK-arcmin, Lmax=12000)

Production run: 1000 QE/GI sims, ~500 MAP sims (zero start, αmax=0.05, 80 steps, CG tol=1e-4, nsteps=500). All estimators debiased by ensemble-mean W_L.

| Dataset | Auto-QE | Auto-GI | Auto-MAP | Cross-QE | Cross-GI | Cross-MAP |
|---------|---------|---------|---------|---------|---------|---------|
| UL (0.1 µK-arcmin) | 339.4 | 303.6 | 657.5 | 1631.7 | 3031.7 | 1718.5 |
| Paper UL (Hadzhiyska+2019) | 205 | 1515 | — | 710 | 4100 | — |

MAP dominates on auto-SNR (657 vs 339 QE, 303 GI). GI dominates on cross-SNR. MAP beats QE on cross-SNR (1718 vs 1631). The MAP-GI cross-SNR gap is explained below.

---

## MAP vs GI: Theoretical Analysis

### Does MAP use the realised gradient?

Yes — through the likelihood. In each MAP iteration the CG inner loop solves for the Wiener-filtered CMB `f_wf` given the current `ϕ`. That `f_wf` gradient drives the ϕ update via `∂ log p/∂ϕ`. MAP conditions on the realised lensed field through the full nonlinear likelihood. It is not missing this information.

### Why GI wins on cross-SNR despite MAP being Bayesian-optimal

GI is not a Bayesian estimator. It is a matched filter that **fixes** ∇T_obs and inverts the gradient equation directly:

```
T_hp(x) ≈ ∇T_obs(x) · ∇ϕ(x) + noise   →   ϕ̂_GI(L) = [numerator] / F_obs(L)
```

By conditioning on the specific gradient pattern ∇T_obs in that realisation, GI exploits non-Gaussian structure that falls outside the Gaussian likelihood model. MAP marginalises over `f` jointly with `ϕ` under a Gaussian prior, which treats all CMB realisations with the same power spectrum as equally likely. The prior Cϕ then Wiener-filters the result: W_L = Cϕ/(Cϕ + N0) → 0 at high L. Ensemble-mean debiasing by 1/W_L corrects the mean but does not recover the information lost in the Gaussian marginalisation over f. This is correct Bayesian behaviour, not a bug.

**Gradient-conditioned MAP.** If you fix ∇T_obs and maximise over ϕ alone, the problem becomes linear and has a closed-form solution:

```
ϕ̂_GI-MAP(L) = Cϕ(L) / [Cϕ(L) + 1/F_obs(L)]  ×  ϕ̂_GI_unnorm(L)
```

This IS Wiener-filtered GI (Bayesian GI). It conditions on the realised gradient and applies the prior correctly. It is a different estimator from `MAP_joint`, which marginalises over f rather than fixing it. Implementing "gradient-conditioned MAP" means computing GI with a Bayesian Wiener filter per-sim — not modifying `MAP_joint`.

### What MAP's advantage is actually used for

- **Auto-SNR**: MAP 657 >> QE 339, GI 303. MAP correctly handles the noise model and prior — GI's high cross-SNR comes at the cost of amplifying noise in auto-power.
- **Delensing**: for B-mode delensing, the optimal quantity is the posterior mean E[ϕ|d] (Wiener-filtered MAP), not the debiased estimate. MAP is correct for this purpose.
- **Parameter inference**: MAP is the correct Bayesian posterior for cosmological parameter extraction. GI is not calibrated for this.
- **No N0 bias**: MAP needs no external noise debiasing (no N^(0) subtraction), unlike QE and GI.

The GI cross-SNR advantage is a property of GI being non-Bayesian and gradient-conditioned — not a MAP failure.

### Realisation-dependent debiasing

Replacing the ensemble-mean W_L with a per-sim estimate W_L_i does NOT reduce response suppression — it rescales the output after the fact. The MAP fixed point (and Wiener suppression) is determined by the estimator itself, not the debiasing step. For GI, RDN0 achieves something analogous by tracking per-sim noise fluctuations (see §2), but this reduces variance rather than increasing the mean cross-correlation.

---

## MAP Experiments — Attempts to Improve High-L Cross-SNR

### 1. Realised-Nϕ preconditioner (`diag_realised_map.jl`)

Scales `ds.Nϕ → Nϕ / r(L)` where `r(Lx,Ly) = F_real(L)/F_fid(L)` is the 2D ratio of realised to fiducial gradient Fisher. Nϕ is the MAP preconditioner approximating the Hessian `Cϕ⁻¹ + N0⁻¹`.

**Result: ΔW ≈ 0.** The MAP fixed point is unchanged. This confirms that Nϕ controls only convergence speed — changing the preconditioner cannot move the converged estimate. Reported: mean Δlogpdf(last 5 steps) = 62.7 for the `_ul_pw` (prior-weakened) run, confirming that run was not converged; the `_ul_zero_a01` run converged fully.

### 2. Prior-weakening run (`_ul_pw`, `diag_map_12000_ul_pw.jld2`)

Uniform scalar weakening of Cϕ at all L (map_prior_weakening > 1). W_L rises toward 1 as expected. ρ_L does not improve. Sim 29 diverged (CG residual = 2×10⁸²). This confirms that the Wiener suppression is a consequence of the prior, but weakening the prior uniformly does not help ρ_L because the likelihood SNR at high L is genuinely low.

### 3. Realised-Cϕ test (`diag_cphi_realised.jl`)

Scales `ds.Cϕ → Cϕ × r(Lx,Ly)` mode-by-mode for L > L_SPLIT=3000, where the 2D anisotropic scaling factor `r(Lx,Ly) = F_real(Lx,Ly) / F_fid(Lx,Ly)` is the ratio of the realised gradient Fisher `F_real = Lx²σ_xx_r + 2LxLy σ_xy_r + Ly²σ_yy_r` to its fiducial expectation, with σ_r computed from the low-pass (ℓ<2000) data gradient. MAP B runs warm from the MAP A solution with αmax=0.1 (smaller step to stay near the modified fixed point). GI is also run on each sim for comparison.

**Result: ρ_A = ρ_B to 3 decimal places across all L bins.** MAP B produces no improvement over MAP A:

| bin | ρ_GI | ρ_A | ρ_B | frac |
|-----|------|-----|-----|------|
| 4k–6k | 0.310 | 0.442 | 0.442 | ≈0 |
| 6k–8k | 0.247 | 0.274 | 0.274 | ≈0 |
| 8k–10k | 0.129 | 0.144 | 0.144 | ≈0 |
| 10k–12k | 0.056 | 0.060 | 0.060 | ≈0 |

**Why it fails.** The scalar summary `r_iso = (σ_xx_r + σ_yy_r)/(σ_xx_f + σ_yy_f) ≈ 0.97–1.00` in every sim. The gradient covariance σ_r is computed from the data at ℓ<2000, where ~2000² Fourier modes contribute; cosmic variance over this broad range is tiny (~1%), so every sim gives r ≈ 1 regardless of the specific high-L φ realisation. The 2D anisotropy (σ_xy ≈ −8×10⁵ vs σ_xx, σ_yy ≈ 9×10⁸) is also negligible. The Cϕ × r prior is therefore indistinguishable from the fiducial Cϕ, and MAP B converges to the same fixed point as MAP A. More fundamentally, low-ℓ CMB gradient power has no significant correlation with the high-L φ amplitude that MAP is trying to recover — the two scales are nearly decoupled.

**Recommended next step: anisotropic per-mode Cϕ from the GI Fisher, not the CMB gradient.** The r_iso approach fails because the proxy (low-ℓ gradient variance) doesn't correlate with what matters (per-mode φ SNR at L>3000). The correct per-mode SNR at each (Lx,Ly) is `F_obs(Lx,Ly) = Lx²σ_xx_obs + 2LxLy σ_xy_obs + Ly²σ_yy_obs` where σ_obs is estimated from the **high-pass** temperature gradient at L ≳ Lhp=4000, i.e., the same gradient that GI uses. This quantity does vary meaningfully between modes: it is large where L is aligned with the dominant gradient direction and near zero where L is perpendicular. Scaling `Cϕ(Lx,Ly) = Cϕ(|L|) × F_obs(Lx,Ly) / F_fid(|L|)` would relax the prior precisely in the high-L modes where the gradient is locally informative. However, this is equivalent to Wiener-filtered GI: the MAP fixed point with this prior is `ϕ̂ = [Cϕ(|L|) + 1/F_obs(Lx,Ly)]⁻¹ Cϕ(|L|) × ϕ̂_GI(Lx,Ly)`, and the resulting transfer function is `W(Lx,Ly) = Cϕ(|L|) / (Cϕ(|L|) + 1/F_obs(Lx,Ly))`. This is strictly larger than the isotropic MAP W_L at high L, and reduces to standard GI when `Cϕ → ∞`. Implementing it inside `MAP_joint` is non-trivial (requires a full 2D `FlatFourier` field for `ds.Cϕ` rather than an isotropic `FuncCℓs`), but computing it analytically from an existing GI run and applying it as a post-processing Wiener filter costs nothing.

---

## QE Outlier Sims — `diag_phi_outlier.jl`

A small number of simulations exhibit catastrophically negative pixel-space correlation between the QE reconstruction and φ_true. Five sims are excluded from all analyses at both noise levels:

| Sim | Pixel-space corr(φ̂_QE, φ_true) |
|-----|----------------------------------|
| 1916 | −0.176 |
| 121  | −0.123 |
| 661  | −0.012 |
| 967  | −0.011 |
| 1692 | −0.007 |

**Cause.** The QE involves T(ℓ₁) T(ℓ₂) / (C^{TT}_{ℓ₁} C^{TT}_{ℓ₂}) summed over all ℓ pairs. When a single Fourier mode T(ℓ₀) is an outlier (|T(ℓ₀)| ≫ √C^{TT}_{ℓ₀}, a valid but rare Gaussian fluctuation), that mode contributes to the QE output at all L values:

```
φ̂_QE(L) ≈ T(ℓ₀) T(L − ℓ₀) f(L, ℓ₀) / (C^{TT}_{ℓ₀} C^{TT}_{L−ℓ₀}) + ...
```

This injects large power at all L, producing a stripe artifact in the φ map associated with wavenumber ℓ₀. The auto-spectrum of φ̂_QE for that sim is dominated by this artifact, corrupting the per-sim bandpower estimate. The GI estimator is not affected: it inverts the gradient equation without dividing by C^{TT} in Fourier space, so a single large T mode does not produce a pole.

**Rate.** With ~130 000 independent Fourier modes per sim and 2000 sims, a handful of sims with ≳ 5σ mode outliers is expected. The 5 excluded sims are identified by `diag_phi_outlier.jl`, which scans the pixel-space correlation corr(φ̂_QE, φ_true) across all sims and flags those more than 4σ below the median.

**Exclusion.** Sims are excluded via the `exclude_sims` parameter of `process_noise_level` in `plot_qe_gi_sigma_12k.jl`. The set is defined once as `GI_OUTLIERS` and passed to both noise-level calls.

---

## Running the Code

All scripts activate the project environment via `Pkg.activate(@__DIR__)` and should be run from the project root. The `.jld2` checkpoint files are large and are not tracked by git (see `.gitignore`); they must be regenerated locally.

To reproduce the full analysis from scratch:

1. **Production run** (takes several hours on CPU):
   ```
   julia run_qe_gi_wl12k.jl
   ```
   This generates all `WL_*.jld2` and `phi_maps_*.jld2` files.

2. **Figures**:
   ```
   julia plot_qe_gi_sigma_12k.jl
   ```
   Outputs to `results/Boryana's paper/`.

3. **Pixel-space error analysis** (optional, Lmax=8000 only):
   ```
   julia run_error_analysis.jl
   julia plot_mse_analysis.jl
   ```

---

## HPC Job Log

**diag_map_zero_gihess (29490228)** — killed at time limit after 3 hours. Got through sims 1–2 fully, sim 3 baseline still running at kill.

| Sim | Baseline W | GI-Hess W | ΔW | GI-Hess faster? |
|-----|-----------|-----------|-----|-----------------|
| 1 | 0.6048 (3212s) | 0.6367 (1858s) | +0.032 | yes, ~42% faster |
| 2 | 0.6007 (3664s) | 0.6494 (1692s) | +0.049 | yes, ~54% faster |

GI-Hessian is consistently better W and ~2× faster convergence. Checkpoints saved for sims 1–2 — resubmit and it picks up from sim 3.

**map_settings_s4 (29494547)** — killed at time limit after ~3 hours. Sim 1: MAP-F failed (NaN/CG assertion) but caught and continued. Sim 2: only got through QE+GI+MAP-A before timeout. Checkpoint at `test_map_settings_s4_convergence_checkpoint.jld2` (184 KB, updated 00:25) — resubmit resumes from sim 2 MAP-H onward.

---

## Dependencies

Julia packages: CMBLensing (local fork), JLD2, PythonPlot, LinearAlgebra, Statistics, Printf.
Python (via PythonCall): matplotlib.
