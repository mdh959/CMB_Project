# Paper Discussions

Notes on key papers relevant to this CMB lensing reconstruction pipeline. For each paper: what the core contribution is, how it connects to what we have implemented, what we have not done but could, and what limitations to be aware of.

---

## 1. Carron & Lewis (2017) — *Maximum likelihood CMB lensing reconstruction*

**Reference:** Carron, J. & Lewis, A. (2017). *Maximum likelihood CMB lensing reconstruction.* Physical Review D, 96(6), 063510. arXiv:1704.08373.

### Core contribution

This paper derives the MAP (maximum a posteriori) lensing estimator from first principles and proves it is optimal for low-noise CMB data. The key insight is that the standard quadratic estimator (QE) is a single Newton–Raphson step starting from zero, and the MAP estimator is the converged fixed point of the same iteration.

The posterior for the lensing field φ given observed temperature T is:

    ln P(φ | T) = ln L(T | φ) − (1/2) Σ_L φ_L C_L^{φφ,fid −1} φ_L*

where the likelihood L is Gaussian in the unlensed CMB. The gradient of this posterior (the "wiener-filtered quadratic gradient") is:

    g_L = −i L · FFT[ T̄ · ∇T_WF ]

with T̄ = C^{−1}T the inverse-variance filtered data, and T_WF = C^{φφ} (C^{φφ} + N)^{−1} T the Wiener-filtered lensed CMB. This is the "unlensed weights" QE when the lensing operation is linearised.

MAP convergence is guaranteed in the limit where the lensing potential is small enough that LenseFlow does not create caustics (κ < 1 everywhere). For CMB-S4 noise levels (1 µK-arcmin), this is satisfied in the diffuse field. For near-perfect SNR (0.1 µK-arcmin ultra-low noise), individual realisations of QE reconstruction can have κ ≈ 1 near rare overdense pixels, causing LenseFlow to diverge at the first MAP evaluation.

### The W_L transfer function

The MAP estimate is biassed low by a factor W_L < 1, analogous to the QE response. This is estimated empirically:

    W_L = mean_sims[ C_L(φ_true, φ̂) / C_L(φ_true, φ_true) ]

**This is exactly what the run script computes.** The per-sim ratio is accumulated over all sims and averaged. Debiasing is then φ̂_debias = φ̂ / W_L (implemented in `debias_phi_with_WL`).

### What Carron & Lewis show about QE vs MAP

The key Figure 4 in C&L (2017) shows the cross-correlation coefficient ρ_L = C_L^{φ_true, φ̂} / sqrt(C_L^{φφ} C_L^{φ̂φ̂}) as a function of L. For CMB-S4-like noise:

- QE: ρ_L ≈ 0.5–0.7 at the reconstruction peak L ~ 200–500
- MAP: ρ_L → 1 at all L ≥ 30

This is the cleanest statement of MAP optimality: it is not just that the auto-power is lower, but that the MAP estimate is more correlated with the true lensing field on a per-mode basis. This is **ρ_L in Fig. 6 of our plot script**, directly equivalent.

### Connection to GI

C&L do not discuss the GI (gradient inversion) estimator. GI uses the same gradient structure as MAP but: (a) does not iterate (single step from zero), (b) uses empirical rather than Wiener-filtered weights, and (c) applies a hard high-pass on T rather than inverse-variance filtering. GI therefore does not converge to MAP for diffuse field reconstruction. See Horowitz et al. discussion below.

### What we have implemented

- MAP joint via `MAP_joint` in CMBLensing.jl: warm start from WF QE, αmax capping Brent line search, 30 Newton–Raphson steps.
- W_L transfer function estimated from sims and applied to debias MAP φ̂.
- ρ_L plotted in Fig. 6 of the plot script.
- N_L^{φφ} = C_L^{φφ} (1/ρ_L² − 1) in Fig. 7.

### What we have NOT done (and whether it matters)

1. **Iterative N0 (RD-N0) debiasing.** C&L Sec. IV introduce a realization-dependent N0 subtraction for the MAP power spectrum that is robust to misspecified noise. We compute the raw C_L(φ̂, φ̂) and C_L(φ_true, φ̂) directly from sims — since we have access to φ_true, we do not need N0 for the cross-spectrum. For the auto-spectrum, our σ is scatter-across-sims rather than a debiased power, which is valid for comparing estimators but not for a cosmological measurement from real data.

2. **Mean-field subtraction.** For a masked or anisotropic observation, MAP has a non-trivial mean field ⟨φ̂^MAP⟩ ≠ 0 that must be subtracted before computing the power spectrum. Our simulations use full-sky, isotropic noise with no mask, so the mean field is zero by symmetry. This is fine for the current pipeline.

3. **Curved-sky reconstruction.** CMBLensing.jl uses flat-sky. For a survey fraction larger than a few percent, curved-sky effects enter at the ~1% level. Not relevant for our 4° × 4° patch sims.

---

## 2. Millea et al. (2020) — *Optimal CMB lensing reconstruction and parameter estimation*

**Reference:** Millea, M., Anderes, E., & Wandelt, B. D. (2020). *Optimal CMB lensing reconstruction and parameter estimation with NeuralCMBLens.* Physical Review D, 102(12), 123542. arXiv:2002.12186.

### Core contribution

Millea et al. implement a full Bayesian joint sampling approach: rather than finding a single MAP point, they sample the joint posterior P(φ, f | d) where f is the unlensed CMB field. This is done with a Hamiltonian Monte Carlo (HMC) sampler. The key claim is that sampling is strictly more powerful than MAP: MAP finds a mode of the posterior and reads off a point estimate, while sampling integrates over all posterior mass including the posterior width, recovering the optimal estimator in the MMSE (minimum mean-square error) sense.

The posterior is:

    ln P(φ, f | d) = −(1/2)||d − B D_φ f||^2_N − (1/2) f^T C_f^{−1} f − (1/2) φ^T C_φ^{−1} φ

where D_φ is the lensing remapping operator (LenseFlow).

### MAP vs sampling: when does it matter?

For a Gaussian posterior (which is a good approximation when the lensing SNR per mode is low), MAP ≈ MMSE and the two approaches give the same result. This is the case for the diffuse lensing power spectrum at L > 200 with CMB-S4 noise. The difference between MAP and sampling appears primarily at:

- Very large scales L < 50 where C_φ is large and the posterior is broad.
- Ultra-low noise (σ_T ≪ 1 µK-arcmin) where the posterior is highly non-Gaussian due to lensing being strong.
- Cluster lensing where κ ~ O(1) and the posterior is strongly non-Gaussian.

For our S4-like simulation (1 µK-arcmin, patch), MAP is an excellent approximation to optimal. For the ultra-low noise case (0.1 µK-arcmin), MAP may be slightly sub-optimal relative to full sampling, but the improvement would be small.

### MAP_joint in CMBLensing.jl

CMBLensing.jl implements both MAP and sampling. The `MAP_joint` function we use finds the joint MAP of (φ, f) via Newton–Raphson / L-BFGS. This is more powerful than the marginal MAP (optimising only over φ with f analytically marginalised, i.e., C&L 2017), because it simultaneously delenses the CMB while reconstructing φ. At convergence, MAP_joint should match the marginal MAP up to the approximation of the Newton step discretisation.

### Warm start and αmax

The `αmax` parameter caps the Brent line-search step size at each Newton iteration. Millea et al. do not discuss αmax directly, but the underlying issue is that LenseFlow (the remapping operator) can diverge if κ > 1 at any pixel. Large Newton steps can push κ to unphysical values. The fix:

- For S4-like noise: `αmax=0.3` (default) is safe; QE warm start gives a reasonable starting point.
- For ultra-low noise: `αmax=0.05` + zero warm start. The QE warm start at 0.1 µK-arcmin has near-unit Wiener filter, so φ̂_WF ≈ φ_true, but the QE noise component (which is small but non-zero) can push κ above 1 at rare pixels. Zero start avoids this entirely.

### Number of steps

Millea et al. (Fig. 14) show that the MAP logpdf converges in ≈ 20–30 Newton–Raphson steps for S4-like noise. For the ultra-low noise regime, convergence is slower because the posterior is more curved. The logpdf trace (6278k → 6498k, ~3.5% gain over 30 steps from zero start) suggests convergence is healthy but not yet saturated. Monitor: if the per-step gain at step 28–30 is still O(1%), increase MAPJ_STEPS to 50.

### What we have implemented

- `MAP_joint` with joint optimisation over (φ, f).
- Warm start from Wiener-filtered QE for S4; zero start for UL.
- αmax control via function parameter.
- W_L estimated from sims and used for debiasing.

### What we have NOT done

1. **HMC posterior sampling.** The full benefit of CMBLensing.jl is realised with `sample_joint`, not `MAP_joint`. For a paper-quality comparison of MAP vs optimal, one would run HMC chains. This is ≈ 100× more expensive per sim.

2. **Convergence diagnostics.** We do not currently plot the logpdf trace or the per-step gradient norm. For a paper, one should verify convergence (e.g., relative logpdf change < 0.01% over the last 5 steps).

3. **N1 bias for MAP power spectrum.** The MAP auto-power has a connected N1 bias analogous to the QE. Millea et al. estimate this via paired simulations. We do not subtract it; instead we use the cross-power C_L(φ_true, φ̂) which is N1-free by construction (no disconnected contractions when one leg is the true field).

---

## 3. Horowitz et al. (2019) — *Gradient Inversion for CMB lensing*

**Reference:** Horowitz, B., Ferraro, S., & Sherwin, B. D. (2019). *Reconstructing small scale lensing with the CMB: the gradient inversion estimator.* arXiv:1904.09575.

### Core contribution

Horowitz et al. propose the gradient inversion (GI) estimator as a fast, near-optimal lensing estimator in the **cluster lensing regime** (small patches, high convergence, L ≫ ℓ_Silk). The key physical observation is:

For a lensed CMB temperature field in the presence of a compact lens (cluster),

    ∇T_obs(x) = ∇T_unl(x − α(x)) + [deflection terms]

where for a strong lens, the dominant effect is that T_obs is a remapped version of T_unl. In Fourier space, the displacement field α(L) can be recovered from the cross-correlation of the gradient of T with the high-pass filtered T:

    φ̂(L) = −i L · FFT[ gx · T_hp + gy · T_hp ](L) / ⟨(L · g)²⟩

where g = ∇T_{ℓ ≤ L_grad} is the low-pass gradient (Lgrad = 2000), T_hp = T_{ℓ > L_hp} is the high-pass map (Lhp = 4000), and the denominator normalises by the mean squared projected gradient.

### When does GI converge to the optimal (ML) estimator?

Horowitz et al. show (Eqs. 14–16) that in the limit:
- L ≫ ℓ_Silk (lensing modes much larger than CMB coherence scale, i.e., cluster rather than diffuse regime)
- Noise → 0 (near-perfect SNR)

the GI estimator converges to the maximum likelihood estimator **per mode**. The key conditions are:

1. The CMB gradient is coherent over the patch — satisfied when the lens is much larger than the CMB correlation length (~10 arcmin).
2. The lens is compact — the approximation that α is constant over the CMB gradient coherence length.
3. Low noise so that the gradient is measured accurately.

**These conditions are NOT met for our diffuse lensing power spectrum reconstruction** at L ~ 5000–11000 with a 4° × 4° patch. There, lensing modes (L ~ few thousand) are comparable to or smaller than the CMB correlation length. GI is not optimal in this regime.

### Why MAP outperforms GI for diffuse power in our pipeline

The ordering (MAP better in auto-σ than GI, and comparably in cross-σ) is consistent with C&L (2017): MAP maximises the posterior and achieves ρ_L closer to 1. GI's single-step, empirical-normalisation approach does not reach the MAP fixed point. The fact that MAP has better auto-σ but comparable cross-σ to GI is explained by two effects:

1. **Physical (real):** MAP has lower per-mode reconstruction noise, so its auto-power C_L(φ̂_MAP, φ̂_MAP) = C_L^{φφ} + N_L^{MAP} has smaller N_L than GI. But MAP's W_L (transfer function) is estimated from fewer sims (~100 MAP vs ~1000 QE/GI), so its debiased auto-power has larger sampling noise in W_L², amplifying the auto-σ. The cross-σ (C_L(φ_true, φ̂)) is not affected by W_L, giving a cleaner comparison. When the number of MAP sims matches GI sims, MAP auto-σ should also be lower.

2. **Practical (sim-count artefact):** W_L² amplification of the uncertainty in W_L from finite MAP sims worsens the auto-σ estimate relative to cross-σ. This is not a physical effect.

### Connection to cluster lensing

For the cluster extension (AbacusSummit patches, M > 10^{14.5} M_sun), the GI conditions are better satisfied:
- Cluster convergence κ ~ 0.1–0.5, deflection α ~ arcmin scales
- L_cluster ~ few hundred (cluster mass scale), much larger than ℓ_Silk ~ 1500
- Near-perfect SNR within the cluster aperture if CMB-S4 noise

In this regime GI should perform comparably to MAP. Testing this comparison is the primary novel contribution enabled by the cluster extension.

### What we have implemented

- `gi_estimate_boryana` (empirical normalisation, σ_xx/σ_yy from per-sim gradient covariance)
- `gi_estimate_boryana_th` (theoretical normalisation from P_obs)
- Both correctly implement Eqs. 14–16 of Horowitz et al.
- Lgrad = 2000, Lhp = 4000 matching the Horowitz et al. fiducial.

### What we have NOT done

1. **The cluster stacking analysis.** Horowitz et al. Fig. 3–5 show GI applied to a stack of clusters, recovering the mean convergence profile κ(θ). We have the AbacusSummit cluster cutouts but have not yet applied GI/QE/MAP to them.

2. **N0 subtraction for cluster stacking.** When stacking reconstructed φ at cluster positions, the mean CMB reconstruction from random (low-mass) patches must be subtracted to remove the noise bias. This is not in the current pipeline.

---

## 4. Legrand & Carron (2023) — *Robust and efficient CMB lensing power spectrum from polarization surveys*

**Reference:** Legrand, L. & Carron, J. (2023). *Robust and efficient CMB lensing power spectrum from polarization surveys.* arXiv:2304.02584.

### Core contribution

This paper extends C&L (2017) to realistic (masked, anisotropic noise) surveys, using **polarization only** (E and B modes, no temperature). The key result is that the MAP lensing spectrum estimator — using realization-dependent N0 debiasing (RD-N0) — is robust and essentially optimal even when:

- The survey has a sky mask (Planck lensing mask, f_sky = 0.67).
- The noise is highly anisotropic (FFP10 Planck noise map) or has a large atmospheric component.
- The input lensing power spectrum or amplitude differs from the fiducial used in reconstruction by ~10%.

The method works on polarization because at CMB-S4 noise levels, the polarization reconstruction dominates over temperature (B/N ratios are higher for E modes at small scales). They use E and B modes with ℓ ∈ [2, 3000] for reconstruction and L ∈ [2, 4000] for the lensing field.

### RD-N0 debiasing

The realization-dependent N0 estimator (Eq. 5.5) is the key robustness tool. For a data map d with unknown noise, one generates simulations s_i lensed by the MAP estimate φ̂^MAP. The N0 is estimated by cross-correlating QE-like gradients with one leg on d and one on s_i:

    RD-N0_L = (1/R^MAP_L)² ⟨4Ĉ^{d s_i}_L − 2Ĉ^{s_i s_j}_L⟩_{i≠j}

This combination is first-order insensitive to misspecification of the noise model. **This is more powerful than our current approach** which computes C_L(φ_true, φ̂) directly — valid only in simulation where φ_true is known. On real data, RD-N0 is required.

### Mean-field treatment

In the presence of masking, the MAP estimate has a mean field ⟨φ̂^MAP⟩ that must be subtracted. Legrand & Carron find that using the QE mean field (estimated from 320 simulations) as the template during MAP iterations is sufficient — the residual mean field after convergence is small and can be subtracted using 40 simulations. The perturbative mean-field correction used in C&L (2017) is inadequate near mask boundaries.

**For our pipeline:** Full-sky simulations with isotropic noise → zero mean field by symmetry. No action needed for the current work.

### Normalization correction

Two separate normalisation corrections are needed for the masked MAP:
1. **Wiener-filter correction:** R^true_L / R^fid_L, estimated from pairs of sims with same φ_in but different CMB realisations (Eq. 4.4). This corrects the response of φ̂^MAP to the signal.
2. **Bias response correction:** (R^fid_L / R^true_L)² for the N0/N1 bias terms (Eq. 5.6).

These are O(1–3%) corrections for a CMB-S4-like experiment. The QE has a similar but smaller (~0.2%) correction on the full sky.

**For our pipeline:** We estimate W_L empirically from sims, which absorbs both the Wiener-filter and any signal response correction. This is correct but uses a different normalization convention to Legrand & Carron.

### N1 bias

The N1 bias for the MAP is analogous to the QE N1, estimated from pairs of sims with same lensing but different CMB (Eq. 5.3). Unlike the QE N1, the MAP N1 depends on the noise level because the achievable delensing affects the reconstructed B-mode spectrum. In our pipeline, we avoid N1 by using the cross-spectrum C_L(φ_true, φ̂) rather than the auto-spectrum.

### Robustness tests

The paper's main results (Table I, Fig. 8) show that with RD-N0 debiasing:
- Fiducial cosmology: unbiased recovery, χ² ≈ 0.96.
- 10% lower lensing amplitude (A_lens = 0.9): unbiased with RD-N0 (χ² ≈ 1.04), biased without (χ² ≈ 4.66).
- Large atmospheric noise: nearly unbiased with RD-N0 (χ² ≈ 1.32), catastrophically biased without.
- Anisotropic Planck noise: completely unbiased with RD-N0 (χ² ≈ 0.93), 48σ biased without.

The conclusion is that RD-N0 is essential for robust results with realistic noise.

### What we have implemented

- MAP joint reconstruction matching the framework, though using temperature (not polarization).
- W_L transfer function from sims (equivalent to R^true_L correction for full-sky isotropic noise).
- RD-N0 computation for QE (`run_rdn0=true` in the run script), generating simulations to estimate the QE noise bias. This follows the same logic as the Legrand & Carron QE RD-N0.

### What we have NOT done

1. **Polarization-only reconstruction.** This paper and C&L (2017) argue polarization dominates for CMB-S4 at small scales. Our pipeline is temperature-only. Adding EB polarization would be the single largest improvement available from the CMBLensing.jl framework.

2. **MAP RD-N0.** We compute QE RD-N0 but not MAP RD-N0. For the auto-spectrum of MAP φ̂, an RD-N0 debiaser would allow an unbiased power spectrum estimate. Currently we can only measure the cross-spectrum C_L(φ_true, φ̂) cleanly.

3. **Masking.** Real data requires masking of point sources, galactic emission, etc. The mean-field and normalization corrections described here would be needed.

---

## 5. Darwish et al. (2024) — *Non-Gaussian deflections in iterative optimal CMB lensing reconstruction*

**Reference:** Darwish, O., Belkner, S., Legrand, L., Carron, J. & Fabbian, G. (2024). *Non-Gaussian deflections in iterative optimal CMB lensing reconstruction.* arXiv:2407.00228.

### Core contribution

This paper addresses the N^(3/2)_L bias — the contamination of the lensing auto-spectrum due to the non-Gaussianity of the lensing field (from non-linear LSS clustering and post-Born lensing). For the QE at CMB-S4, this bias is ~1–10% of C_L^{φφ} at L ~ 500–3000, causing 1–2σ shifts on neutrino mass and large impacts on CMB-lensing × LSS cross-correlations if unaccounted for.

**The key result for our work:** the MAP estimator naturally suppresses this bias compared to the QE, without requiring any explicit modelling. The MAP N^(3/2)_L is smaller by a factor of order (1 − W_L)^2 relative to the QE, because the MAP partially delenses the CMB at each iteration, and the residual lensing field driving the bias is proportional to (1 − W_L) (the unresolved lensing after debiasing).

### Physical interpretation

The N^(3/2)_L bias arises from the lensing bispectrum B^{φφφ}(L, l_1, l_2) coupling the reconstruction mode L to two CMB modes. For the QE, this bispectrum enters at full weight. For the MAP, the lensing field reconstructed on the CMB legs is already subtracted at each Newton–Raphson step, so the residual enters with a suppression factor (1 − W_l_1)(1 − W_l_2). Since W_L → 1 at low noise, the MAP N^(3/2)_L → 0 precisely in the regime where it is most important.

The analytic prediction (using partially-lensed CMB spectra as the effective QE spectra for the converged MAP) matches simulations well at L < 2000.

### Limitations of MAP in temperature

A notable warning for temperature-only MAP reconstruction (our pipeline):

> "The gradient of the likelihood also introduces a mean field term which, in the absence of other sources of anisotropies, represents the anisotropy introduced [by] delensing the noise map by the estimated α. This term predominantly produces dilations (convergence-like) rather than local anisotropies (shear-like) terms, hence generally very small for polarization-only estimators. However, it is larger for temperature estimators, **possibly contributing up to 10–20% to the cross-spectrum with the input field**."

This is the delensing-induced mean field: even with isotropic noise and no mask, iteratively delensing the noise by the current φ estimate creates a convergence-like anisotropy in the noise. For polarization, the effect is negligible. For temperature, it can bias the cross-spectrum by 10–20%. This is an intrinsic limitation of temperature MAP that does not appear in polarization MAP.

**For our pipeline:** We use full-sky isotropic noise with no mask, so the mean field should cancel in the mean over sims (it is realization-dependent and averages to zero when both a W_L estimation sim and a test sim draw the same noise). However, on a per-sim basis this contamination would be present. For the power spectrum averaged over many sims, the effect is:
- Cross-spectrum C_L(φ_true, φ̂_MAP): the mean field does not contribute to the mean because ⟨φ_true, g_MF⟩ = 0 (φ_true is independent of the noise realisation). Not biased.
- Auto-spectrum C_L(φ̂_MAP, φ̂_MAP): the mean field contributes squared, i.e., it adds positive power. This could inflate the MAP auto-power slightly.

In practice, for our 4° × 4° patch with Gaussian isotropic noise, this effect is subdominant compared to the statistical scatter from the finite number of sims.

### Non-Gaussian prior: marginal benefit

The paper tests using a log-normal prior for κ instead of a Gaussian prior. The log-normal MAP suppresses the N^(3/2)_L bias only slightly further (~0.1–0.2% additional reduction at most scales). The Gaussian-prior MAP already captures most of the improvement over QE because the bias reduction is primarily driven by the iterative delensing, not the prior shape. Conclusion: a non-Gaussian prior is not worth implementing.

### Convergence

5 Newton–Raphson iterations are sufficient for convergence of the N^(3/2)_L bias in polarization. Temperature convergence is faster (fewer iterations needed). This is consistent with Millea et al. and C&L (2017).

### Experimental setup

CMB-S4 wide-field-like: f_sky = 0.40, σ_T = 1 µK-arcmin, σ_P = √2 µK-arcmin, 1 arcmin beam, ℓ_max = 4000. Reconstruction of L ∈ [2, 5120]. Results for TT, EE, EE+EB (pol), and minimum variance (MV) estimators.

### What this means for our pipeline

1. **Temperature MAP mean field:** The 10–20% cross-spectrum bias from the delensing-induced mean field is specific to temperature MAP. Our cross-spectrum σ comparison between MAP and GI is computed from C_L(φ_true, φ̂), so it is **not** biased by the mean field (φ_true is independent of the noise). However, if we ever use the MAP auto-power or compare to a theoretical N_L, this contamination is present.

2. **N^(3/2)_L in our sims:** We use Gaussian lensing fields (CMBLensing.jl draws φ from the Gaussian prior C_L^{φφ}). Our sims therefore have zero N^(3/2)_L by construction. This means our MAP vs QE comparison is measuring the improvement from the iterative delensing alone (W_L → 1), which is valid and clean.

3. **On real data:** The N^(3/2)_L bias would need to be subtracted for a cosmological analysis. MAP reduces it naturally but does not eliminate it entirely (especially at L > 1500 for TT where the paper shows a small residual rise). For a real CMB-S4 analysis using temperature, explicit N^(3/2)_L modelling or bias hardening would still be needed.

4. **Cluster lensing:** For cluster stacking, N^(3/2)_L is not directly relevant since we measure the stacked convergence profile, not the power spectrum. The lensing bispectrum at cluster positions is large but this appears as signal (the cluster mass), not bias.

---

## 6. Flöss et al. (2024) — *Denoising Diffusion Delensing Delight: Reconstructing the Non-Gaussian CMB Lensing Potential with Diffusion Models*

**Reference:** Flöss, T., Coulton, W. R., Duivenvoorden, A. J., Villaescusa-Navarro, F. & Wandelt, B. D. (2024). *Denoising Diffusion Delensing Delight.* arXiv:2405.05598.

### Core contribution

This paper replaces HMC posterior sampling (Millea et al.) with a score-based generative diffusion model (SGM) that learns the conditional posterior P(κ | CMB data) directly from simulation pairs. The trained neural network can draw uncorrelated samples of the lensing convergence map given an observed Q/U map in ~1.75 seconds per sample on an A100 GPU (vs. HMC which produces correlated samples requiring long chains to converge).

The key advantage is that the posterior is **implicitly specified** through training data — no analytic form of the posterior is required. This allows training on non-Gaussian lensing maps from N-body simulations, producing posterior samples with the correct non-Gaussian statistics (one-point PDF, bispectrum) without modifying any equations.

### Why MAP is limited and what this paper addresses

The paper directly addresses two limitations of MAP and HMC:

1. **MAP gives a point estimate, not samples.** Once you have φ̂_MAP, you need expensive additional simulations to propagate uncertainty into derived quantities (e.g. delensed B-mode power, bispectrum). HMC samples the full posterior but takes long correlated chains. Diffusion draws uncorrelated samples instantly.

2. **Gaussian prior on φ.** All analytical methods (QE, MAP, HMC as implemented in CMBLensing.jl) assume a Gaussian prior on φ. The true lensing field is non-Gaussian (from non-linear structure formation and post-Born effects). MAP and HMC with a Gaussian prior still produce good point estimates (MAP N^(3/2)_L is suppressed; see Darwish et al.), but the posterior shape is wrong — samples drawn from the Gaussian-prior posterior have incorrect non-Gaussian statistics. The diffusion model trained on N-body κ maps reproduces the correct bispectrum on signal-dominated scales.

### Experimental setup and validation

- Flat-sky 73 deg² patches, 2 arcmin/pixel, σ_P = 1 µK-arcmin, 2 arcmin beam (polarization only, Q+U input).
- Validation: 2048 SGM samples vs. CMBLensing.jl HMC chain (17000 steps, keep every 5th → 3200 independent). Mean and standard deviation maps agree; power spectra and cross-correlation coefficient ρ_ℓ agree up to ℓ ~ 3000.
- Non-Gaussian case: trained on 25600 flat-sky patches from 100 N-body full-sky κ maps. Bispectrum of samples follows truth on signal-dominated scales and falls back to the non-Gaussian prior on noise-dominated scales. A Gaussian-prior model trained on Gaussian maps fails to recover the bispectrum at noise-dominated scales.

### Connection to our pipeline

This paper is **not directly actionable** for our current pipeline. We are using MAP (point estimate) for power spectrum estimation, which is appropriate for the diffuse lensing power spectrum where:
- The power spectrum is the target statistic — non-Gaussian corrections to the lensing power spectrum are at the ~1–3% level (N^(3/2)_L) and MAP already suppresses these.
- We have access to φ_true in simulations, so we can compute C_L(φ_true, φ̂) directly without needing posterior samples.

The diffusion approach would be relevant if we wanted to:
- Measure non-Gaussian statistics of φ (bispectrum, one-point PDF) from a single reconstruction without biases from noise.
- Propagate reconstruction uncertainty into delensing without expensive simulations.
- Work in a regime where the Gaussian prior is a poor approximation (e.g., clusters with κ ~ 0.3–0.5).

### MAP limitations highlighted by this paper

The paper confirms what Millea et al. and Darwish et al. also noted: MAP is a point estimate, and for analyses beyond the power spectrum it needs expensive forward modelling to remove noise biases. For our diffuse power spectrum analysis this is handled by:
- Using C_L(φ_true, φ̂) (cross-spectrum, bias-free).
- For the auto-spectrum, using RD-N0 debiasing (Legrand & Carron 2023).

For the cluster lensing extension, if we want to measure the convergence profile κ(θ) beyond the mean (e.g., shape statistics, mass function inference), diffusion sampling would be a more powerful approach than stacking point estimates. However, this is beyond current project scope.

### Speed comparison

- SGM: ~1.75 sec/sample (vectorized), fully uncorrelated.
- CMBLensing.jl HMC: ~1.6 sec/sample, but autocorrelation length ~290 → effective independent sample rate ~100× worse.
- MAP: single optimization, much faster than either sampling method.

For a Part III project, MAP is the right balance of optimality and computational cost.

---

## Summary: what improvements are actionable for this project

| Improvement | Paper | Impact | Difficulty |
|---|---|---|---|
| Monitor MAP logpdf convergence; increase steps if needed | Millea et al. | Medium | Low |
| Use cross-spectrum (not auto) as primary MAP figure of merit | C&L (2017) | High | Done |
| Apply GI/QE/MAP to cluster cutouts; compare ρ_L | Horowitz et al. | High | Medium |
| N0 subtraction for cluster stacking | Horowitz et al. | Required for cluster | Medium |
| MAP RD-N0 debiasing | Legrand & Carron (2023) | Required for real data | High |
| Polarization (EB) reconstruction | Legrand & Carron (2023) | Large SNR gain | High |
| N^(3/2)_L modelling or bias hardening | Darwish et al. (2024) | Needed for real data TT | Medium |
| Full posterior sampling (HMC) instead of MAP | Millea et al. / Flöss et al. | Small for diffuse power; useful for non-Gaussian statistics | Very High |
| Diffusion posterior sampling for non-Gaussian statistics | Flöss et al. (2024) | Relevant for cluster κ profile beyond mean | Very High |

For the current Part III project scope, the cluster lensing comparison (Horowitz et al.) is the most actionable novel contribution. Key conclusions from the broader literature:
- MAP is optimal for the diffuse lensing power spectrum at CMB-S4 noise (C&L 2017, Millea et al.)
- MAP naturally suppresses the N^(3/2)_L non-Gaussian bias without explicit modelling (Darwish et al. 2024)
- MAP with temperature has a delensing-induced mean field (~10–20% on cross-spectrum) absent in polarization (Darwish et al. 2024)
- For full Bayesian inference with non-Gaussian statistics, diffusion sampling is faster and more accurate than HMC (Flöss et al. 2024), but this is beyond current scope
