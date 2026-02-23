import Pkg
Pkg.activate(@__DIR__)

using CMBLensing
using JLD2
using PythonPlot
using Statistics: mean, std, quantile

include("utils.jl")
using .Utils

checkpoint_file = "results/error_analysis_checkpoint.jld2"
checkpoint_file_qe = "results/error_analysis_checkpoint_qe.jld2"  # QE-only version
results_file    = "results/error_analysis_final.jld2"
results_file_qe = "results/error_analysis_final_qe.jld2"  # QE-only version

# ── Load per-sim gradient-binned curves from checkpoint ───────────────
# Needed for cross-sim error bars; the final results file only stores means.
@load checkpoint_file grad_binned_joint_sims grad_binned_marg_sims grad_bin_cen nsims_completed
@load checkpoint_file_qe grad_binned_qe_sims grad_bin_cen nsims_completed  # QE-only version
println("Loaded per-sim curves: $nsims_completed sims")

# ── Load averaged 2D maps from final results ──────────────────────────
@load results_file mean_Δϕ²_qe mean_Δϕ²_joint mean_Δϕ²_marg nsims
@load results_file_qe mean_Δϕ²_qe nsims  # QE-only version (no joint/marg)

# ── Load one representative sim for the gradient histogram ────────────
# Parameters match run_error_analysis.jl exactly. seed0=2000 → first sim
# uses seed=1001, which is also where grad_bin_cen was recorded, so the
# histogram edges align perfectly with the bin centres above.
println("Loading representative sim for gradient histogram...")
Cℓ  = camb(r=0.05, ℓmax=21000)
Cℓn = noiseCℓs(μKarcminT=1.0, ℓknee=0)
(; f) = load_sim(
    seed=1001, Cℓ=Cℓ, Cℓn=Cℓn,
    θpix=0.7438046267475303, T=Float64, Nside=512,
    beamFWHM=1.0, pol=:I,
    bandpass_mask=LowPass(6000),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)
_, _, grad_mag = grad_fft(f)
grad_vec = vec(Float64.(grad_mag))
println("Gradient map ready.")

# ── Per-bin RMS mean and SEM across sims ──────────────────────────────
# Plot RMS = sqrt(<|Δϕ|²>) rather than MSE: same units as ϕ, more readable.
# SEM = std/√N is the uncertainty on the mean curve given nsims realisations.
function bin_rms_stats(sims_vec)
    mat   = sqrt.(abs.(reduce(hcat, sims_vec)))  # [nbins × nsims], per-sim RMS
    nbins = size(mat, 1)
    μ     = Vector{Float64}(undef, nbins)
    sem   = Vector{Float64}(undef, nbins)
    for i in 1:nbins
        vals  = filter(isfinite, mat[i, :])
        n     = length(vals)
        μ[i]  = n > 0 ? mean(vals)          : NaN
        sem[i]= n > 1 ? std(vals) / sqrt(n) : NaN
    end
    return μ, sem
end

rms_qe,    sem_qe    = bin_rms_stats(grad_binned_qe_sims)
rms_joint, sem_joint = bin_rms_stats(grad_binned_joint_sims)
rms_marg,  sem_marg  = bin_rms_stats(grad_binned_marg_sims)



cen   = grad_bin_cen
nbins = length(cen)

# ── Plot style ────────────────────────────────────────────────────────
PythonPlot.rc("font",        size=12, family="serif")
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)

colours = ["#4477AA", "#EE6677", "#228833"]

# ── Figure 1: RMS(Δϕ) vs |∇T|  [3 panels] ───────────────────────────
fig1, (ax1, ax2, ax3) = PythonPlot.subplots(3, 1, figsize=(5.5, 8),
    gridspec_kw=Dict("height_ratios" => [3, 1.5, 1], "hspace" => 0.05),
    sharex=true)

# ── Top: RMS per estimator, log y-scale, ±1 SEM shaded band ──
for (μ, σ, col, lab, mk) in [
    (rms_qe,    sem_qe,    colours[1], "QE",       "o"),
    (rms_joint, sem_joint, colours[2], "MAP joint", "s"),
    (rms_marg,  sem_marg,  colours[3], "MAP marg",  "^"),
]
    ok = isfinite.(μ) .& isfinite.(σ)
    ax1.plot(cen[ok], μ[ok],
             marker=mk, color=col, label=lab,
             markersize=4, linewidth=1.2,
             markeredgewidth=0.6, markeredgecolor="white")
    lo = max.(μ[ok] .- σ[ok], μ[ok] .* 0.01)  # clip so log scale is safe
    ax1.fill_between(cen[ok], lo, μ[ok] .+ σ[ok], alpha=0.2, color=col)
end

ax1.set_yscale("log")
ax1.set_ylabel(L"\sqrt{\langle|\Delta\phi|^2\rangle}", fontsize=13)
ax1.legend(frameon=false, fontsize=11)
ax1.tick_params(labelsize=11, labelbottom=false)
ax1.set_title("$nsims_completed sims  (shading = ±1 SEM across sims)", fontsize=9, color="grey")
ax1.spines["top"].set_visible(false)
ax1.spines["right"].set_visible(false)

# ── Middle: improvement ratio QE/MAP with propagated error bands ──
ratio_joint = rms_qe ./ rms_joint
ratio_marg  = rms_qe ./ rms_marg

# Error propagation for ratio a/b:  σ_r = r × sqrt((σ_a/a)² + (σ_b/b)²)
err_ratio_joint = ratio_joint .* sqrt.((sem_qe ./ rms_qe) .^ 2 .+ (sem_joint ./ rms_joint) .^ 2)
err_ratio_marg  = ratio_marg  .* sqrt.((sem_qe ./ rms_qe) .^ 2 .+ (sem_marg  ./ rms_marg)  .^ 2)

for (rat, err, col, lab, mk) in [
    (ratio_joint, err_ratio_joint, colours[2], "QE / MAP joint", "s"),
    (ratio_marg,  err_ratio_marg,  colours[3], "QE / MAP marg",  "^"),
]
    ok = isfinite.(rat) .& isfinite.(err)
    ax2.plot(cen[ok], rat[ok],
             marker=mk, color=col, label=lab,
             markersize=4, linewidth=1.2,
             markeredgewidth=0.6, markeredgecolor="white")
    ax2.fill_between(cen[ok], rat[ok] .- err[ok], rat[ok] .+ err[ok],
                     alpha=0.2, color=col)
end

ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
ax2.set_ylabel("RMS ratio\n(QE / MAP)", fontsize=11)
ax2.legend(frameon=false, fontsize=10)
ax2.tick_params(labelsize=11, labelbottom=false)
ax2.spines["top"].set_visible(false)
ax2.spines["right"].set_visible(false)

# ── Bottom: N_pixels histogram of |∇T| ──
# Bin edges match the analysis binning (same seed, same bin_stat_err logic),
# so the x-axis is shared exactly with the panels above.
lo_g, hi_g  = minimum(grad_vec), maximum(grad_vec)
hist_edges  = collect(range(lo_g, hi_g; length=nbins + 1))

ax3.hist(grad_vec, bins=hist_edges, color="grey", alpha=0.6, edgecolor="none")

# Log scale on N_pixels
ax3.set_yscale("log")

ax3.set_xlabel(L"|\nabla T|\;\mathrm{(per\;pixel)}", fontsize=13)
ax3.set_ylabel(L"N_\mathrm{pixels}", fontsize=11)
ax3.tick_params(labelsize=10)
ax3.spines["top"].set_visible(false)
ax3.spines["right"].set_visible(false)

fig1.subplots_adjust(hspace=0.05)
fig1.savefig("results/phi_error_vs_gradT.png", dpi=150, bbox_inches="tight")
println("Saved to results/phi_error_vs_gradT.png")

PythonPlot.plotclose("all")