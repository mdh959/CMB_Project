import Pkg
Pkg.activate(@__DIR__)

using JLD2
using PythonPlot
using Statistics: mean, std

checkpoint_file = "results/error_analysis_checkpoint_8000.jld2"
results_file    = "results/error_analysis_final_8000.jld2"

# grad_binned_*_sims: Vector of length-nbins vectors, one per sim.
# Each entry is the mean Δϕ² for pixels in that |∇T| bin.
@load checkpoint_file grad_binned_qe_sims grad_binned_joint_sims grad_binned_marg_sims grad_bin_cen nsims_completed
println("Loaded per-sim curves: $nsims_completed sims")

@load results_file mean_grad_counts nsims

# bin_rms_stats: for each gradient bin, averages the per-sim MSE (mean Δϕ²) across
# sims, then converts to RMS via sqrt. SEM is computed on the MSE and propagated
# through the sqrt via the delta method: σ(√μ) ≈ σ_μ / (2√μ).
# This order (mean then sqrt) is unbiased; taking sqrt per-sim first would give
# mean(sqrt(x)) < sqrt(mean(x)) by Jensen's inequality.
function bin_rms_stats(sims_vec)
    mat   = reduce(hcat, sims_vec)   # [nbins × nsims], per-sim mean Δϕ² per bin
    nbins = size(mat, 1)
    rms   = Vector{Float64}(undef, nbins)
    drms  = Vector{Float64}(undef, nbins)
    for i in 1:nbins
        v = filter(isfinite, mat[i, :])
        n = length(v)
        if n < 1; rms[i] = NaN; drms[i] = NaN; continue; end
        μ       = mean(v)
        sem_mse = n > 1 ? std(v) / sqrt(n) : NaN    # SEM on mean MSE
        rms[i]  = sqrt(abs(μ))
        drms[i] = sem_mse / (2 * max(rms[i], 1e-30)) # delta method
    end
    return rms, drms
end

rms_qe,    sem_qe    = bin_rms_stats(grad_binned_qe_sims)
rms_joint, sem_joint = bin_rms_stats(grad_binned_joint_sims)
rms_marg,  sem_marg  = bin_rms_stats(grad_binned_marg_sims)

cen   = Float64.(collect(grad_bin_cen))
nbins = length(cen)

PythonPlot.rc("font",        size=12, family="serif")
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)

colours = ["#4477AA", "#EE6677", "#228833"]

# Three-panel figure:
#   Top:    RMS(Δϕ) vs |∇T| — shows gradient-dependent noise for each estimator
#   Middle: ratio RMS_QE / RMS_MAP — improvement factor as a function of gradient
#   Bottom: pixel counts per bin — shows dynamic range and bin quality
fig1, (ax1, ax2, ax3) = PythonPlot.subplots(
    3, 1, figsize=(5.5, 8),
    gridspec_kw=Dict("height_ratios" => [3, 1.5, 1], "hspace" => 0.05),
    sharex=true
)

# log y-scale chosen because RMS spans ~2 decades across estimators.
# lo = max(μ - σ, 0.01μ) prevents the shaded band from crossing zero on log scale.
for (μ, σ, col, lab, mk) in [
    (rms_qe,    sem_qe,    colours[1], "QE",        "o"),
    (rms_joint, sem_joint, colours[2], "MAP joint", "s"),
    (rms_marg,  sem_marg,  colours[3], "MAP marg",  "^"),
]
    ok = isfinite.(μ) .& isfinite.(σ) .& (μ .> 0)
    ax1.plot(cen[ok], μ[ok],
             marker=mk, color=col, label=lab,
             markersize=4, linewidth=1.2,
             markeredgewidth=0.6, markeredgecolor="white")
    lo = max.(μ[ok] .- σ[ok], μ[ok] .* 0.01)
    ax1.fill_between(cen[ok], lo, μ[ok] .+ σ[ok], alpha=0.20, color=col)
end

ax1.set_yscale("log")
ax1.set_ylabel(L"\sqrt{\langle|\Delta\phi|^2\rangle}", fontsize=13)
ax1.legend(frameon=false, fontsize=11)
ax1.tick_params(labelsize=11, labelbottom=false)
ax1.set_title("$nsims_completed sims  (shading = ±1 SEM across sims)", fontsize=9, color="grey")
ax1.spines["top"].set_visible(false)
ax1.spines["right"].set_visible(false)

# Ratio panel: QE / MAP gives improvement factor > 1 where MAP outperforms.
# Error bars from standard propagation: σ(A/B)/（A/B) = √[(σ_A/A)² + (σ_B/B)²].
ratio_joint = rms_qe ./ rms_joint
ratio_marg  = rms_qe ./ rms_marg

err_ratio_joint = ratio_joint .* sqrt.((sem_qe ./ rms_qe).^2 .+ (sem_joint ./ rms_joint).^2)
err_ratio_marg  = ratio_marg  .* sqrt.((sem_qe ./ rms_qe).^2 .+ (sem_marg  ./ rms_marg ).^2)

for (rat, err, col, lab, mk) in [
    (ratio_joint, err_ratio_joint, colours[2], "QE / MAP joint", "s"),
    (ratio_marg,  err_ratio_marg,  colours[3], "QE / MAP marg",  "^"),
]
    ok = isfinite.(rat) .& isfinite.(err)
    ax2.plot(cen[ok], rat[ok],
             marker=mk, color=col, label=lab,
             markersize=4, linewidth=1.2,
             markeredgewidth=0.6, markeredgecolor="white")
    ax2.fill_between(cen[ok], rat[ok] .- err[ok], rat[ok] .+ err[ok], alpha=0.20, color=col)
end

ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
ax2.set_ylabel("RMS ratio\n(QE / MAP)", fontsize=11)
ax2.legend(frameon=false, fontsize=10)
ax2.tick_params(labelsize=11, labelbottom=false)
ax2.spines["top"].set_visible(false)
ax2.spines["right"].set_visible(false)

# Pixel-count panel on log scale — confirms that high-|∇T| bins contain many pixels
# (otherwise the RMS estimates in those bins would be noisy)
bin_width = length(cen) > 1 ? cen[2] - cen[1] : 1.0
ax3.bar(cen, mean_grad_counts, width=0.85 * bin_width, color="grey", alpha=0.6, edgecolor="none")
ax3.set_yscale("log")
ax3.set_xlabel(L"|\nabla T|\;\mathrm{(per\;pixel)}", fontsize=13)
ax3.set_ylabel(L"\langle N_\mathrm{pix}\rangle", fontsize=11)
ax3.tick_params(labelsize=10)
ax3.spines["top"].set_visible(false)
ax3.spines["right"].set_visible(false)

fig1.subplots_adjust(hspace=0.05)
outpath = "results/phi_error_vs_gradT_8000.png"
fig1.savefig(outpath, dpi=150, bbox_inches="tight")
println("Saved to $outpath")

PythonPlot.plotclose("all")
