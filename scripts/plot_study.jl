# Loads saved results and produces all thesis plots.
#
# Usage:
#   julia --project plot_thesis.jl

using PythonCall   # must load before CMBLensing to register Python extension
include("study_helpers.jl")
using PythonPlot

mkpath("plots")

# Theory Cϕ for reference curves in N(L) plots
# (same camb call as run_study.jl — memoized so fast on repeat runs)
const Cℓ_theory = camb(r=0.05, ℓmax=21000)

# -----------------------------------------------------------------------
# Colour/style conventions — consistent across all plots
# -----------------------------------------------------------------------
const COLORS = (;
    qe_raw    = "#888888",
    qe_wf     = "#2196F3",
    map_joint = "#E91E63",
    map_marg  = "#FF9800",
)
const LABELS = (;
    qe_raw    = "QE (raw)",
    qe_wf     = "QE (WF)",
    map_joint = "MAP joint",
    map_marg  = "MAP marg",
)
const METHODS = (:qe_raw, :qe_wf, :map_joint, :map_marg)

# -----------------------------------------------------------------------
# Helper: load all seeds for a given study/key prefix
# -----------------------------------------------------------------------
function load_seeds(filename, key_prefix, seeds=1:20)
    results = []
    jldopen(filename, "r") do f
        for seed in seeds
            key = "$(key_prefix)_seed_$(seed)"
            key in keys(f) && push!(results, f[key])
        end
    end
    return results
end

# -----------------------------------------------------------------------
# Helper: mean and std of a metric across seeds
# -----------------------------------------------------------------------
function mean_std(results, field, method)
    vals = [getfield(getfield(r, field), method) for r in results]
    m = mean(vals)
    s = std(vals) ./ sqrt(length(vals))   # standard error
    return m, s
end

# -----------------------------------------------------------------------
# PLOT 1: ρ(L) for all 4 methods at each noise level
# -----------------------------------------------------------------------
function plot_rho_noise()
    NOISE_LEVELS = [5.0, 2.0, 1.0, 0.5, 0.1]
    NOISE_LABELS = ["5.0", "2.0", "1.0", "0.5", "0.1"]

    fig, axes = subplots(1, 5, figsize=(18, 4), sharey=true)
    fig.suptitle("Reconstruction correlation ρ(L) — noise dependence", fontsize=13)

    for (ax, _, label) in zip(axes, NOISE_LEVELS, NOISE_LABELS)
        results = load_seeds("results/noise_study.jld2", "noise_$(label)")
        isempty(results) && continue
        ℓ = results[1].ℓ

        for m in METHODS
            ρ_mean, ρ_err = mean_std(results, :rho, m)
            ax.plot(ℓ, ρ_mean, color=COLORS[m], label=LABELS[m], lw=1.8)
            ax.fill_between(ℓ, ρ_mean .- ρ_err, ρ_mean .+ ρ_err,
                            color=COLORS[m], alpha=0.15)
        end

        ax.set_title("$(label) μK-arcmin", fontsize=10)
        ax.set_xlabel("L")
        ax.set_xscale("log")
        ax.set_xlim(50, 7000)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, ls="--", color="k", lw=0.8, alpha=0.4)
        ax.grid(true, alpha=0.3)
    end

    axes[1].set_ylabel("ρ(L)")
    axes[3].legend(loc="lower left", fontsize=8)
    tight_layout()
    savefig("plots/rho_noise.pdf", bbox_inches="tight")
    println("Saved plots/rho_noise.pdf")
    PythonPlot.close(fig)
end

# -----------------------------------------------------------------------
# PLOT 2: N(L) reconstruction noise at each noise level
# -----------------------------------------------------------------------
function plot_NL_noise()
    NOISE_LEVELS = [5.0, 2.0, 1.0, 0.5, 0.1]
    NOISE_LABELS = ["5.0", "2.0", "1.0", "0.5", "0.1"]

    fig, axes = subplots(1, 5, figsize=(18, 4), sharey=false)
    fig.suptitle("Reconstruction noise N(L) — noise dependence", fontsize=13)

    for (ax, _, label) in zip(axes, NOISE_LEVELS, NOISE_LABELS)
        results = load_seeds("results/noise_study.jld2", "noise_$(label)")
        isempty(results) && continue
        ℓ = results[1].ℓ

        for m in METHODS
            NL_mean, _ = mean_std(results, :NL, m)
            ax.plot(ℓ, ℓ.^4 .* NL_mean, color=COLORS[m], label=LABELS[m], lw=1.8)
        end

        # overplot signal Cϕ for reference (from theory)
        Cϕ_th = Cℓ_theory.unlensed_total.ϕϕ
        ax.plot(ℓ, ℓ.^4 .* Cϕ_th[ℓ], "k--", lw=1.0, alpha=0.5, label="Cϕ signal")

        ax.set_title("$(label) μK-arcmin", fontsize=10)
        ax.set_xlabel("L")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(50, 7000)
        ax.grid(true, alpha=0.3)
    end

    axes[1].set_ylabel("L⁴ N(L)")
    axes[3].legend(loc="upper left", fontsize=8)
    tight_layout()
    savefig("plots/NL_noise.pdf", bbox_inches="tight")
    println("Saved plots/NL_noise.pdf")
    PythonPlot.close(fig)
end

# -----------------------------------------------------------------------
# PLOT 3: Fractional improvement (N_QE_raw - N_MAP) / N_QE_raw
# -----------------------------------------------------------------------
function plot_fractional_improvement_noise()
    NOISE_LEVELS = [5.0, 2.0, 1.0, 0.5, 0.1]
    NOISE_LABELS = ["5.0", "2.0", "1.0", "0.5", "0.1"]

    fig, axes = subplots(1, 5, figsize=(18, 4), sharey=true)
    fig.suptitle("Fractional improvement in N(L) over QE raw — noise dependence", fontsize=13)

    for (ax, _, label) in zip(axes, NOISE_LEVELS, NOISE_LABELS)
        results = load_seeds("results/noise_study.jld2", "noise_$(label)")
        isempty(results) && continue
        ℓ = results[1].ℓ

        NL_qe_raw_mean, _ = mean_std(results, :NL, :qe_raw)

        for m in (:qe_wf, :map_joint, :map_marg)
            NL_mean, _ = mean_std(results, :NL, m)
            frac = @. (NL_qe_raw_mean - NL_mean) / NL_qe_raw_mean
            ax.plot(ℓ, frac .* 100, color=COLORS[m], label=LABELS[m], lw=1.8)
        end

        ax.axhline(0, ls="--", color="k", lw=0.8, alpha=0.4)
        ax.set_title("$(label) μK-arcmin", fontsize=10)
        ax.set_xlabel("L")
        ax.set_xscale("log")
        ax.set_xlim(50, 7000)
        ax.set_ylim(-5, 60)
        ax.grid(true, alpha=0.3)
    end

    axes[1].set_ylabel("(N_QE_raw − N_est) / N_QE_raw  [%]")
    axes[3].legend(loc="upper left", fontsize=8)
    tight_layout()
    savefig("plots/fractional_improvement_noise.pdf", bbox_inches="tight")
    println("Saved plots/fractional_improvement_noise.pdf")
    PythonPlot.close(fig)
end

# -----------------------------------------------------------------------
# PLOT 4: Summary — improvement vs noise level at fixed ℓ bins
# -----------------------------------------------------------------------
function plot_improvement_vs_noise()
    NOISE_LEVELS = [5.0, 2.0, 1.0, 0.5, 0.1]
    NOISE_LABELS = ["5.0", "2.0", "1.0", "0.5", "0.1"]

    # ℓ bins to summarise (indices into ℓedges_ϕ bands)
    summary_ℓ_ranges = [(500,700), (1000,1500), (2000,3000), (4000,5000)]

    fig, axes = subplots(2, 2, figsize=(10, 8), sharey=false)
    axes = vec(axes)
    fig.suptitle("MAP improvement vs noise level", fontsize=13)

    for (ax, (ℓlo, ℓhi)) in zip(axes, summary_ℓ_ranges)
        for m in (:qe_wf, :map_joint, :map_marg)
            improvements = Float64[]
            for label in NOISE_LABELS
                results = load_seeds("results/noise_study.jld2", "noise_$(label)")
                isempty(results) && continue
                ℓ = results[1].ℓ

                # find bins in range
                idx = findall(x -> ℓlo ≤ x ≤ ℓhi, ℓ)
                isempty(idx) && continue

                NL_ref_mean,  _ = mean_std(results, :NL, :qe_raw)
                NL_mean, _      = mean_std(results, :NL, m)

                frac = mean(@. (NL_ref_mean[idx] - NL_mean[idx]) / NL_ref_mean[idx]) * 100
                push!(improvements, frac)
            end
            ax.plot(NOISE_LEVELS[1:length(improvements)], improvements,
                    "o-", color=COLORS[m], label=LABELS[m], lw=1.8, ms=5)
        end

        ax.axhline(0, ls="--", color="k", lw=0.8, alpha=0.4)
        ax.set_title("L = $(ℓlo)–$(ℓhi)", fontsize=10)
        ax.set_xlabel("Noise (μK-arcmin)")
        ax.set_ylabel("% improvement over QE raw")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.grid(true, alpha=0.3)
        ax.legend(fontsize=8)
    end

    tight_layout()
    savefig("plots/improvement_vs_noise.pdf", bbox_inches="tight")
    println("Saved plots/improvement_vs_noise.pdf")
    PythonPlot.close(fig)
end

# -----------------------------------------------------------------------
# PLOT 5: Bandpass dependence — ρ(L) for MAP_joint at each lmax
# -----------------------------------------------------------------------
function plot_bandpass()
    BANDPASS_LEVELS = [4000, 6000, 8000, 12000]
    fig, axes = subplots(1, 4, figsize=(16, 4), sharey=true)
    fig.suptitle("Bandpass dependence — ρ(L) at μK=0.1 arcmin", fontsize=13)

    for (ax, lmax) in zip(axes, BANDPASS_LEVELS)
        results = load_seeds("results/bandpass_study.jld2", "lmax_$(lmax)")
        isempty(results) && continue
        ℓ = results[1].ℓ

        for m in METHODS
            ρ_mean, _ = mean_std(results, :rho, m)
            ax.plot(ℓ, ρ_mean, color=COLORS[m], label=LABELS[m], lw=1.8)
        end

        ax.axvline(lmax, ls=":", color="gray", lw=1.0, alpha=0.6, label="bandpass")
        ax.set_title("lmax = $(lmax)", fontsize=10)
        ax.set_xlabel("L")
        ax.set_xscale("log")
        ax.set_xlim(50, 7000)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, ls="--", color="k", lw=0.8, alpha=0.4)
        ax.grid(true, alpha=0.3)
    end

    axes[1].set_ylabel("ρ(L)")
    axes[2].legend(loc="lower left", fontsize=8)
    tight_layout()
    savefig("plots/bandpass_rho.pdf", bbox_inches="tight")
    println("Saved plots/bandpass_rho.pdf")
    PythonPlot.close(fig)
end

# -----------------------------------------------------------------------
# PLOT 6: QE_raw vs QE_WF vs MAP — the key WF comparison
# Single panel at 0.1 μK-arcmin, clear visual showing WF accounts for most gain
# -----------------------------------------------------------------------
function plot_wf_comparison()
    results = load_seeds("results/noise_study.jld2", "noise_0.1")
    isempty(results) && (println("No results for 0.1 μK"); return)
    ℓ = results[1].ℓ

    fig, axes = subplots(1, 2, figsize=(12, 5))
    fig.suptitle("QE raw vs WF vs MAP at 0.1 μK-arcmin", fontsize=13)

    # ρ(L)
    ax = axes[1]
    for m in METHODS
        ρ_mean, ρ_err = mean_std(results, :rho, m)
        ax.plot(ℓ, ρ_mean, color=COLORS[m], label=LABELS[m], lw=2)
        ax.fill_between(ℓ, ρ_mean .- ρ_err, ρ_mean .+ ρ_err,
                        color=COLORS[m], alpha=0.15)
    end
    ax.set_xlabel("L"); ax.set_ylabel("ρ(L)")
    ax.set_xscale("log"); ax.set_xlim(50, 7000); ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls="--", color="k", lw=0.8, alpha=0.4)
    ax.legend(fontsize=9); ax.grid(true, alpha=0.3)
    ax.set_title("Cross-correlation coefficient")

    # Fractional improvement over QE_raw
    ax = axes[2]
    NL_ref, _ = mean_std(results, :NL, :qe_raw)
    for m in (:qe_wf, :map_joint, :map_marg)
        NL_mean, _ = mean_std(results, :NL, m)
        frac = @. (NL_ref - NL_mean) / NL_ref * 100
        ax.plot(ℓ, frac, color=COLORS[m], label=LABELS[m], lw=2)
    end
    ax.axhline(0, ls="--", color="k", lw=0.8, alpha=0.4)
    ax.set_xlabel("L"); ax.set_ylabel("% improvement over QE raw")
    ax.set_xscale("log"); ax.set_xlim(50, 7000)
    ax.legend(fontsize=9); ax.grid(true, alpha=0.3)
    ax.set_title("Fractional N(L) improvement")

    tight_layout()
    savefig("plots/wf_comparison.pdf", bbox_inches="tight")
    println("Saved plots/wf_comparison.pdf")
    PythonPlot.close(fig)
end

# -----------------------------------------------------------------------
# Run all plots
# -----------------------------------------------------------------------
println("Generating plots...")
plot_rho_noise()
plot_NL_noise()
plot_fractional_improvement_noise()
plot_improvement_vs_noise()
plot_bandpass()
plot_wf_comparison()
println("Done. All plots in plots/")
