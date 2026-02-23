import Pkg
Pkg.activate(@__DIR__)

using CMBLensing
using PythonPlot
using Statistics
using LinearAlgebra
using JLD2

include("utils.jl")
using .Utils

# ── Parameters (must match run_WL_sims.jl) ──────────────────────
Cℓ  = camb(r=0.05, ℓmax=21000)
Cℓn = noiseCℓs(μKarcminT=1.0, ℓknee=0)

θpix  = 0.7438046267475303
Nside = 512
pol   = :I
bandpass_mask = LowPass(6000)

# Define serial fallback only once (if the installed CMBLensing lacks these)
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset() = _DIST_DS[]
    end
end

nsims = 100
Δℓ    = 50

WL_checkpoint    = "results/WL_checkpoint.jld2"
checkpoint_file  = "results/error_analysis_checkpoint.jld2"
results_file     = "results/error_analysis_final.jld2"
WL_checkpoint_qe = "results/WL_checkpoint_qe.jld2"  # QE-only version
checkpoint_file_qe = "results/error_analysis_checkpoint_qe.jld2"  # QE-only version
results_file_qe    = "results/error_analysis_final_qe.jld2"  # QE-only version


main_file = "results/error_analysis_final.jld2"        # has ℓ, N_joint, N_marg (and maybe N_qe)
qe_file   = "results/error_analysis_final_qe.jld2"     # has ℓ, N_qe

# ---- load ----
@load main_file ℓ N_joint N_marg
@load qe_file   ℓ N_qe

# ── Run ──────────────────────────────────────────────────────────
println("Starting error analysis ($nsims sims)")
println("WL checkpoint:  $WL_checkpoint")
println("Error checkpoint: $checkpoint_file")


# ---- plot ----
plot_noise_curves(ℓ, N_joint, N_marg, N_qe, Δℓ; Nside=Nside, θpix=θpix,
                  savepath="results/noise_curves_combined.png")

println("Saved: results/noise_curves_combined.png")


results = run_error_analysis(
    Cℓ, Cℓn, θpix, Nside, pol, bandpass_mask;
    nsims = nsims,
    Δℓ = Δℓ,
    checkpoint_file = checkpoint_file,
    WL_checkpoint = WL_checkpoint,
    seed0 = 1000,
)

println("\nDone! $(results.nsims) sims completed.")


# ── Save final results + plot ────────────────────────────────────
save_error_results(results, Δℓ, Nside, θpix;
    results_file = results_file,
    plot_path = "results/error_analysis_plot.png",
)
