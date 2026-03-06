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
bandpass_mask = LowPass(8000)

Δℓ = 30   # must match run_WL_sims.jl


WL_checkpoint   = "results/WL_checkpoint_8000.jld2"
phi_maps_file   = "results/phi_maps_8000.jld2"
checkpoint_file = "results/error_analysis_checkpoint_8000.jld2"
results_file    = "results/error_analysis_final_8000.jld2"
plot_path       = "results/noise_curves_combined_8000.png"

# ── Run error analysis ───────────────────────────────────────────
println("Starting error analysis")
println("WL checkpoint:    $WL_checkpoint")
println("Phi maps:         $phi_maps_file")
println("Error checkpoint: $checkpoint_file")

results = run_error_analysis_all_debiased(
    Cℓ, Cℓn, θpix, Nside, pol, bandpass_mask;
    Δℓ                   = Δℓ,
    WL_checkpoint        = WL_checkpoint,
    phi_maps_file        = phi_maps_file,
    checkpoint_file      = checkpoint_file,
    beamFWHM             = 1.0,
    progress             = true,
    max_sims             = 100,
    raw_qe_phi_file      = "results/phi_maps_raw_qe.jld2",
    WL_raw_qe_checkpoint = "results/WL_raw_qe_checkpoint.jld2",
)

# ── Save results and plot ────────────────────────────────────────
save_error_results(results, Δℓ, Nside, θpix;
    results_file = results_file,
    plot_path    = plot_path,
)

println("\nDone! $(results.nsims) sims completed.")
println("Results: $results_file")
println("Plot:    $plot_path")
