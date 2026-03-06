#!/usr/bin/env julia
"""
Script to compute empirical W_L via Monte Carlo simulations.

Usage:
    julia run_WL_sims.jl

Outputs:
    results/WL_checkpoint.jld2  (checkpointed after each sim)
"""

import Pkg
Pkg.activate(@__DIR__)

using CMBLensing
using PythonPlot
using Statistics
using LinearAlgebra
using JLD2

include("utils.jl")
using .Utils

# ── Parameters ──────────────────────────────────────────────────
Cℓ  = camb(r=0.05, ℓmax=21000)
Cℓn = noiseCℓs(μKarcminT=1.0, ℓknee=0)

θpix  = 0.7438046267475303
Nside = 512
pol   = :I
T     = Float64          
bandpass_mask = LowPass(8000)

nsims = 100

# Define serial fallback only once (if the installed CMBLensing lacks these)
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset() = _DIST_DS[]
    end
end

checkpoint_file = "results/WL_checkpoint_8000.jld2"

# ── Run ─────────────────────────────────────────────────────────
println("Starting empirical W_L computation ($nsims sims, Float64)")
println("Checkpoint: $checkpoint_file")

out = empirical_WL_all_loadsim(
    Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
    nsims = nsims,
    Δℓ = 30,
    seed0 = 1000,
    beamFWHM = 1.0,
    qe_wiener_filtered = true,
    checkpoint_file = checkpoint_file,
    phi_maps_file  = "results/phi_maps_8000.jld2",
)



println("\nDone! $(out.nsims) sims completed.")
println("Checkpoint saved to: $checkpoint_file")
