#!/usr/bin/env julia
"""
Standalone script to compute empirical W_L via Monte Carlo simulations.

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
T     = Float64          # overridden to Float64 inside the function anyway
bandpass_mask = LowPass(1000)

nsims = 100
nbins = 300
checkpoint_file = "results/WL_checkpoint64.jld2"

# ── Run ─────────────────────────────────────────────────────────
println("Starting empirical W_L computation ($nsims sims, Float64)")
println("Checkpoint: $checkpoint_file")

out = empirical_WL_maps_loadsim(
    Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
    nsims = nsims,
    nbins = nbins,
    checkpoint_file = checkpoint_file,
)

println("\nDone! $(out.nsims) sims completed.")
println("Checkpoint saved to: $checkpoint_file")
