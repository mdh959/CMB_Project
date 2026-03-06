#!/usr/bin/env julia

#run_qe_wl20k.jl

#Compute QE (wiener_filtered=true, LowPass(20000)) W_L normalization from 100 sims
#and save per-sim raw κ cross/auto spectra for plotting by plot_qe_sigma_20k.jl.

#Seeds 1001–1100, matching run_WL_sims.jl convention.
#Δℓ=30 bins, same as run_WL_sims.jl.

#Output:
#  results/WL_qe_20000.jld2      — W_L, R_sims, ℓ_template
# results/spectra_qe_20000.jld2 — per-sim raw (undebiased) κ spectra


import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using Statistics: mean, std
using JLD2
using Printf

include("utils.jl")
using .Utils

# ── Parameters ──────────────────────────────────────────────────────────────
const Cℓ            = camb(r=0.05, ℓmax=21000)
const Cℓn           = noiseCℓs(μKarcminT=1.0, ℓknee=0)
const θpix          = 0.7438046267475303
const Nside         = 512
const pol           = :I
const bandpass_mask = LowPass(20000)
const beamFWHM      = 1.0
const nsims         = 100
const seed0         = 1000
const Δℓ            = 30      # same as run_WL_sims.jl

const WL_file      = "results/WL_rawqe_20000.jld2"
const spectra_file = "results/spectra_rawqe_20000.jld2"

const load_kwargs = (
    Cℓ=Cℓ, Cℓn=Cℓn, θpix=θpix, T=Float64, Nside=Nside,
    beamFWHM=beamFWHM, pol=pol, bandpass_mask=bandpass_mask,
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

# Geometry
const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
println("f_sky_patch = $(round(f_sky_patch; sigdigits=4))")

# ── Resume state ─────────────────────────────────────────────────────────────
R_qe_sims    = Vector{Vector{Float64}}()
ℓ_template   = nothing
W_qe_running = nothing
nsims_completed = 0
seeds_done   = Int[]

# Raw κ spectra: stored in κ units, NOT debiased yet.
# Debiasing applied later in plot_qe_sigma_20k.jl using the converged W_L.
#   Cl_cross_raw(s) = kfac .* C_L^{ϕ_true, ϕ̂_WF}(s)   — biased by W_L
#   Cl_auto_raw(s)  = kfac .* C_L^{ϕ̂_WF, ϕ̂_WF}(s)    — biased by W_L^2
#   Cl_true(s)      = kfac .* C_L^{ϕ_true, ϕ_true}(s)  — unbiased truth
Cl_cross_raw_sims = Vector{Vector{Float64}}()
Cl_auto_raw_sims  = Vector{Vector{Float64}}()
Cl_true_sims      = Vector{Vector{Float64}}()
ℓ_kk              = nothing

if isfile(WL_file)
    jldopen(WL_file, "r") do f
        haskey(f, "R_qe_sims")       && (R_qe_sims       = read(f, "R_qe_sims"))
        haskey(f, "ℓ_template")      && (ℓ_template       = read(f, "ℓ_template"))
        haskey(f, "W_qe_running")    && (W_qe_running     = read(f, "W_qe_running"))
        haskey(f, "nsims_completed") && (nsims_completed  = read(f, "nsims_completed"))
        haskey(f, "seeds_done")      && (seeds_done       = read(f, "seeds_done"))
    end
    println("Resumed W_L checkpoint: $nsims_completed / $nsims sims")
end
if isfile(spectra_file)
    jldopen(spectra_file, "r") do f
        haskey(f, "Cl_cross_raw_sims") && (Cl_cross_raw_sims = read(f, "Cl_cross_raw_sims"))
        haskey(f, "Cl_auto_raw_sims")  && (Cl_auto_raw_sims  = read(f, "Cl_auto_raw_sims"))
        haskey(f, "Cl_true_sims")      && (Cl_true_sims      = read(f, "Cl_true_sims"))
        haskey(f, "ℓ_kk")             && (ℓ_kk              = read(f, "ℓ_kk"))
    end
    println("Resumed spectra checkpoint: $(length(Cl_cross_raw_sims)) / $nsims sims")
end

# ── Main loop ────────────────────────────────────────────────────────────────
println("\n=== QE W_L + spectra ($nsims sims, LowPass(20000), wiener_filtered=false) ===")

for s in (nsims_completed + 1):nsims
    seed = seed0 + s
    print("  Sim $s/$nsims (seed=$seed) ... ")
    flush(stdout)

    (; ϕ, ds) = load_sim(; seed=seed, load_kwargs...)
    qe  = quadratic_estimate(ds; weights=:lensed, wiener_filtered=false)
    ϕqe = qe.ϕqe

    # W_L: R_L(s) = C_L^{ϕ_true, ϕ̂_WF}(s) / C_L^{ϕ_true, ϕ_true}(s)
    # Mean over sims → W_L (QE Wiener-filter transfer function)
    cl_tt = get_Cℓ(ϕ;      Δℓ=Δℓ)   # C_L^{φφ}
    cl_tq = get_Cℓ(ϕ, ϕqe; Δℓ=Δℓ)   # C_L^{φ_true, ϕ̂_WF}
    cl_qq = get_Cℓ(ϕqe;    Δℓ=Δℓ)   # C_L^{ϕ̂_WF, ϕ̂_WF}

    if ℓ_template === nothing
        global ℓ_template = Float64.(collect(cl_tt.ℓ))
    end
    if ℓ_kk === nothing
        global ℓ_kk = Float64.(collect(cl_tt.ℓ))
    end

    # Compute ratio in κ units to avoid precision loss:
    # C^φφ(L=5000) ~ 1e-26, below Float64 threshold, but C^κκ = (L²/2)² C^φφ ~ 1e-10.
    # kfac cancels in the ratio so W_L is identical, but threshold check is safe.
    kfac_l  = @. (ℓ_template^2 / 2)^2
    denom_κ = kfac_l .* Float64.(cl_tt.Cℓ)   # C^{κ_true, κ_true}
    numer_κ = kfac_l .* Float64.(cl_tq.Cℓ)   # C^{κ_true, κ̂}
    R = @. ifelse(abs(denom_κ) > 1e-20, numer_κ / denom_κ, 0.0)
    push!(R_qe_sims, R)
    global W_qe_running = mean(reduce(hcat, R_qe_sims); dims=2)[:]
    push!(seeds_done, seed)
    global nsims_completed += 1

    # κ spectra: C_L^κκ = (L²/2)² × C_L^φφ
    kfac = @. (ℓ_kk^2 / 2)^2
    push!(Cl_cross_raw_sims, kfac .* Float64.(cl_tq.Cℓ))
    push!(Cl_auto_raw_sims,  kfac .* Float64.(cl_qq.Cℓ))
    push!(Cl_true_sims,      kfac .* Float64.(cl_tt.Cℓ))

    # Checkpoint both files
    @save WL_file R_qe_sims ℓ_template W_qe_running nsims_completed seeds_done
    @save spectra_file Cl_cross_raw_sims Cl_auto_raw_sims Cl_true_sims ℓ_kk

    println("done")
    flush(stdout)
end

# ── Diagnostics ──────────────────────────────────────────────────────────────
println("\n=== W_L diagnostics ===")
for ℓ_check in [1000, 2000, 3000, 5000, 7000, 9000, 11000]
    idx = argmin(abs.(ℓ_template .- ℓ_check))
    # Per-sim std of R_L at this bin
    R_at_idx = [R_qe_sims[s][idx] for s in eachindex(R_qe_sims)]
    sem = std(R_at_idx) / sqrt(length(R_at_idx))
    @printf "  W_L(ℓ≈%5d) = %8.4f  ±  %8.4f (1σ/√N)\n" round(Int, ℓ_template[idx]) W_qe_running[idx] sem
end

println("\nDone! $nsims_completed sims completed.")
println("Output: $WL_file, $spectra_file")
