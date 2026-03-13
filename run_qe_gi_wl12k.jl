#!/usr/bin/env julia

# run_qe_gi_wl12k.jl
#
# For each sim: compute QE raw (wiener_filtered=false), QE WF (wiener_filtered=true), GI.
# Stores raw phi maps (ϕ_true, ϕ_qe, ϕ_qe_wf, ϕ_gi) in phi_maps_qe_gi_*.jld2.
#
# W_L is estimated as ratio-of-ensemble-means (more stable than mean-of-per-sim-ratios):
#   W_L = mean_sims[C(ϕ_true, ϕ̂)] / mean_sims[C(ϕ_true, ϕ_true)]
#
# The plot script (plot_qe_gi_sigma_12k.jl) debiases ϕ̂ at the phi level
# and computes σ from std of per-sim debiased spectra.

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using LinearAlgebra
using Statistics: mean
using JLD2
using Printf

include("utils.jl")
using .Utils

const Cℓ       = camb(r=0.05, ℓmax=21000)
const θpix     = 0.7438046267475303
const Nside    = 512
const pol      = :I
const nsims    = 400
const seed0    = 1000
const Δℓ       = 30

const GI_Lgrad = 2000
const GI_Lhp   = 4000

const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
println("f_sky_patch = $(round(f_sky_patch; sigdigits=4))")

function run_noise_level(μKarcminT::Float64, suffix::String, Lmax::Int, beamFWHM::Float64)
    println("\n" * "="^70)
    println("=== Noise: μKarcminT=$μKarcminT, Lmax=$Lmax, beamFWHM=$(beamFWHM)arcmin  (suffix=\"$suffix\") ===")
    println("="^70)

    Cℓn      = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    bandpass = LowPass(Lmax)
    GI_Lmax  = Lmax

    WL_file       = "results/WL_qe_gi_12000$(suffix).jld2"
    phi_maps_file = "results/phi_maps_qe_gi_12000$(suffix).jld2"

    load_kwargs = (
        Cℓ=Cℓ, Cℓn=Cℓn, θpix=θpix, T=Float64, Nside=Nside,
        beamFWHM=beamFWHM, pol=pol, bandpass_mask=bandpass,
        pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
    )

    # ── Resume state ─────────────────────────────────────────────────────────
    R_qe_sims    = Vector{Vector{Float64}}()
    R_qe_wf_sims = Vector{Vector{Float64}}()
    R_gi_sims    = Vector{Vector{Float64}}()
    W_qe         = nothing
    W_qe_wf      = nothing
    W_gi         = nothing
    ℓ_template   = nothing
    nsims_completed = 0
    seeds_done      = Int[]

    if isfile(WL_file)
        d = JLD2.load(WL_file)
        haskey(d, "R_qe_sims")       && (R_qe_sims       = d["R_qe_sims"])
        haskey(d, "R_qe_wf_sims")    && (R_qe_wf_sims    = d["R_qe_wf_sims"])
        haskey(d, "R_gi_sims")       && (R_gi_sims        = d["R_gi_sims"])
        haskey(d, "ℓ_template")      && (ℓ_template       = d["ℓ_template"])
        haskey(d, "nsims_completed") && (nsims_completed  = d["nsims_completed"])
        haskey(d, "seeds_done")      && (seeds_done       = d["seeds_done"])
        println("Resumed W_L checkpoint: $nsims_completed / $nsims sims")
    end

    phi_sims_done = Set{Int}()
    if isfile(phi_maps_file)
        try
            jldopen(phi_maps_file, "r") do f
                for key in keys(f)
                    m = match(r"^sim_(\d+)$", key)
                    m !== nothing && push!(phi_sims_done, parse(Int, m.captures[1]))
                end
            end
            println("$(length(phi_sims_done)) phi maps already in $phi_maps_file")
        catch err
            @warn "phi_maps_file appears corrupted ($err) — deleting"
            rm(phi_maps_file)
        end
    end

    # ── Main loop ─────────────────────────────────────────────────────────────
    for s in (nsims_completed + 1):nsims
        seed = seed0 + s
        print("  Sim $s/$nsims (seed=$seed) ... ")
        flush(stdout)

        (; ϕ, ds) = load_sim(; seed=seed, load_kwargs...)

        ϕqe    = quadratic_estimate(ds; weights=:lensed, wiener_filtered=false).ϕqe
        ϕqe_wf = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true).ϕqe
        ϕgi    = gi_estimate(ds; Lgrad=GI_Lgrad, Lhp=GI_Lhp, Lmax=GI_Lmax)

        cl_tt   = get_Cℓ(ϕ;          Δℓ=Δℓ)
        cl_tq   = get_Cℓ(ϕ, ϕqe;    Δℓ=Δℓ)
        cl_tqwf = get_Cℓ(ϕ, ϕqe_wf; Δℓ=Δℓ)
        cl_tgi  = get_Cℓ(ϕ, ϕgi;    Δℓ=Δℓ)

        if ℓ_template === nothing
            ℓ_template = Float64.(collect(cl_tt.ℓ))
        end

        # R_L = C(ϕ_true, ϕ̂) / C(ϕ_true, ϕ_true)
        denom = cl_tt.Cℓ
        push!(R_qe_sims,    cl_tq.Cℓ   ./ denom)
        push!(R_qe_wf_sims, cl_tqwf.Cℓ ./ denom)
        push!(R_gi_sims,    cl_tgi.Cℓ  ./ denom)

        nsims_completed += 1
        push!(seeds_done, seed)

        W_qe    = mean(reduce(hcat, R_qe_sims);    dims=2)[:]
        W_qe_wf = mean(reduce(hcat, R_qe_wf_sims); dims=2)[:]
        W_gi    = mean(reduce(hcat, R_gi_sims);    dims=2)[:]

        # Store raw phi maps so plot script can debias at the phi level
        if s ∉ phi_sims_done
            jldopen(phi_maps_file, "a+") do f
                f["sim_$s/ϕ_true"]  = Float64.(Map(ϕ).arr)
                f["sim_$s/ϕ_qe"]    = Float64.(Map(ϕqe).arr)
                f["sim_$s/ϕ_qe_wf"] = Float64.(Map(ϕqe_wf).arr)
                f["sim_$s/ϕ_gi"]    = Float64.(Map(ϕgi).arr)
                f["sim_$s/seed"]    = seed
            end
            push!(phi_sims_done, s)
        end

        @save WL_file R_qe_sims R_qe_wf_sims R_gi_sims ℓ_template W_qe W_qe_wf W_gi nsims_completed seeds_done
        println("done")
        flush(stdout)
    end

    # ── Diagnostics ───────────────────────────────────────────────────────────
    W_qe    = mean(reduce(hcat, R_qe_sims);    dims=2)[:]
    W_qe_wf = mean(reduce(hcat, R_qe_wf_sims); dims=2)[:]
    W_gi    = mean(reduce(hcat, R_gi_sims);    dims=2)[:]

    println("\n=== W_L diagnostics (μKarcminT=$μKarcminT, Lmax=$Lmax, beamFWHM=$(beamFWHM)arcmin) ===")
    println("  ℓ        W_QE_raw   W_QE_WF    W_GI")
    for ℓ_check in [1000, 3000, 5000, 7000, 9000, 11000]
        idx = argmin(abs.(ℓ_template .- ℓ_check))
        @printf "  ℓ≈%5d   %8.4f   %8.4f   %8.4f\n" round(Int, ℓ_template[idx]) W_qe[idx] W_qe_wf[idx] W_gi[idx]
    end
    println("Done! $nsims_completed sims.  Output: $WL_file, $phi_maps_file")
end

# S4-like: 1 µK-arcmin, beam=1.0 arcmin, Lmax=12000
run_noise_level(1.0, "",    12000, 1.0)

# Ultra-low: 0.1 µK-arcmin, beam=0.3 arcmin, Lmax=20000
run_noise_level(0.1, "_ul", 20000, 0.3)

println("\nAll done.")
