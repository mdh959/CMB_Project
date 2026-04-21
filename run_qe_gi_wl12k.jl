#!/usr/bin/env julia

# run_qe_gi_wl12k.jl
#
# For each sim: compute QE (unlensed weights, raw/normalised), GI (Lhp=4000),
# and per-sim realization-dependent N0 (RDN0) via N0_bias (CMBLensing.jl fork).
# Stores phi maps in phi_maps_qe_gi_*.jld2:
#   ϕ_true, ϕ_qe_raw, ϕ_gi_b, N0_rdn0 (band powers at Δℓ_wl)
#
# W_L (transfer function) stored in WL_qe_gi_*.jld2:
#   W_L[ℓ] = mean_sims[ C(ϕ_true, ϕ̂)[ℓ] / C(ϕ_true, ϕ_true)[ℓ] ]
# QE: W_L ≈ 1 (normalised by A_L internally).  GI: W_L < 1, encodes partial recovery.
#
# MAP joint stored in WL_map_*.jld2 and phi_maps_map_*.jld2:
#   S4  (1 µK-arcmin): warm start = QE WF,  αmax=0.3, 30 steps
#   UL (0.1 µK-arcmin): zero start,          αmax=0.05, 30 steps

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using LinearAlgebra
using Statistics: mean
using JLD2
using Printf

include("utils.jl")
using .Utils

# ── Preconditioner patch: use lensed Cf̃ instead of unlensed Cf ───────────────
# Cf̃ = L(ϕ)CfL(ϕ)† is the actual covariance in the f-posterior Hessian.
# Using it reduces CG iterations (especially at low noise) without changing
# the MAP solution — preconditioner only affects convergence speed.
@eval CMBLensing begin
    function Hessian_logpdf_preconditioner(::Val{:f}, ds::DataSet)
        @unpack Cf̃, B̂, M̂, Cn̂ = ds
        v   = copy(diag(pinv(Cf̃) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂))
        arr = v.arr
        good = isfinite.(arr) .& (arr .> 0)
        fill!(view(arr, .!good), any(good) ? mean(arr[good]) : 1.0)
        arr .= clamp.(arr, 1e-30, 1e30)
        return Diagonal(v)
    end
end


const Cℓ       = camb(r=0.05, ℓmax=35000)
const θpix     = 0.7438046267475303
const Nside    = 512
const pol      = :I
const nsims    = 1000
const seed0    = 1000
const Δℓ       = 30

const nsims_map  = 400    # MAP sims 
const MAPJ_STEPS = 40

const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
println("f_sky_patch = $(round(f_sky_patch; sigdigits=4))")

function run_noise_level(μKarcminT::Float64, suffix::String, Lmax::Int, beamFWHM::Float64;
                         Δℓ_wl::Int=Δℓ, run_map::Bool=true,
                         run_rdn0::Bool=false,
                         θpix_sim::Float64=θpix, Nside_sim::Int=Nside,
                         map_Lmax::Int=Lmax,
                         map_αmax::Float64=0.3,
                         map_zero_start::Bool=false,
                         map_warmstart_Lmax::Int=0)          # LP-filter QE WF warm start (0=full); ignored when map_zero_start=true
    println("\n" * "="^70)
    println("=== Noise: μKarcminT=$μKarcminT, Lmax=$Lmax, beamFWHM=$(beamFWHM)arcmin  (suffix=\"$suffix\") ===")
    println("    θpix=$(θpix_sim) arcmin, Nside=$(Nside_sim),  ℓ_Nyquist≈$(round(Int, π/(θpix_sim*π/(180*60))))")
    println("="^70)

    Cℓn      = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    bandpass = LowPass(Lmax)

    WL_file       = "results/WL_qe_gi_12000$(suffix).jld2"
    phi_maps_file = "results/phi_maps_qe_gi_12000$(suffix).jld2"

    load_kwargs = (
        Cℓ=Cℓ, Cℓn=Cℓn, θpix=θpix_sim, T=Float64, Nside=Nside_sim,
        beamFWHM=beamFWHM, pol=pol, bandpass_mask=bandpass,
        pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
    )

    # ── Resume state ─────────────────────────────────────────────────────────
    # W_L estimated as mean of per-sim ratios:
    #   W_L[ℓ] = (1/N) Σ_i  C(ϕ_true_i, ϕ̂_i)[ℓ] / C(ϕ_true_i, ϕ_true_i)[ℓ]
    # Accumulated as running sums of the per-sim ratio; divide by nsims_completed to get W_L.
    safe_div(a, b) = @. ifelse(abs(b) > 0.0, a / b, 0.0)
    sum_R_qe_raw = nothing
    sum_R_gi_b   = nothing
    W_qe_raw     = nothing
    W_gi_b       = nothing
    ℓ_template   = nothing
    nsims_completed = 0
    seeds_done      = Int[]

    if isfile(WL_file)
        d = JLD2.load(WL_file)
        haskey(d, "sum_R_qe_raw") && (sum_R_qe_raw = Float64.(d["sum_R_qe_raw"]))
        # backward compat: old checkpoints stored QE WF under "sum_R_qe_wf"
        sum_R_qe_raw === nothing && haskey(d, "sum_R_qe_wf") && (sum_R_qe_raw = Float64.(d["sum_R_qe_wf"]))
        haskey(d, "sum_R_gi_b")   && (sum_R_gi_b   = Float64.(d["sum_R_gi_b"]))
        haskey(d, "ℓ_template")   && (ℓ_template   = Float64.(d["ℓ_template"]))
        haskey(d, "W_qe_raw")     && (W_qe_raw      = Float64.(d["W_qe_raw"]))
        # backward compat: old checkpoints stored QE WF under "W_qe_wf"
        W_qe_raw === nothing && haskey(d, "W_qe_wf") && (W_qe_raw = Float64.(d["W_qe_wf"]))
        haskey(d, "W_gi_b")       && (W_gi_b        = Float64.(d["W_gi_b"]))
        haskey(d, "nsims_completed") && (nsims_completed = d["nsims_completed"])
        haskey(d, "seeds_done")      && (seeds_done      = d["seeds_done"])
        println("Resumed W_L checkpoint: $nsims_completed / $nsims sims")
    end

    phi_sims_done  = Set{Int}()
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

        ϕqe_raw = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=false).ϕqe
        ϕgi_b   = gi_estimate(ds; Lhp=4000, Lmax=Lmax)  # GI estimator (Lhp=4000), O(N log N)

        cl_tt    = get_Cℓ(ϕ;           Δℓ=Δℓ_wl)
        cl_tqraw = get_Cℓ(ϕ, ϕqe_raw; Δℓ=Δℓ_wl)
        cl_tgib  = get_Cℓ(ϕ, ϕgi_b;   Δℓ=Δℓ_wl)

        # Realization-dependent N0 bias (Louis's fork: N0_bias with fixed cov_to_Cℓ normalization)
        N0_rdn0_Cℓ = run_rdn0 ? Float64.(cov_to_Cℓ(N0_bias(ds; weights=:unlensed, realization_spec=:data).N0; Δℓ=Δℓ_wl).Cℓ) : nothing

        if ℓ_template === nothing
            ℓ_template = Float64.(collect(cl_tt.ℓ))
            nb = length(ℓ_template)
            sum_R_qe_raw = zeros(Float64, nb)
            sum_R_gi_b   = zeros(Float64, nb)
        end

        # Accumulate per-sim ratios — W_L[ℓ] = mean_i( C_cross_i[ℓ] / C_true_i[ℓ] )
        ctt = Float64.(cl_tt.Cℓ)
        sum_R_qe_raw .+= safe_div(Float64.(cl_tqraw.Cℓ), ctt)
        sum_R_gi_b   .+= safe_div(Float64.(cl_tgib.Cℓ),  ctt)

        nsims_completed += 1
        push!(seeds_done, seed)

        W_qe_raw = sum_R_qe_raw ./ nsims_completed
        W_gi_b   = sum_R_gi_b   ./ nsims_completed

        # Store phi maps (raw QE + RDN0 per sim; no WF stored)
        jldopen(phi_maps_file, "a+") do f
            if s ∉ phi_sims_done
                f["sim_$s/ϕ_true"]   = Float64.(Map(ϕ).arr)
                f["sim_$s/ϕ_qe_raw"] = Float64.(Map(ϕqe_raw).arr)
                f["sim_$s/ϕ_gi_b"]   = Float64.(Map(ϕgi_b).arr)
                f["sim_$s/seed"]     = seed
                push!(phi_sims_done, s)
            end
            if run_rdn0 && N0_rdn0_Cℓ !== nothing && !haskey(f, "sim_$s/N0_rdn0")
                f["sim_$s/N0_rdn0"] = N0_rdn0_Cℓ
            end
        end

        @save WL_file sum_R_qe_raw sum_R_gi_b ℓ_template W_qe_raw W_gi_b nsims_completed seeds_done
        println("done")
        flush(stdout)
    end

    # ── Diagnostics ───────────────────────────────────────────────────────────
    # W_L is already up-to-date from the loop (ratio of accumulated sums)

    println("\n=== W_L diagnostics (μKarcminT=$μKarcminT, Lmax=$Lmax, beamFWHM=$(beamFWHM)arcmin) ===")
    println("  ℓ        W_QE_RAW   W_GI_B")
    for ℓ_check in [1000, 3000, 5000, 7000, 9000, 11000]
        idx = argmin(abs.(ℓ_template .- ℓ_check))
        @printf "  ℓ≈%5d   %8.4f   %8.4f\n" round(Int, ℓ_template[idx]) W_qe_raw[idx] W_gi_b[idx]
    end
    println("Done! $nsims_completed sims.  Output: $WL_file, $phi_maps_file")

    # ── W_L sanity check ─────────────────────────────────────────────────────
    # QE raw W_L (cross with truth / auto truth) ≈ 1 for unlensed weights.
    # GI W_L should be ~0.5-1 for L>4000.  Values outside [-0.1, 1.5] or large
    # std suggest too-fine Δℓ_wl for the number of sims (→ increase Δℓ_wl).
    for (name, W) in [("W_qe_raw", W_qe_raw), ("W_gi_b", W_gi_b)]
        W === nothing && continue
        msk = @. isfinite(W) & (ℓ_template >= 5000) & (ℓ_template <= 11000)
        !any(msk) && continue
        wv = W[msk]
        wmin, wmax = minimum(wv), maximum(wv)
        bad = wmin < -0.1 || wmax > 1.5
        @printf "  SANITY %-10s L=5000-11000: min=%.3f  max=%.3f  %s\n" name wmin wmax (bad ? "⚠ SUSPICIOUS" : "OK")
    end

    # ── MAP section ───────────────────────────────────────────────────────────
    # Runs nsims_map sims.  Stores QE_WF and MAP_joint phi maps in MAP_phi_file.
    MAP_WL_file  = "results/WL_map_12000$(suffix).jld2"
    MAP_phi_file = "results/phi_maps_map_12000$(suffix).jld2"

    if !run_map
        println("MAP section skipped (run_map=false).")
        return
    end

    map_load_kwargs = if map_Lmax == Lmax
        load_kwargs
    else
        println("MAP using bandpass Lmax=$map_Lmax (QE/GI used Lmax=$Lmax)")
        map_Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=map_Lmax)
        (load_kwargs..., Cℓn=map_Cℓn, bandpass_mask=LowPass(map_Lmax))
    end

    R_mj_sims        = Vector{Vector{Float64}}()
    logpdf_histories = Vector{Vector{Float64}}()
    W_mj = nothing
    nsims_map_done = 0; map_seeds_done = Int[]
    ℓ_template_map = nothing

    if isfile(MAP_WL_file)
        d = JLD2.load(MAP_WL_file)
        haskey(d, "R_mj_sims")        && (R_mj_sims        = d["R_mj_sims"])
        haskey(d, "logpdf_histories")  && (logpdf_histories  = d["logpdf_histories"])
        haskey(d, "nsims_map_done")    && (nsims_map_done    = d["nsims_map_done"])
        haskey(d, "map_seeds_done")    && (map_seeds_done    = d["map_seeds_done"])
        haskey(d, "W_mj")              && (W_mj              = d["W_mj"])
        haskey(d, "ℓ_template")        && (ℓ_template_map    = Float64.(d["ℓ_template"]))
        println("Resumed MAP checkpoint: $nsims_map_done / $nsims_map sims")
    end

    map_phi_done = Set{Int}()
    if isfile(MAP_phi_file)
        try
            jldopen(MAP_phi_file, "r") do f
                for key in keys(f)
                    m = match(r"^sim_(\d+)$", key)
                    m !== nothing && push!(map_phi_done, parse(Int, m.captures[1]))
                end
            end
            println("$(length(map_phi_done)) MAP phi maps in $MAP_phi_file")
        catch err
            @warn "MAP phi file corrupted ($err) — deleting"; rm(MAP_phi_file)
        end
    end

    for s in (nsims_map_done + 1):nsims_map
        seed = seed0 + s
        print("  MAP Sim $s/$nsims_map (seed=$seed) ... ")
        flush(stdout)

        (; ϕ, ds) = load_sim(; seed=seed, map_load_kwargs...)

        # Warm start: QE WF, optionally low-pass filtered to avoid LenseFlow NaN
        # at UL noise where high-L modes of QE WF have large amplitude.
        # map_warmstart_Lmax=0 → full QE WF; >0 → LP filtered to that scale.
        ϕqe_ws  = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true).ϕqe
        ϕ_ws    = map_warmstart_Lmax > 0 ? LowPass(map_warmstart_Lmax) * ϕqe_ws : ϕqe_ws
        ϕ_start = map_zero_start ? 0 * ϕqe_ws : ϕ_ws

        # MAP_joint
        ϕ_mj = nothing
        try
            result_mj = MAP_joint(ds, FieldTuple(ϕ=ϕ_start);
                nsteps=MAPJ_STEPS,
                αmax=map_αmax,
                conjgrad_kwargs=(tol=1e-3, nsteps=200), progress=false,
                history_keys=(:total_logpdf,))
            ϕ_mj = result_mj.ϕ
            lp0 = result_mj.history[1].total_logpdf
            lpN = result_mj.history[end].total_logpdf
            @printf "(MJ %.1f→%.1f, Δ=%.2f) " lp0 lpN (lpN - lp0)
            push!(logpdf_histories, Float64.([h.total_logpdf for h in result_mj.history]))
        catch err
            print("MAP_joint failed($err) ")
        end

        # W_L for MAP (fine Δℓ=30 — MAP W_L is already smooth)
        cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)
        ℓ_template_map === nothing && (ℓ_template_map = Float64.(collect(cl_tt.ℓ)))
        if ϕ_mj !== nothing
            ctt = Float64.(cl_tt.Cℓ)
            push!(R_mj_sims, safe_div(Float64.(get_Cℓ(ϕ, ϕ_mj; Δℓ=Δℓ).Cℓ), ctt))
        end

        # Store phi maps
        if s ∉ map_phi_done
            jldopen(MAP_phi_file, "a+") do f
                f["sim_$s/ϕ_true"] = Float64.(Map(ϕ).arr)
                ϕ_mj !== nothing && (f["sim_$s/ϕ_mj"] = Float64.(Map(ϕ_mj).arr))
                f["sim_$s/seed"]   = seed
            end
            push!(map_phi_done, s)
        end

        nsims_map_done += 1
        push!(map_seeds_done, seed)
        W_mj = !isempty(R_mj_sims) ? mean(reduce(hcat, R_mj_sims); dims=2)[:] : nothing
        jldsave(MAP_WL_file; R_mj_sims, logpdf_histories, W_mj, nsims_map_done,
                map_seeds_done, ℓ_template=ℓ_template_map)
        println("done")
        flush(stdout)
    end

    if !isempty(R_mj_sims)
        println("\n=== MAP W_L diagnostics (μKarcminT=$μKarcminT) ===")
        println("  ℓ        W_MJ")
        for ℓ_check in [3000, 5000, 7000, 9000, 11000]
            idx = argmin(abs.(ℓ_template_map .- ℓ_check))
            W_mj_v = W_mj !== nothing ? W_mj[idx] : NaN
            @printf "  ℓ≈%5d   %8.4f\n" round(Int, ℓ_template_map[idx]) W_mj_v
        end
        println("MAP done! $nsims_map_done/$nsims_map sims. Output: $MAP_WL_file, $MAP_phi_file")
    end

    # ── Convergence check: mean log-posterior trajectory ─────────────────────
    if length(logpdf_histories) >= 2
        nsteps_done = minimum(length.(logpdf_histories))
        mean_lp = [mean(h[t] for h in logpdf_histories) for t in 1:nsteps_done]
        total_Δ  = mean_lp[end] - mean_lp[1]
        last5_Δ  = nsteps_done >= 5 ? mean_lp[end] - mean_lp[end-4] : NaN
        per_step = total_Δ / (nsteps_done - 1)
        println("\n  MAP convergence (mean logpdf over $(length(logpdf_histories)) sims):")
        @printf "    Step 1:  %.2f\n" mean_lp[1]
        @printf "    Step %d: %.2f  (total Δ = %.2f,  %.3f/step avg)\n" nsteps_done mean_lp[end] total_Δ per_step
        isfinite(last5_Δ) && @printf "    Last 5 steps Δ = %.4f\n" last5_Δ
        if isfinite(last5_Δ) && abs(per_step) > 0
            frac = abs(last5_Δ / 5) / abs(per_step)
            println(frac > 0.05 ?
                "    ⚠ STILL CONVERGING (last-step rate = $(round(100*frac;digits=1))% of mean — consider more steps)" :
                "    ✓ CONVERGED (last-step rate < 5% of mean)")
        end
    end

end

# S4-like (1 µK-arcmin): MAP warm-starts from QE WF (stable at S4 noise levels).
run_noise_level(1.0, "",    12000, 1.0; run_rdn0=true, run_map=true)

# Ultra-low noise (0.1 µK-arcmin): zero MAP start (QE WF causes LenseFlow NaN at
# this noise level due to large high-L amplitudes).  αmax=0.1 is safe from zero
# start since early ϕ amplitudes are small.  Δℓ_wl=150 for smoother W_L.
run_noise_level(0.1, "_ul", 12000, 0.3; Δℓ_wl=150, run_rdn0=true, run_map=true,
                map_αmax=0.1, map_zero_start=true)


println("\nAll done.")
