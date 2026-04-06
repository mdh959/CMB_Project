#!/usr/bin/env julia

# run_qe_gi_wl12k.jl
#
# For each sim: compute QE (unlensed weights, Wiener-filtered), GI (Lhp=4000),
# GI (pixel-exact corrected). Stores phi maps in phi_maps_qe_gi_*.jld2.
#
# W_L is estimated as mean-of-per-sim-ratios (paper eq. 3.1):
#   W_L = mean_sims[ C(ϕ_true, ϕ̂) / C(ϕ_true, ϕ_true) ]
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
const nsims    = 100
const seed0    = 1000
const Δℓ       = 30

const nsims_map  = 800    # MAP sims 
const MAPJ_STEPS = 20

const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
println("f_sky_patch = $(round(f_sky_patch; sigdigits=4))")

function run_noise_level(μKarcminT::Float64, suffix::String, Lmax::Int, beamFWHM::Float64;
                         Δℓ_wl::Int=Δℓ, run_map::Bool=true, run_gi_c::Bool=false,
                         run_rdn0::Bool=false,
                         θpix_sim::Float64=θpix, Nside_sim::Int=Nside,
                         map_Lmax::Int=Lmax)
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
    sum_R_qe_wf  = nothing
    sum_R_gi_b   = nothing
    sum_R_gi_c   = nothing
    W_qe_wf      = nothing
    W_gi_b       = nothing
    W_gi_c       = nothing
    ℓ_template   = nothing
    nsims_completed = 0
    seeds_done      = Int[]

    if isfile(WL_file)
        d = JLD2.load(WL_file)
        haskey(d, "sum_R_qe_wf") && (sum_R_qe_wf = Float64.(d["sum_R_qe_wf"]))
        haskey(d, "sum_R_gi_b")  && (sum_R_gi_b  = Float64.(d["sum_R_gi_b"]))
        haskey(d, "sum_R_gi_c")  && run_gi_c && d["sum_R_gi_c"] !== nothing && (sum_R_gi_c = Float64.(d["sum_R_gi_c"]))
        haskey(d, "ℓ_template")      && (ℓ_template      = Float64.(d["ℓ_template"]))
        haskey(d, "W_qe_wf")         && (W_qe_wf         = Float64.(d["W_qe_wf"]))
        haskey(d, "W_gi_b")          && (W_gi_b          = Float64.(d["W_gi_b"]))
        haskey(d, "W_gi_c") && run_gi_c && d["W_gi_c"] !== nothing && (W_gi_c       = Float64.(d["W_gi_c"]))
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

        ϕqe_wf = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true).ϕqe
        ϕgi_b  = gi_estimate(ds; Lhp=4000, Lmax=Lmax)  # GI estimator (Lhp=4000), O(N log N)
        # gi_corrected: O(N_modes × FFT) ≈ ~35s/sim — only if run_gi_c=true
        ϕgi_c  = run_gi_c ? gi_estimate_corrected(ds) : nothing

        # Force evaluation of ϕqe_wf and ϕgi_b NOW, before any ds.Cf mutation.
        # quadratic_estimate (wiener_filtered=true) is lazy — it captures a reference
        # to ds and re-evaluates on first use.  The RDN0 block below temporarily
        # swaps ds.Cf; if that mutates a cached normalisation inside ds, the lazy
        # ϕqe_wf would be corrupted.  Computing W_L spectra here pins the results.
        cl_tt   = get_Cℓ(ϕ;          Δℓ=Δℓ_wl)
        cl_tqwf = get_Cℓ(ϕ, ϕqe_wf; Δℓ=Δℓ_wl)
        cl_tgib = get_Cℓ(ϕ, ϕgi_b;  Δℓ=Δℓ_wl)
        cl_tgic = run_gi_c && ϕgi_c !== nothing ? get_Cℓ(ϕ, ϕgi_c; Δℓ=Δℓ_wl) : nothing

        # RDN0-like QE variant: replace ΣTtot = TF² * Cf̃ + Cn by the observed
        # total TT power from the data, while keeping Cf (unlensed QE weights)
        # unchanged. Temporarily swap ds.Cf̃ only, run QE, then restore.
        ϕqe_rdn0 = if run_rdn0
            d_map  = Map(ds.d)
            proj   = d_map.proj
            Ny, Nx = size(d_map.arr)
            NyF    = Ny ÷ 2 + 1
            cl_obs = get_Cℓ(d_map; Δℓ=30)
            ℓ_obs  = Float64.(cl_obs.ℓ)
            C_obs  = Float64.(cl_obs.Cℓ)
            ℓmag   = [sqrt(proj.ℓx[i]^2 + proj.ℓy[j]^2) for j in 1:NyF, i in 1:Nx]
            CT_2D  = [let ℓ = ℓmag[j,i]; k = searchsortedfirst(ℓ_obs, ℓ)
                          k <= 1 ? C_obs[1] : k > length(ℓ_obs) ? C_obs[end] :
                              (t = (ℓ-ℓ_obs[k-1])/(ℓ_obs[k]-ℓ_obs[k-1]);
                               (1-t)*C_obs[k-1] + t*C_obs[k])
                      end for j in 1:NyF, i in 1:Nx]

            # CT_2D ≈ B²·C_signal + C_noise (observed data spectrum).
            # QE source uses:
            #   ΣTtot = TF² * Cf̃ + Cn   (uses Cf̃, the lensed covariance)
            #   CT    = Cf               (unlensed, for weights=:unlensed)
            # We want only ΣTtot to match C_ℓ^TT,obs, so set
            #   Cf̃_new = (C_obs - Cn) / B²
            # and leave Cf unchanged.
            f_ones  = FlatFourier(ones(ComplexF64, NyF, Nx), proj)
            B_arr   = real.(ds.B̂.diag.arr)[1:NyF, 1:Nx]
            Cn_arr  = real.((ds.Cn̂ * f_ones).arr)[1:NyF, 1:Nx]
            CT_signal = @. max(1e-30, (CT_2D - Cn_arr) / max(B_arr^2, 1e-30))
            CT_obs    = Diagonal(FlatFourier(ComplexF64.(CT_signal), proj))

            Cf̃_saved = ds.Cf̃
            local r
            try
                ds.Cf̃ = CT_obs
                r = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=false).ϕqe
                _ = Map(r)   # force evaluation now
            finally
                ds.Cf̃ = Cf̃_saved
            end
            r
        else
            nothing
        end

        if ℓ_template === nothing
            ℓ_template = Float64.(collect(cl_tt.ℓ))
            nb = length(ℓ_template)
            sum_R_qe_wf = zeros(Float64, nb)
            sum_R_gi_b  = zeros(Float64, nb)
            sum_R_gi_c  = zeros(Float64, nb)
        end

        # Accumulate per-sim ratios — W_L[ℓ] = mean_i( C_cross_i[ℓ] / C_true_i[ℓ] )
        ctt = Float64.(cl_tt.Cℓ)
        sum_R_qe_wf .+= safe_div(Float64.(cl_tqwf.Cℓ), ctt)
        sum_R_gi_b  .+= safe_div(Float64.(cl_tgib.Cℓ),  ctt)
        if run_gi_c && cl_tgic !== nothing
            sum_R_gi_c .+= safe_div(Float64.(cl_tgic.Cℓ), ctt)
        end

        nsims_completed += 1
        push!(seeds_done, seed)

        W_qe_wf = sum_R_qe_wf ./ nsims_completed
        W_gi_b  = sum_R_gi_b  ./ nsims_completed
        W_gi_c  = run_gi_c ? sum_R_gi_c ./ nsims_completed : nothing

        # Store raw phi maps so plot script can debias at the phi level
        jldopen(phi_maps_file, "a+") do f
            if s ∉ phi_sims_done
                f["sim_$s/ϕ_true"]  = Float64.(Map(ϕ).arr)
                f["sim_$s/ϕ_qe_wf"] = Float64.(Map(ϕqe_wf).arr)
                f["sim_$s/ϕ_gi_b"]  = Float64.(Map(ϕgi_b).arr)
                f["sim_$s/seed"]    = seed
                push!(phi_sims_done, s)
            end
            if run_gi_c && ϕgi_c !== nothing && !haskey(f, "sim_$s/ϕ_gi_c")
                f["sim_$s/ϕ_gi_c"] = Float64.(Map(ϕgi_c).arr)
            end
            if run_rdn0 && ϕqe_rdn0 !== nothing && !haskey(f, "sim_$s/ϕ_qe_rdn0")
                f["sim_$s/ϕ_qe_rdn0"] = Float64.(Map(ϕqe_rdn0).arr)
            end
        end

        @save WL_file sum_R_qe_wf sum_R_gi_b sum_R_gi_c ℓ_template W_qe_wf W_gi_b W_gi_c nsims_completed seeds_done
        println("done")
        flush(stdout)
    end

    # ── Diagnostics ───────────────────────────────────────────────────────────
    # W_L is already up-to-date from the loop (ratio of accumulated sums)

    println("\n=== W_L diagnostics (μKarcminT=$μKarcminT, Lmax=$Lmax, beamFWHM=$(beamFWHM)arcmin) ===")
    hdr = "  ℓ        W_QE_WF    W_GI_B" * (W_gi_c !== nothing ? "    W_GI_C" : "")
    println(hdr)
    for ℓ_check in [1000, 3000, 5000, 7000, 9000, 11000]
        idx = argmin(abs.(ℓ_template .- ℓ_check))
        line = @sprintf "  ℓ≈%5d   %8.4f   %8.4f" round(Int, ℓ_template[idx]) W_qe_wf[idx] W_gi_b[idx]
        W_gi_c !== nothing && (line *= @sprintf "   %8.4f" W_gi_c[idx])
        println(line)
    end
    println("Done! $nsims_completed sims.  Output: $WL_file, $phi_maps_file")

    # ── W_L sanity check ─────────────────────────────────────────────────────
    # QE W_L should be in [0,1] and increase with lower noise.
    # GI W_L should be ~0.5-1 for L>4000.  Values outside [-0.1, 1.5] or large
    # std suggest too-fine Δℓ_wl for the number of sims (→ increase Δℓ_wl).
    for (name, W) in [("W_qe_wf", W_qe_wf), ("W_gi_b", W_gi_b)]
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

    R_mj_sims = Vector{Vector{Float64}}()
    W_mj = nothing
    nsims_map_done = 0; map_seeds_done = Int[]
    ℓ_template_map = nothing

    if isfile(MAP_WL_file)
        d = JLD2.load(MAP_WL_file)
        haskey(d, "R_mj_sims")      && (R_mj_sims      = d["R_mj_sims"])
        haskey(d, "nsims_map_done") && (nsims_map_done = d["nsims_map_done"])
        haskey(d, "map_seeds_done") && (map_seeds_done = d["map_seeds_done"])
        haskey(d, "W_mj")           && (W_mj           = d["W_mj"])
        haskey(d, "ℓ_template")     && (ℓ_template_map = Float64.(d["ℓ_template"]))
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

        # Warm start: Wiener-filtered QE (safe initial ϕ for MAP_joint gradient)
        ϕqe_ws  = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true).ϕqe
        ϕ_start = ϕqe_ws

        # MAP_joint — warm start
        ϕ_mj = nothing
        try
            result_mj = MAP_joint(ds, FieldTuple(ϕ=ϕ_start);
                nsteps=MAPJ_STEPS,
                conjgrad_kwargs=(tol=1e-3, nsteps=200), progress=false,
                history_keys=(:logpdf,))
            ϕ_mj = result_mj.ϕ
            @printf "(MJ %.1f→%.1f) " result_mj.history[1].logpdf result_mj.history[end].logpdf
        catch err
            print("MAP_joint failed($err) ")
        end

        # W_L for MAP (kept at fine Δℓ=30 — MAP W_L is already smooth)
        cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)
        ℓ_template_map === nothing && (ℓ_template_map = Float64.(collect(cl_tt.ℓ)))
        denom = cl_tt.Cℓ
        ϕ_mj !== nothing && push!(R_mj_sims, get_Cℓ(ϕ, ϕ_mj; Δℓ=Δℓ).Cℓ ./ denom)

        # Store phi maps
        if s ∉ map_phi_done
            jldopen(MAP_phi_file, "a+") do f
                f["sim_$s/ϕ_true"]   = Float64.(Map(ϕ).arr)
                f["sim_$s/ϕ_qe_wf"]  = Float64.(Map(ϕqe_ws).arr)
                ϕ_mj !== nothing && (f["sim_$s/ϕ_mj"] = Float64.(Map(ϕ_mj).arr))
                f["sim_$s/seed"]     = seed
            end
            push!(map_phi_done, s)
        end

        nsims_map_done += 1
        push!(map_seeds_done, seed)
        W_mj = !isempty(R_mj_sims) ? mean(reduce(hcat, R_mj_sims); dims=2)[:] : nothing
        jldsave(MAP_WL_file; R_mj_sims, W_mj, nsims_map_done, map_seeds_done,
                ℓ_template=ℓ_template_map)
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

end

# S4-like: standard QE (unlensed weights, WF) + GI Boryana
run_noise_level(1.0, "",    12000, 1.0; run_map=true, run_gi_c=true)

# Ultra-low noise: same pixel grid as S4 (θpix=0.744, Lmax=12000) for direct comparison.
# RDN0: QE rerun with C_ℓ^TT replaced by observed signal spectrum (noise-subtracted, debeamed).
run_noise_level(0.1, "_ul", 20000, 0.3; Δℓ_wl=150, run_rdn0=true, run_map=false)

# Ultra-low noise — MAP joint runs with Lmax=6000 bandpass
# run_noise_level(0.1, "_ul", 20000, 0.3; θpix_sim=0.5, Δℓ_wl=150, run_rdn0=true, run_map=false, map_Lmax=6000)

println("\nAll done.")
