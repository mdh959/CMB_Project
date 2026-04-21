#!/usr/bin/env julia
"""
plot_qe_gi_sigma_12k.jl  —  Boryana+2019 matching figures

Estimators (all: debias_phi_with_WL first, then get_Cℓ):
  QE  : raw phi → W_L debiased → power spectra
  GI  : raw phi → W_L debiased → power spectra
  MAP : raw phi → W_L debiased → power spectra (when available)

W_L smoothed with running mean before Fourier debiasing.
Output: results/Boryana's paper/   ΔL=2000, proc_edges 4000:2000:12000
"""

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using Statistics: mean, std
using PythonPlot
using Printf
using JLD2

include("utils.jl")
using .Utils

# ── Constants ─────────────────────────────────────────────────────────────────
const Cℓ_theory   = camb(r=0.05, ℓmax=170000)
const θpix        = 0.7438046267475303
const Nside       = 512
const pol         = :I
const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
const f_sky_paper = 0.4
const minW        = 1e-8
const ΔL          = 2000
const Δℓ_spec     = 30
const proc_edges  = collect(3000.0:ΔL:13001.0)   # edges at [3000,5000,7000,9000,11000,13000], centres at [4000,6000,8000,10000,12000]
const OUT_DIR     = "results/Boryana's paper"

mkpath(OUT_DIR)
println("f_sky_patch = $(round(f_sky_patch; sigdigits=4))")
println("proc_edges  = $(proc_edges)")

# ── Helpers ───────────────────────────────────────────────────────────────────
function smooth_wl(W::Vector{Float64}; window::Int=9)
    n = length(W); Ws = similar(W); hw = window ÷ 2
    for i in 1:n
        vals = filter(isfinite, W[max(1,i-hw):min(n,i+hw)])
        Ws[i] = isempty(vals) ? 0.0 : mean(vals)
    end
    return Ws
end

function coarsen(ℓ::Vector, M::Matrix; edges::Vector)
    nb = length(edges) - 1
    Lc = fill(NaN, nb); μv = fill(NaN, nb); σv = fill(NaN, nb)
    for b in 1:nb
        idx = findall(x -> edges[b] <= x < edges[b+1], ℓ)
        isempty(idx) && continue
        per = filter(isfinite, vec(mean(M[idx, :]; dims=1)))
        length(per) < 2 && continue
        Lc[b] = 0.5*(edges[b]+edges[b+1]); μv[b] = mean(per); σv[b] = std(per)
    end
    Lc, μv, σv
end

function bands_per_sim(ℓ::Vector, sims::Vector, edges::Vector)
    nb = length(edges) - 1; ns = length(sims)
    ns == 0 && return fill(NaN, nb, 0)
    M = fill(NaN, nb, ns)
    for (j, s) in enumerate(sims), b in 1:nb
        idx = findall(x -> edges[b] <= x < edges[b+1], ℓ)
        isempty(idx) && continue
        vals = filter(isfinite, s[idx])
        isempty(vals) || (M[b, j] = mean(vals))
    end
    M
end

function snr_compute(M_meas::Matrix, M_true::Matrix, Lc::Vector,
                     L_lo::Real, L_hi::Real, σ_sc::Real)
    size(M_meas, 2) < 2 && return NaN
    sel = findall(b -> !isnan(Lc[b]) && L_lo <= Lc[b] <= L_hi, 1:length(Lc))
    isempty(sel) && return NaN
    snr2 = 0.0; nb = 0
    for b in sel
        a_b = filter(isfinite, vec(M_meas[b,:])); t_b = filter(isfinite, vec(M_true[b,:]))
        (length(a_b) < 2 || length(t_b) < 2) && continue
        σ_b = σ_sc * std(a_b); σ_b <= 0 && continue
        snr2 += (mean(t_b) / σ_b)^2; nb += 1
    end
    nb == 0 && return NaN
    sqrt(max(0.0, snr2))
end

# ── Main processing ───────────────────────────────────────────────────────────
function process_noise_level(WL_file, phi_maps_file, label;
                             Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
                             map_wl_file=nothing, map_phi_file=nothing,
                             snr_L_lo=5000.0, snr_L_hi=11000.0,
                             xlim_plot=(4000.0, 12000.0),
                             θpix_sim::Float64=θpix, Nside_sim::Int=Nside)
    println("\n" * "="^70)
    println("=== $label ===")

    # W_L checkpoint (backward compat: W_qe_wf → W_qe_raw)
    d_wl       = JLD2.load(WL_file)
    ℓ_template = Float64.(d_wl["ℓ_template"])
    nsims_wl   = d_wl["nsims_completed"]
    _lw(d,k)   = (haskey(d,k) && d[k] !== nothing) ? Float64.(d[k]) : nothing

    W_qe_raw = _lw(d_wl, "W_qe_raw")
    W_qe_raw === nothing && (W_qe_raw = _lw(d_wl, "W_qe_wf"))
    W_gi_b   = _lw(d_wl, "W_gi_b")

    # smooth W_L
    W_qe_s = W_qe_raw !== nothing ? smooth_wl(W_qe_raw) : nothing
    W_gi_s = W_gi_b   !== nothing ? smooth_wl(W_gi_b)   : nothing
    WL_qe  = W_qe_s !== nothing ? Cℓs(ℓ_template, W_qe_s) : nothing
    WL_gi  = W_gi_s !== nothing ? Cℓs(ℓ_template, W_gi_s) : nothing
    has_qe = WL_qe !== nothing
    has_gi = WL_gi !== nothing
    println("  W_L from $nsims_wl sims  (QE raw: $has_qe, GI: $has_gi)")

    # MAP W_L
    has_map = map_wl_file !== nothing && isfile(map_wl_file) &&
              map_phi_file !== nothing && isfile(map_phi_file)
    W_mj_raw = WL_mj = nothing
    ℓ_template_map = ℓ_template
    nsims_map_done = 0
    if has_map
        d_map = JLD2.load(map_wl_file)
        ℓ_template_map = haskey(d_map, "ℓ_template") ?
            Float64.(d_map["ℓ_template"]) : ℓ_template
        W_mj_raw = haskey(d_map, "W_mj") ?
            Float64.(d_map["W_mj"]) : ones(length(ℓ_template_map))
        nsims_map_done = get(d_map, "nsims_map_done", 0)
        WL_mj  = Cℓs(ℓ_template_map, smooth_wl(W_mj_raw))
        println("  MAP W_L: $nsims_map_done sims")
    end

    # Projection metadata (one load_sim for ϕ_ref)
    Cℓn_meta  = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    meta_seed = jldopen(phi_maps_file, "r") do f
        first_sim = minimum(parse(Int, match(r"^sim_(\d+)$", k).captures[1])
                            for k in keys(f) if occursin(r"^sim_\d+$", k))
        read(f, "sim_$first_sim/seed")
    end
    (; ϕ) = load_sim(; seed=meta_seed, Cℓ=Cℓ_theory, Cℓn=Cℓn_meta, θpix=θpix_sim,
                       T=Float64, Nside=Nside_sim, beamFWHM=beamFWHM, pol=pol,
                       bandpass_mask=LowPass(Lmax),
                       pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))
    ϕ_ref = Map(ϕ)
    wrap(arr) = typeof(ϕ_ref)(Float64.(arr), ϕ_ref.metadata)

    # Per-sim arrays
    Cl_auto_qe_sims  = Vector{Vector{Float64}}()   # QE: debiased auto
    Cl_cross_qe_sims = Vector{Vector{Float64}}()   # QE: debiased cross
    Cl_auto_gi_sims  = Vector{Vector{Float64}}()   # GI: debiased auto
    Cl_cross_gi_sims = Vector{Vector{Float64}}()   # GI: debiased cross
    Cl_true_sims     = Vector{Vector{Float64}}()
    ρ_qe_sims        = Vector{Vector{Float64}}()
    ρ_gi_sims        = Vector{Vector{Float64}}()
    ℓ_kk             = nothing

    qegi_sims = Int[]
    jldopen(phi_maps_file, "r") do f
        for key in keys(f)
            m = match(r"^sim_(\d+)$", key)
            m !== nothing && push!(qegi_sims, parse(Int, m.captures[1]))
        end
        sort!(qegi_sims)
        for (i, s) in enumerate(qegi_sims)
            ϕt = wrap(read(f, "sim_$s/ϕ_true"))
            cl_tt = get_Cℓ(ϕt; Δℓ=Δℓ_spec)
            ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl_tt.ℓ)))
            kfac = @. (ℓ_kk^2 / 2)^2
            push!(Cl_true_sims, kfac .* Float64.(cl_tt.Cℓ))

            # QE: debias phi first, then compute power spectra (same pipeline as GI/MAP)
            if has_qe && haskey(f, "sim_$s/ϕ_qe_raw")
                ϕr     = wrap(read(f, "sim_$s/ϕ_qe_raw"))
                ϕr_deb = debias_phi_with_WL(ϕr, WL_qe; minW=minW)
                cl_aa  = get_Cℓ(ϕr_deb; Δℓ=Δℓ_spec)
                cl_xa  = get_Cℓ(ϕt, ϕr_deb; Δℓ=Δℓ_spec)
                Cl_aa_vals = Float64.(cl_aa.Cℓ)
                # Subtract per-sim RDN0, debiased by W_qe^2 (matches phi debiasing).
                # Sanitize N0 before interpolation: clamp ℓ to template range (same
                # as debias_phi_with_WL) and replace non-finite/negative N0 with 0
                # to prevent spline extrapolation artifacts creating outlier sims.
                if haskey(f, "sim_$s/N0_rdn0")
                    N0_stored  = Float64.(read(f, "sim_$s/N0_rdn0"))
                    N0_safe    = [isfinite(v) && v >= 0.0 ? v : 0.0 for v in N0_stored]
                    ℓkk_cl     = clamp.(ℓ_kk, ℓ_template[1], ℓ_template[end])
                    N0_at_ℓkk  = max.(Float64.(Cℓs(ℓ_template, N0_safe).(ℓkk_cl)), 0.0)
                    W_at_ℓkk   = Float64.(WL_qe.(ℓ_kk))
                    N0_deb     = @. N0_at_ℓkk / max(W_at_ℓkk^2, minW^2)
                    Cl_aa_vals = Cl_aa_vals .- N0_deb
                end
                push!(Cl_auto_qe_sims,  kfac .* Cl_aa_vals)
                push!(Cl_cross_qe_sims, kfac .* Float64.(cl_xa.Cℓ))
                denom = sqrt.(max.(Float64.(cl_tt.Cℓ) .* max.(Cl_aa_vals, 0.0), 0.0))
                push!(ρ_qe_sims, clamp.(Float64.(cl_xa.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
            end

            # GI: debias phi first, then compute power spectra
            if has_gi && haskey(f, "sim_$s/ϕ_gi_b")
                ϕg_raw = wrap(read(f, "sim_$s/ϕ_gi_b"))
                ϕg_deb = debias_phi_with_WL(ϕg_raw, WL_gi; minW=minW)
                cl_gi_a = get_Cℓ(ϕg_deb; Δℓ=Δℓ_spec)
                cl_gi_x = get_Cℓ(ϕt, ϕg_deb; Δℓ=Δℓ_spec)
                push!(Cl_auto_gi_sims,  kfac .* Float64.(cl_gi_a.Cℓ))
                push!(Cl_cross_gi_sims, kfac .* Float64.(cl_gi_x.Cℓ))
                denom = sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_gi_a.Cℓ), 0.0))
                push!(ρ_gi_sims, clamp.(Float64.(cl_gi_x.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
            end
            print("\r  QE/GI sim $i/$(length(qegi_sims))"); flush(stdout)
        end
    end
    println(); println("  $(length(qegi_sims)) QE/GI sims")

    # MAP sims
    Cl_auto_mj_sims  = Vector{Vector{Float64}}()   # MAP: debiased
    Cl_cross_mj_sims = Vector{Vector{Float64}}()   # MAP: debiased
    ρ_mj_sims        = Vector{Vector{Float64}}()
    if has_map && ℓ_kk !== nothing
        map_sims = Int[]
        jldopen(map_phi_file, "r") do f
            for key in keys(f)
                m = match(r"^sim_(\d+)$", key); m !== nothing && push!(map_sims, parse(Int, m.captures[1]))
            end
            sort!(map_sims)
            for (i, s) in enumerate(map_sims)
                haskey(f, "sim_$s/ϕ_mj") || continue
                ϕt   = wrap(read(f, "sim_$s/ϕ_true"))
                ϕm_r = wrap(read(f, "sim_$s/ϕ_mj"))
                ϕm_d = debias_phi_with_WL(ϕm_r, WL_mj; minW=minW)
                kfac = @. (ℓ_kk^2 / 2)^2
                cl_a = get_Cℓ(ϕm_d; Δℓ=Δℓ_spec)
                cl_x = get_Cℓ(ϕt, ϕm_d; Δℓ=Δℓ_spec)
                push!(Cl_auto_mj_sims,  kfac .* Float64.(cl_a.Cℓ))
                push!(Cl_cross_mj_sims, kfac .* Float64.(cl_x.Cℓ))
                cl_tt  = get_Cℓ(ϕt; Δℓ=Δℓ_spec)
                denom  = sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_a.Cℓ), 0.0))
                push!(ρ_mj_sims, clamp.(Float64.(cl_x.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
                print("\r  MAP sim $i/$(length(map_sims))"); flush(stdout)
            end
        end
        println(); println("  $(length(Cl_auto_mj_sims)) MAP sims")
    end

    # σ_scale for this patch
    θpr = θpix_sim * π / (180 * 60)
    f_sky_sim   = (Nside_sim * θpr)^2 / (4π)
    σ_sc        = sqrt(f_sky_sim / f_sky_paper)

    ℓv = ℓ_kk !== nothing ? ℓ_kk : Float64[]
    Tmat = reduce(hcat, Cl_true_sims)
    Lc, C̄_true, _ = coarsen(ℓv, Tmat; edges=proc_edges)
    Neff_v = @. (2*Lc + 1) * ΔL * f_sky_paper

    function proc(sa, sx)
        nb = length(Lc)
        z = fill(NaN, nb)
        isempty(sa) && return z,z,z,z,z,z
        A = reduce(hcat, sa); X = reduce(hcat, sx)
        _, C̄a, σa = coarsen(ℓv, A; edges=proc_edges); σa .*= σ_sc
        _, C̄x, σx = coarsen(ℓv, X; edges=proc_edges); σx .*= σ_sc
        σ_th_a = @. sqrt(2 / Neff_v) * abs(C̄a)
        σ_th_x = @. sqrt(max(abs(C̄_true)*abs(C̄a) + C̄x^2, 0.0) / Neff_v)
        C̄a, σa, σ_th_a, C̄x, σx, σ_th_x
    end

    C̄_a_qe, σ_a_qe, σ_th_a_qe, C̄_x_qe, σ_x_qe, σ_th_x_qe = proc(Cl_auto_qe_sims,  Cl_cross_qe_sims)
    C̄_a_gi, σ_a_gi, σ_th_a_gi, C̄_x_gi, σ_x_gi, σ_th_x_gi  = proc(Cl_auto_gi_sims,  Cl_cross_gi_sims)
    C̄_a_mj, σ_a_mj, σ_th_a_mj, C̄_x_mj, σ_x_mj, σ_th_x_mj  = proc(Cl_auto_mj_sims,  Cl_cross_mj_sims)

    ρ_proc(sims) = begin
        isempty(sims) && return (fill(NaN,length(Lc)), fill(NaN,length(Lc)))
        R = reduce(hcat, sims)
        _, ρ̄, σρ = coarsen(ℓv, R; edges=proc_edges); ρ̄, σρ
    end
    ρ̄_qe, σρ_qe = ρ_proc(ρ_qe_sims)
    ρ̄_gi, σρ_gi = ρ_proc(ρ_gi_sims)
    ρ̄_mj, σρ_mj = ρ_proc(ρ_mj_sims)

    # SNR
    T_b   = bands_per_sim(ℓv, Cl_true_sims,            proc_edges)
    Ba_qe = bands_per_sim(ℓv, Cl_auto_qe_sims,         proc_edges)
    Bx_qe = bands_per_sim(ℓv, Cl_cross_qe_sims,        proc_edges)
    Ba_gi = bands_per_sim(ℓv, Cl_auto_gi_sims,      proc_edges)
    Bx_gi = bands_per_sim(ℓv, Cl_cross_gi_sims,     proc_edges)
    Ba_mj = bands_per_sim(ℓv, Cl_auto_mj_sims,      proc_edges)
    Bx_mj = bands_per_sim(ℓv, Cl_cross_mj_sims,     proc_edges)
    Lc_e  = [0.5*(proc_edges[b]+proc_edges[b+1]) for b in 1:(length(proc_edges)-1)]

    snr(M,) = snr_compute(M, T_b, Lc_e, snr_L_lo, snr_L_hi, σ_sc)
    snr_a_qe = snr(Ba_qe); snr_x_qe = snr(Bx_qe)
    snr_a_gi = snr(Ba_gi); snr_x_gi = snr(Bx_gi)
    snr_a_mj = snr(Ba_mj); snr_x_mj = snr(Bx_mj)

    _sf(x) = isnan(x) ? "     -" : @sprintf("%6.1f", x)
    println("  SNR (L=$(Int(snr_L_lo))-$(Int(snr_L_hi))):")
    println("    QE             Auto=$(_sf(snr_a_qe))  Cross=$(_sf(snr_x_qe))")
    !isempty(Cl_auto_gi_sims) && println("    GI             Auto=$(_sf(snr_a_gi))  Cross=$(_sf(snr_x_gi))")
    !isempty(Cl_auto_mj_sims) && println("    MAP joint      Auto=$(_sf(snr_a_mj))  Cross=$(_sf(snr_x_mj))")

    (label=label, Lc=Lc, Neff_v=Neff_v, xlim=xlim_plot, C̄_true=C̄_true,
     C̄_a_qe=C̄_a_qe, σ_a_qe=σ_a_qe, σ_th_a_qe=σ_th_a_qe,
     C̄_x_qe=C̄_x_qe, σ_x_qe=σ_x_qe, σ_th_x_qe=σ_th_x_qe,
     C̄_a_gi=C̄_a_gi, σ_a_gi=σ_a_gi, σ_th_a_gi=σ_th_a_gi,
     C̄_x_gi=C̄_x_gi, σ_x_gi=σ_x_gi, σ_th_x_gi=σ_th_x_gi,
     C̄_a_mj=C̄_a_mj, σ_a_mj=σ_a_mj, σ_th_a_mj=σ_th_a_mj,
     C̄_x_mj=C̄_x_mj, σ_x_mj=σ_x_mj, σ_th_x_mj=σ_th_x_mj,
     ρ̄_qe=ρ̄_qe, σρ_qe=σρ_qe, ρ̄_gi=ρ̄_gi, σρ_gi=σρ_gi, ρ̄_mj=ρ̄_mj, σρ_mj=σρ_mj,
     W_qe_raw=W_qe_raw, W_gi_b=W_gi_b, W_qe_s=W_qe_s, W_gi_s=W_gi_s,
     W_mj_raw=W_mj_raw, ℓ_wl=ℓ_template, ℓ_wl_map=ℓ_template_map,
     snr=(a_qe=snr_a_qe, x_qe=snr_x_qe, a_gi=snr_a_gi, x_gi=snr_x_gi,
          a_mj=snr_a_mj, x_mj=snr_x_mj),
     snr_L_range=(snr_L_lo, snr_L_hi), Lc_edges=Lc_e,
     nsims_qegi=length(qegi_sims), nsims_map=length(Cl_auto_mj_sims),
     has_map=!isempty(Cl_auto_mj_sims),
     has_qe=!isempty(Cl_auto_qe_sims), has_gi=!isempty(Cl_auto_gi_sims),
     Ba_qe=Ba_qe, Bx_qe=Bx_qe, Ba_gi=Ba_gi, Bx_gi=Bx_gi,
     Ba_mj=Ba_mj, Bx_mj=Bx_mj, T_b=T_b,
     Cl_auto_qe_sims=Cl_auto_qe_sims, Cl_cross_qe_sims=Cl_cross_qe_sims,
     Cl_auto_gi_sims=Cl_auto_gi_sims, Cl_cross_gi_sims=Cl_cross_gi_sims,
     Cl_auto_mj_sims=Cl_auto_mj_sims, Cl_cross_mj_sims=Cl_cross_mj_sims,
     Cl_true_sims=Cl_true_sims, ℓ_kk=ℓv,
     σ_scale=σ_sc, f_sky_sim=f_sky_sim)
end

# ── Run both noise levels ─────────────────────────────────────────────────────
s4 = process_noise_level(
    "results/WL_qe_gi_12000.jld2",
    "results/phi_maps_qe_gi_12000.jld2",
    "S4-like (1 µK-arcmin)";
    Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
    map_wl_file  = "results/WL_map_12000.jld2",
    map_phi_file = "results/phi_maps_map_12000.jld2",
    snr_L_lo=6000.0, snr_L_hi=10000.0,
    xlim_plot=(5000.0, 11000.0))

ul_files_exist = isfile("results/WL_qe_gi_12000_ul.jld2") &&
                 isfile("results/phi_maps_qe_gi_12000_ul.jld2")
ul = ul_files_exist ? process_noise_level(
    "results/WL_qe_gi_12000_ul.jld2",
    "results/phi_maps_qe_gi_12000_ul.jld2",
    "UL (0.1 µK-arcmin)";
    Lmax=12000, beamFWHM=0.3, μKarcminT=0.1,
    map_wl_file  = isfile("results/WL_map_12000_ul.jld2")  ? "results/WL_map_12000_ul.jld2"  : nothing,
    map_phi_file = isfile("results/phi_maps_map_12000_ul.jld2") ? "results/phi_maps_map_12000_ul.jld2" : nothing,
    snr_L_lo=6000.0, snr_L_hi=10000.0,
    xlim_plot=(5000.0, 11000.0)) : nothing

datasets = filter(!isnothing, Any[s4, ul])

# ── Plot style ─────────────────────────────────────────────────────────────────
const ticker = PythonPlot.matplotlib.ticker

PythonPlot.rc("font",        family="serif", size=11)
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)
PythonPlot.rc("xtick.minor", width=0.5, size=2.5, visible=true)
PythonPlot.rc("ytick.minor", width=0.5, size=2.5, visible=true)

CLR = Dict("qe"=>"#D62728", "gi"=>"#1F77B4", "mj"=>"#9467BD")   # red=QE, blue=GI (matches paper)
LBL = Dict("qe"=>"QE", "gi"=>"GI", "mj"=>"MAP joint")

function set_log_ticks(ax, ymin, ymax)
    lo = floor(Int, log10(max(ymin, 1e-100)))
    hi = ceil(Int,  log10(max(ymax, 1e-100)))
    lo >= hi && (hi = lo + 1)
    decades = [10.0^k for k in lo:hi]
    ax.yaxis.set_major_locator(ticker.FixedLocator(decades))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9)))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10, labelOnlyBase=true))
    ax.set_ylim(10.0^lo * 0.6, 10.0^hi * 2.0)
end

# ── fig3: sigma panels (auto + cross) ─────────────────────────────────────────
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols;
        figsize=(5.5*ncols, 8.0), sharex="col", sharey="row", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    row0_vals = Float64[]   # collect across all datasets for shared y-range
    row1_vals = Float64[]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1
        title = "$(d.label)  ($(d.nsims_qegi) sims)"
        d.has_map && (title *= ", MAP: $(d.nsims_map)")

        # Row 0: auto sigma
        ax = getax(0, c)
        for (key, σ, σ_th) in [("qe", d.σ_a_qe, d.σ_th_a_qe),
                                ("gi", d.σ_a_gi, d.σ_th_a_gi),
                                ("mj", d.σ_a_mj, d.σ_th_a_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(σ) & (σ > 0)
            !any(msk) && continue
            ax.semilogy(Lc[msk], σ[msk]; color=CLR[key], lw=2, label=LBL[key])
            append!(row0_vals, σ[msk])
            msk_th = @. !isnan(Lc) & isfinite(σ_th) & (σ_th > 0)
            any(msk_th) && ax.semilogy(Lc[msk_th], σ_th[msk_th];
                color=CLR[key], ls="--", lw=1.2, alpha=0.7)
        end
        ax.set_title(title, fontsize=9)
        ax.set_xlim(d.xlim...)
        ci == 1 && ax.legend(loc="upper left", frameon=false, fontsize=8,
            title="solid=sim, dashed=Knox", title_fontsize=7)

        # Row 1: cross sigma
        ax = getax(1, c)
        for (key, σ, σ_th) in [("qe", d.σ_x_qe, d.σ_th_x_qe),
                                ("gi", d.σ_x_gi, d.σ_th_x_gi),
                                ("mj", d.σ_x_mj, d.σ_th_x_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(σ) & (σ > 0)
            !any(msk) && continue
            ax.semilogy(Lc[msk], σ[msk]; color=CLR[key], lw=2, label=LBL[key])
            append!(row1_vals, σ[msk])
            msk_th = @. !isnan(Lc) & isfinite(σ_th) & (σ_th > 0)
            any(msk_th) && ax.semilogy(Lc[msk_th], σ_th[msk_th];
                color=CLR[key], ls="--", lw=1.2, alpha=0.7)
        end
        ax.set_xlim(d.xlim...); ax.set_xlabel(L"L", fontsize=12)
        ci == 1 && ax.legend(frameon=false, fontsize=8)
    end

    # Apply shared y-ranges (sharey="row" means setting one axis sets all in the row)
    !isempty(row0_vals) && set_log_ticks(getax(0, 0), minimum(row0_vals)*0.5, maximum(row0_vals)*2)
    !isempty(row1_vals) && set_log_ticks(getax(1, 0), minimum(row1_vals)*0.5, maximum(row1_vals)*2)

    getax(0, 0).set_ylabel(L"\sigma[C_L^{\hat\kappa\hat\kappa}]", fontsize=12)
    getax(1, 0).set_ylabel(L"\sigma[C_L^{\kappa\hat\kappa}]", fontsize=12)

    fig.savefig("$OUT_DIR/fig3_sigma_panels.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig3_sigma_panels.png")
end

# ── fig2: mean spectra ─────────────────────────────────────────────────────────
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols;
        figsize=(5.5*ncols, 8.0), sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1

        # Row 0: mean cross
        ax = getax(0, c)
        msk_t = @. !isnan(Lc) & isfinite(d.C̄_true) & (d.C̄_true > 0)
        any(msk_t) && ax.semilogy(Lc[msk_t], d.C̄_true[msk_t];
            color="k", ls="--", lw=1.8, label=L"C_L^{\kappa\kappa}\,(true)")
        for (key, C̄x) in [("qe", d.C̄_x_qe), ("gi", d.C̄_x_gi), ("mj", d.C̄_x_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(C̄x) & (abs(C̄x) > 0)
            any(msk) && ax.semilogy(Lc[msk], abs.(C̄x[msk]); color=CLR[key], lw=2, label=LBL[key])
        end
        ax.set_title(d.label, fontsize=9); ax.set_xlim(d.xlim...)
        ci == 1 && ax.legend(loc="upper right", frameon=false, fontsize=8)

        # Row 1: mean auto
        ax = getax(1, c)
        any(msk_t) && ax.semilogy(Lc[msk_t], d.C̄_true[msk_t];
            color="k", ls="--", lw=1.8, label=L"C_L^{\kappa\kappa}\,(true)")
        for (key, C̄a) in [("qe", d.C̄_a_qe), ("gi", d.C̄_a_gi), ("mj", d.C̄_a_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(C̄a) & (abs(C̄a) > 0)
            any(msk) && ax.semilogy(Lc[msk], abs.(C̄a[msk]); color=CLR[key], lw=2, label=LBL[key])
        end
        ax.set_xlim(d.xlim...); ax.set_xlabel(L"L", fontsize=12)
        ci == 1 && ax.legend(loc="upper right", frameon=false, fontsize=8)
    end
    getax(0, 0).set_ylabel(L"\bar{C}_L^{\kappa\hat\kappa}", fontsize=12)
    getax(1, 0).set_ylabel(L"\bar{C}_L^{\hat\kappa\hat\kappa}", fontsize=12)

    fig.savefig("$OUT_DIR/fig2_mean_spectra.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig2_mean_spectra.png")
end

# ── fig4: Effective reconstruction noise (paper Eq. 33) ───────────────────────
# N_L,eff = σ̂[C^{κ̂κ̂}_L] × sqrt(ΔL(2L+1)f_sky/2) - C^κκ_L
# σ̂ is already rescaled to f_sky_paper=0.4; (2L+1)≈2(L+0.5), so:
# N_L,eff = σ̂ × sqrt(ΔL × (Lc+0.5) × f_sky_paper) - C̄_true
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(1, ncols; figsize=(5.5*ncols, 4.5), constrained_layout=true)
    getax(c) = ncols == 1 ? axs : axs[c]

    for (ci, d) in enumerate(datasets)
        ax = getax(ci - 1); Lc = d.Lc
        # Signal curve C^κκ
        msk_t = @. !isnan(Lc) & isfinite(d.C̄_true) & (d.C̄_true > 0)
        any(msk_t) && ax.semilogy(Lc[msk_t], d.C̄_true[msk_t];
            color="k", ls=":", lw=1.5, label=L"C_L^{\kappa\kappa}\,(signal)")
        for (key, σ_a) in [("qe", d.σ_a_qe), ("gi", d.σ_a_gi), ("mj", d.σ_a_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(σ_a) & (σ_a > 0)
            !any(msk) && continue
            # Eq. 33: N_eff = σ̂ × sqrt(ΔL(2L+1)f_sky/2) - C^κκ
            prefac = @. sqrt(ΔL * (Lc[msk] + 0.5) * f_sky_paper)
            N_eff  = @. max(σ_a[msk] * prefac - d.C̄_true[msk], 1e-20)
            ax.semilogy(Lc[msk], N_eff; color=CLR[key], lw=2, label=LBL[key])
        end
        ax.set_xlabel(L"L", fontsize=12); ax.set_title(d.label, fontsize=9)
        ax.set_xlim(d.xlim...)
        ci == 1 && ax.legend(loc="upper left", frameon=false, fontsize=8)
    end
    getax(0).set_ylabel(L"N_{L,\mathrm{eff}}^{\kappa\kappa}", fontsize=12)

    fig.savefig("$OUT_DIR/fig4_Neff.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig4_Neff.png")
end

# ── fig5: improvement ratio σ_QE / σ_estimator ────────────────────────────────
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols; figsize=(5.5*ncols, 7.0),
        sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1
        for (row, (σ_qe, entries)) in enumerate([
                (d.σ_a_qe, [("gi", d.σ_a_gi), ("mj", d.σ_a_mj)]),
                (d.σ_x_qe, [("gi", d.σ_x_gi), ("mj", d.σ_x_mj)])])
            ax = getax(row - 1, c)
            ax.axhline(1.0; color="grey", ls=":", lw=1)
            for (key, σ) in entries
                key == "mj" && !d.has_map && continue
                msk = @. !isnan(Lc) & isfinite(σ_qe) & (σ_qe > 0) & isfinite(σ) & (σ > 0)
                !any(msk) && continue
                ratio = σ_qe[msk] ./ σ[msk]
                ax.plot(Lc[msk], ratio; color=CLR[key], lw=2, label=LBL[key])
            end
            ax.set_xlim(d.xlim...)
            row == 1 && ax.set_title(d.label, fontsize=9)
            row == 2 && ax.set_xlabel(L"L", fontsize=12)
            ci == 1 && ax.legend(frameon=false, fontsize=8)
        end
    end
    getax(0, 0).set_ylabel(L"\sigma_{\rm QE}^{\rm auto} / \sigma_{\rm est}^{\rm auto}", fontsize=10)
    getax(1, 0).set_ylabel(L"\sigma_{\rm QE}^{\rm cross} / \sigma_{\rm est}^{\rm cross}", fontsize=10)

    fig.savefig("$OUT_DIR/fig5_improvement_ratio.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig5_improvement_ratio.png")
end

# ── fig6: correlation coefficient ρ_L ─────────────────────────────────────────
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(1, ncols; figsize=(5.5*ncols, 4.5), constrained_layout=true)
    getax(c) = ncols == 1 ? axs : axs[c]

    for (ci, d) in enumerate(datasets)
        ax = getax(ci - 1); Lc = d.Lc
        ax.axhline(1.0; color="grey", ls=":", lw=1, alpha=0.7)
        for (key, ρ̄, σρ) in [("qe", d.ρ̄_qe, d.σρ_qe),
                               ("gi", d.ρ̄_gi, d.σρ_gi),
                               ("mj", d.ρ̄_mj, d.σρ_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(ρ̄)
            !any(msk) && continue
            ax.plot(Lc[msk], ρ̄[msk]; color=CLR[key], lw=2, label=LBL[key])
            ax.fill_between(Lc[msk], ρ̄[msk] .- σρ[msk], ρ̄[msk] .+ σρ[msk];
                color=CLR[key], alpha=0.15)
        end
        ax.set_xlabel(L"L", fontsize=12)
        ax.set_title(d.label, fontsize=9)
        ax.set_xlim(d.xlim...); ax.set_ylim(-0.1, 1.15)
        ci == 1 && ax.legend(frameon=false, fontsize=8)
    end
    getax(0).set_ylabel(L"\rho_L = \langle C_L^{\phi_{\rm true},\hat\phi} / \sqrt{C_L^{\phi\phi} C_L^{\hat\phi\hat\phi}} \rangle", fontsize=9)

    fig.savefig("$OUT_DIR/fig6_rho_L.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig6_rho_L.png")
end

# ── fig7: combined Neff + ρ ────────────────────────────────────────────────────
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols; figsize=(5.5*ncols, 8.0),
        sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1

        # Row 0: Neff from cross
        ax = getax(0, c)
        for (key, σ, C̄) in [("qe", d.σ_x_qe, d.C̄_x_qe),
                              ("gi", d.σ_x_gi, d.C̄_x_gi),
                              ("mj", d.σ_x_mj, d.C̄_x_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(σ) & (σ > 0) & isfinite(C̄) & (C̄ > 0)
            any(msk) && ax.semilogy(Lc[msk], (C̄[msk] ./ σ[msk]).^2;
                color=CLR[key], lw=2, label=LBL[key])
        end
        ax.set_title(d.label, fontsize=9); ax.set_xlim(d.xlim...)
        ci == 1 && ax.legend(frameon=false, fontsize=8)

        # Row 1: ρ_L
        ax = getax(1, c)
        ax.axhline(1.0; color="grey", ls=":", lw=1, alpha=0.7)
        for (key, ρ̄) in [("qe", d.ρ̄_qe), ("gi", d.ρ̄_gi), ("mj", d.ρ̄_mj)]
            key == "mj" && !d.has_map && continue
            msk = @. !isnan(Lc) & isfinite(ρ̄)
            any(msk) && ax.plot(Lc[msk], ρ̄[msk]; color=CLR[key], lw=2)
        end
        ax.set_xlabel(L"L", fontsize=12); ax.set_xlim(d.xlim...); ax.set_ylim(-0.1, 1.15)
    end
    getax(0, 0).set_ylabel(L"(C_L^{\kappa\hat\kappa} / \sigma_L)^2", fontsize=11)
    getax(1, 0).set_ylabel(L"\rho_L", fontsize=12)

    fig.savefig("$OUT_DIR/fig7_Neff_rho.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig7_Neff_rho.png")
end

# ── fig_WL: transfer function W_L ─────────────────────────────────────────────
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(1, ncols; figsize=(5.5*ncols, 4.0), constrained_layout=true)
    getax(c) = ncols == 1 ? axs : axs[c]

    for (ci, d) in enumerate(datasets)
        ax = getax(ci - 1)
        ax.axhline(1.0; color="grey", ls=":", lw=1, alpha=0.7)
        ax.axhline(0.0; color="k",    ls="--", lw=0.5, alpha=0.5)
        ℓ = d.ℓ_wl; xlo, xhi = d.xlim
        for (key, W_raw, W_s) in [("qe", d.W_qe_raw, d.W_qe_s),
                                   ("gi", d.W_gi_b,   d.W_gi_s)]
            W_raw === nothing && continue
            msk = @. !isnan(ℓ) & isfinite(W_raw) & (ℓ >= xlo) & (ℓ <= xhi)
            any(msk) && ax.plot(ℓ[msk], W_raw[msk];
                color=CLR[key], lw=0.8, alpha=0.35)
            W_s !== nothing && any(msk) &&
                ax.plot(ℓ[msk], W_s[msk]; color=CLR[key], lw=2, label=LBL[key]*" (smooth)")
        end
        if d.has_map && d.W_mj_raw !== nothing
            ℓm = d.ℓ_wl_map; W_m = d.W_mj_raw
            msk = @. !isnan(ℓm) & isfinite(W_m) & (ℓm >= xlo) & (ℓm <= xhi)
            any(msk) && ax.plot(ℓm[msk], W_m[msk]; color=CLR["mj"], lw=2, label=LBL["mj"])
        end
        ax.set_xlabel(L"L", fontsize=12); ax.set_title(d.label, fontsize=9)
        ax.set_xlim(d.xlim...); ax.set_ylim(-0.2, 1.5)
        ci == 1 && ax.legend(frameon=false, fontsize=8)
    end
    getax(0).set_ylabel(L"W_L = \langle C_L^{\phi_{\rm true},\hat\phi} / C_L^{\phi\phi} \rangle", fontsize=10)

    fig.savefig("$OUT_DIR/fig_WL.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig_WL.png")
end

# ── figB: convergence with nsims ──────────────────────────────────────────────
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols; figsize=(5.5*ncols, 7.0),
        sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    for (ci, d) in enumerate(datasets)
        c = ci - 1
        ℓv = d.ℓ_kk
        isempty(ℓv) && continue

        for (row, (sims_list, ylabel)) in enumerate([
                ([("qe", d.Cl_cross_qe_sims), ("gi", d.Cl_cross_gi_sims)],
                 L"\sigma[C_L^{\kappa\hat\kappa}]"),
                ([("qe", d.Cl_auto_qe_sims), ("gi", d.Cl_auto_gi_sims)],
                 L"\sigma[C_L^{\hat\kappa\hat\kappa}]")])
            ax = getax(row - 1, c)
            for (key, sims) in sims_list
                length(sims) < 4 && continue
                nmax = length(sims)
                steps = unique(vcat([div(nmax, 10) * k for k in 1:10], [nmax]))
                σ_mid = Float64[]   # σ at middle ΔL band, vs nsims
                for n in steps
                    A = reduce(hcat, sims[1:n])
                    _, _, σ_v = coarsen(ℓv, A; edges=proc_edges)
                    σ_sc = d.σ_scale
                    # pick the middle band
                    mid = max(1, length(σ_v) ÷ 2)
                    push!(σ_mid, isnan(σ_v[mid]) ? NaN : σ_v[mid] * σ_sc)
                end
                msk = isfinite.(σ_mid)
                any(msk) && ax.plot(steps[msk], σ_mid[msk]; color=CLR[key], lw=2, label=LBL[key])
            end
            ax.set_ylabel(ylabel, fontsize=10)
            row == 2 && ax.set_xlabel("# sims", fontsize=11)
            row == 1 && ax.set_title(d.label, fontsize=9)
            ci == 1 && ax.legend(frameon=false, fontsize=8)
        end
    end
    fig.savefig("$OUT_DIR/figB_convergence.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved figB_convergence.png")
end

# ── fig_covariance_correlation ─────────────────────────────────────────────────
function corr_matrix(B::Matrix)
    nb, ns = size(B); ns < 3 && return fill(NaN, nb, nb)
    Ac = B .- mean(B; dims=2)
    Cov = (Ac * Ac') ./ (ns - 1)
    σ = sqrt.(abs.(diag(Cov))); σ[σ .< 1e-100] .= 1.0
    Cov ./ (σ * σ')
end

function fill_corr_ax(ax, B::Matrix, Lc_e::Vector; title="")
    valid = findall(i -> i <= size(B,1) && i <= length(Lc_e) &&
                         !isnan(Lc_e[i]) && all(isfinite.(B[i,:])) && std(B[i,:]) > 1e-30,
                    1:min(size(B,1), length(Lc_e)))
    isempty(valid) && return nothing
    C = corr_matrix(B[valid, :])
    Lv = Lc_e[valid]
    im = ax.imshow(C; cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                   extent=[Lv[1], Lv[end], Lv[1], Lv[end]], aspect="equal",
                   interpolation="nearest")
    ax.set_xlabel(L"L'", fontsize=9); ax.set_ylabel(L"L", fontsize=9)
    ax.set_title(title, fontsize=8)
    im
end

let
    n_per = 3   # GI, QE, MAP per dataset
    ncols = n_per * length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols; figsize=(4.0*ncols, 8.5), constrained_layout=true)
    getac(r, c) = axs[r, c]

    for (ci, d) in enumerate(datasets)
        c0 = n_per * (ci - 1)
        for (row, spec) in enumerate(["auto", "cross"])
            r = row - 1
            B_qe = spec == "auto" ? d.Ba_qe : d.Bx_qe
            B_gi = spec == "auto" ? d.Ba_gi : d.Bx_gi
            B_mj = spec == "auto" ? d.Ba_mj : d.Bx_mj
            for (ck, B, ttl) in [(c0,   B_qe, "$(d.label) QE $spec"),
                                  (c0+1, B_gi, "$(d.label) GI $spec"),
                                  (c0+2, B_mj, "$(d.label) MAP $spec")]
                ax = getac(r, ck)
                if (!d.has_map && ck == c0+2) || size(B, 2) < 3
                    ax.set_visible(false); continue
                end
                im = fill_corr_ax(ax, B, d.Lc_edges; title=ttl)
                im !== nothing && fig.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
            end
        end
    end
    fig.suptitle(L"Bandpower correlation $\rho_{LL'}$", fontsize=11)
    fig.savefig("$OUT_DIR/fig_covariance_correlation.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig_covariance_correlation.png")
end

# ── figA: quarter-consistency plot ────────────────────────────────────────────
# Split sims into 4 equal quarters; plot σ for each quarter (thin lines, same
# colour as estimator) + full-sample σ as thick dotted line.
# Both auto (rows 0,2) and cross (rows 1,3) panels, one column per noise level.
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols;
        figsize=(5.5*ncols, 8.0), sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    quarter_alphas = [0.85, 0.65, 0.45, 0.30]   # progressively lighter quarters

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1; ℓv = d.ℓ_kk

        auto_list  = [("qe", d.Cl_auto_qe_sims),  ("gi", d.Cl_auto_gi_sims)]
        cross_list = [("qe", d.Cl_cross_qe_sims), ("gi", d.Cl_cross_gi_sims)]
        d.has_map && push!(auto_list,  ("mj", d.Cl_auto_mj_sims))
        d.has_map && push!(cross_list, ("mj", d.Cl_cross_mj_sims))
        for (row, (sims_list, ylabel, is_auto)) in enumerate([
                (auto_list,  L"\sigma[C_L^{\hat\kappa\hat\kappa}]", true),
                (cross_list, L"\sigma[C_L^{\kappa\hat\kappa}]",     false)])
            ax = getax(row - 1, c)

            for (key, sims) in sims_list
                length(sims) < 8 && continue
                n = length(sims)
                q = n ÷ 4   # quarter size

                # Full-sample σ (thick dotted — the "mean" reference)
                A_full = reduce(hcat, sims)
                _, _, σ_full = coarsen(ℓv, A_full; edges=proc_edges)
                σ_full .*= d.σ_scale
                msk = @. !isnan(Lc) & isfinite(σ_full) & (σ_full > 0)
                any(msk) && ax.semilogy(Lc[msk], σ_full[msk];
                    color=CLR[key], lw=2.5, ls=":", label="$(LBL[key]) (all $(n))")

                # 4 quarters (thin solid, progressively lighter)
                for qi in 1:4
                    i_lo = (qi-1)*q + 1
                    i_hi = qi == 4 ? n : qi*q   # last quarter gets remainder
                    sub = sims[i_lo:i_hi]
                    length(sub) < 2 && continue
                    A_q = reduce(hcat, sub)
                    _, _, σ_q = coarsen(ℓv, A_q; edges=proc_edges)
                    σ_q .*= d.σ_scale
                    msk_q = @. !isnan(Lc) & isfinite(σ_q) & (σ_q > 0)
                    lbl = qi == 1 ? "$(LBL[key]) Q$(qi)–Q4 ($(length(sub)) each)" : ""
                    any(msk_q) && ax.semilogy(Lc[msk_q], σ_q[msk_q];
                        color=CLR[key], lw=1.2, ls="-",
                        alpha=quarter_alphas[qi], label=lbl)
                end
            end

            ax.set_xlim(d.xlim...)
            row == 1 && ax.set_title(d.label, fontsize=9)
            row == 2 && ax.set_xlabel(L"L", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=10)
            ci == 1 && ax.legend(loc="upper left", frameon=false, fontsize=7,
                title="dotted=all sims, solid=quarter", title_fontsize=6)
        end
    end

    fig.suptitle("Quarter consistency  (dotted = full sample, solid = 4 quarters)",
        fontsize=10)
    fig.savefig("$OUT_DIR/figA_quarter_consistency.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved figA_quarter_consistency.png")
end

# ── figC: MAP convergence (mean log-posterior vs step) ────────────────────────
let
    map_wl_files = [
        ("S4-like (1 µK-arcmin)",  "results/WL_map_12000.jld2"),
        ("UL (0.1 µK-arcmin)",     "results/WL_map_12000_ul.jld2"),
    ]
    entries = [(lbl, f) for (lbl, f) in map_wl_files if isfile(f)]
    isempty(entries) && (println("No MAP WL files found — skipping convergence plot"); @goto skip_conv)

    fig, axs = PythonPlot.subplots(1, length(entries);
        figsize=(5.5*length(entries), 4.5), constrained_layout=true)
    getax(i) = length(entries) == 1 ? axs : axs[i-1]

    for (i, (lbl, wl_file)) in enumerate(entries)
        d_map = JLD2.load(wl_file)
        !haskey(d_map, "logpdf_histories") && (getax(i).set_title("$lbl\n(no history saved)"); continue)
        hists = d_map["logpdf_histories"]
        isempty(hists) && continue
        nsteps = minimum(length.(hists))
        # Shift so step 1 = 0 for each sim, then average
        shifted = [Float64.(h[1:nsteps]) .- h[1] for h in hists]
        mean_lp = [mean(s[t] for s in shifted) for t in 1:nsteps]
        steps   = 1:nsteps

        ax = getax(i)
        for s in shifted
            ax.plot(steps, s; color="#9467BD", lw=0.4, alpha=0.15)
        end
        ax.plot(steps, mean_lp; color="#9467BD", lw=2.5, label="mean ($(length(hists)) sims)")
        ax.axhline(0; color="k", ls=":", lw=0.8)
        ax.set_xlabel("MAP step"); ax.set_ylabel(L"\Delta\log p\,(\phi|\mathrm{data})")
        ax.set_title(lbl, fontsize=9)
        ax.legend(frameon=false, fontsize=8)

        last5_rate = nsteps >= 6 ? (mean_lp[end] - mean_lp[end-4]) / 5 : NaN
        avg_rate   = mean_lp[end] / (nsteps - 1)
        conv_str   = (!isnan(last5_rate) && abs(avg_rate) > 0) ?
            (abs(last5_rate)/abs(avg_rate) < 0.05 ? "✓ converged" : "⚠ still improving") : ""
        !isempty(conv_str) && ax.set_title("$lbl  $conv_str", fontsize=9)
    end
    fig.savefig("$OUT_DIR/figC_map_convergence.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved figC_map_convergence.png")
    @label skip_conv
end

# ── SNR table ─────────────────────────────────────────────────────────────────
let
    _sf2(x) = isnan(x) ? "         -" : @sprintf("%9.1f", x)
    hdr = "SNR Table  (f_sky=0.4, from sims)"
    sep = "="^80
    col = "  $(rpad("Dataset", 28))  Auto-QE  Auto-GI  Auto-MAP  Cross-QE  Cross-GI  Cross-MAP"
    lines = [hdr, sep, col]
    for d in datasets
        lo, hi = Int.(d.snr_L_range)
        tag = rpad("$(d.label) L=$lo-$hi", 30)
        push!(lines, "  $tag $(_sf2(d.snr.a_qe)) $(_sf2(d.snr.a_gi)) $(_sf2(d.snr.a_mj))  $(_sf2(d.snr.x_qe)) $(_sf2(d.snr.x_gi)) $(_sf2(d.snr.x_mj))")
    end
    push!(lines, sep)
    push!(lines, "  Paper (UL): Auto QE≈205, Auto GI≈1515, Cross QE≈710, Cross GI≈4100")
    push!(lines, "  Paper (S4): Auto QE≈100, Auto GI≈360,  Cross QE≈550, Cross GI≈1440")
    for l in lines; println(l); end
    open("$OUT_DIR/snr_table.txt", "w") do io
        for l in lines; println(io, l); end
    end
    println("Saved snr_table.txt")
end

println("\nAll figures saved to $OUT_DIR")
