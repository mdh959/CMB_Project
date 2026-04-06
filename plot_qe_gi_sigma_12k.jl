#!/usr/bin/env julia
"""
plot_qe_gi_sigma_12k.jl  —  Hadzhiyska+2019 Fig. 3

Pipeline (mirrors old normalization.jl + error_mean.jl approach):
  1. Load empirical W_L and stored phi maps from run_qe_gi_wl12k.jl.
  2. Debias each phi in Fourier space: ϕ_deb(k) = ϕ_raw(k) / W_L(|k|).
  3. Compute per-sim C_L^{κ_true,κ̂_deb} and C_L^{κ̂_deb,κ̂_deb}.
  4. Coarsen to ΔL=2000 bands, σ = std across sims × σ_scale.
  5. Compare with Knox formula.
  6. Plot both estimators. Bottom panel: mean cross (debiasing diagnostic).
"""

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using Statistics: mean, std
using PythonPlot
using Printf
using JLD2

include("utils.jl")
using .Utils

const Cℓ          = camb(r=0.05, ℓmax=170000)
const θpix        = 0.7438046267475303
const Nside       = 512
const pol         = :I

const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
const f_sky_paper = 0.4
const σ_scale     = sqrt(f_sky_patch / f_sky_paper)
const minW        = 1e-8   # floor for W_L debiasing
const ΔL          = 2000
const Δℓ_spec     = 30

println("f_sky_patch = $(round(f_sky_patch; sigdigits=4))")
println("σ_scale     = $(round(σ_scale; sigdigits=4))  (rescale patch σ → f_sky=0.4)")

# ── Coarsen: per-band mean and std across sims
function coarsen(ℓ, M; edges)
    nb = length(edges) - 1
    Lc = fill(NaN, nb); μ_v = fill(NaN, nb); σ_v = fill(NaN, nb)
    for b in 1:nb
        idx = findall(x -> edges[b] <= x < edges[b+1], ℓ)
        isempty(idx) && continue
        per = filter(isfinite, vec(mean(M[idx, :]; dims=1)))
        length(per) < 2 && continue
        Lc[b] = 0.5*(edges[b]+edges[b+1]); μ_v[b] = mean(per); σ_v[b] = std(per)
    end
    return Lc, μ_v, σ_v
end

# ── SNR helpers ───────────────────────────────────────────────────────────────
# Build nb × nsims matrix of per-band, per-sim power spectra
function bands_per_sim(ℓ::Vector, sims::Vector, edges::Vector)
    nb = length(edges) - 1
    ns = length(sims)
    ns == 0 && return fill(NaN, nb, 0)
    M = fill(NaN, nb, ns)
    for (j, s) in enumerate(sims)
        for b in 1:nb
            idx = findall(x -> edges[b] <= x < edges[b+1], ℓ)
            isempty(idx) && continue
            vals = filter(isfinite, s[idx])
            isempty(vals) || (M[b, j] = mean(vals))
        end
    end
    return M
end

# SNR² = Σ_L (C_L / σ_L)²  — diagonal covariance, matching paper convention.
# Computed per-band so that NaN bands (e.g. above Nyquist for old sims) are skipped
# without discarding all sims.
function snr_compute(M_meas::Matrix, M_true::Matrix, Lc::Vector,
                     L_lo::Real, L_hi::Real, σ_scale::Real)
    size(M_meas, 2) < 2 && return NaN
    sel = findall(b -> !isnan(Lc[b]) && L_lo <= Lc[b] <= L_hi, 1:length(Lc))
    isempty(sel) && return NaN
    snr2 = 0.0
    n_bands = 0
    for b in sel
        a_b = filter(isfinite, vec(M_meas[b, :]))
        t_b = filter(isfinite, vec(M_true[b, :]))
        (length(a_b) < 2 || length(t_b) < 2) && continue
        C_s = mean(t_b)
        σ_b = σ_scale * std(a_b)
        σ_b <= 0 && continue
        snr2 += (C_s / σ_b)^2
        n_bands += 1
    end
    n_bands == 0 && return NaN
    return sqrt(max(0.0, snr2))
end

function process_noise_level(WL_file, phi_maps_file, label;
                             Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
                             map_wl_file=nothing, map_phi_file=nothing,
                             proc_edges=collect(4500:Int(ΔL):12500),
                             snr_L_lo=5000.0, snr_L_hi=11000.0,
                             xlim_plot=(4500.0, 12500.0),
                             θpix_sim::Float64=θpix, Nside_sim::Int=Nside)
    println("\n=== Processing $label ===")

    # ── QE/GI W_L (from 1000-sim file) ───────────────────────────────────────
    d_wl       = JLD2.load(WL_file)
    ℓ_template = Float64.(d_wl["ℓ_template"])
    nsims_wl      = d_wl["nsims_completed"]
    _load_wl(d, k) = (haskey(d, k) && d[k] !== nothing) ? Float64.(d[k]) : nothing
    W_qe_wf       = _load_wl(d_wl, "W_qe_wf")
    W_qe_ul       = _load_wl(d_wl, "W_qe_ul")
    W_gi_b        = _load_wl(d_wl, "W_gi_b")
    W_gi_c        = _load_wl(d_wl, "W_gi_c")
    has_wf        = W_qe_wf !== nothing
    has_ul        = W_qe_ul !== nothing
    has_gi_b      = W_gi_b  !== nothing
    has_gi_c      = W_gi_c  !== nothing
    WL_qe_wf_Cℓs = has_wf   ? Cℓs(ℓ_template, W_qe_wf) : nothing
    WL_qe_ul_Cℓs = has_ul   ? Cℓs(ℓ_template, W_qe_ul) : nothing
    WL_gi_b_Cℓs  = has_gi_b ? Cℓs(ℓ_template, W_gi_b)  : nothing
    WL_gi_c_Cℓs  = has_gi_c ? Cℓs(ℓ_template, W_gi_c)  : nothing
    println("  QE/GI W_L: $nsims_wl sims  (QE wf: $has_wf, GI Boryana: $has_gi_b, GI corr: $has_gi_c)")

    # ── MAP W_L (from MAP-specific file, fewer sims) ──────────────────────────
    has_map = map_wl_file !== nothing && isfile(map_wl_file) &&
              map_phi_file !== nothing && isfile(map_phi_file)
    W_mj = WL_mj_Cℓs = nothing
    nsims_map_done = 0
    ℓ_template_map = ℓ_template   # fallback: same as QE/GI
    if has_map
        d_map = JLD2.load(map_wl_file)
        # MAP may have been run with its own Δℓ; load its ℓ_template if present
        ℓ_template_map = haskey(d_map, "ℓ_template") ? Float64.(d_map["ℓ_template"]) : ℓ_template
        W_mj  = haskey(d_map, "W_mj") ? Float64.(d_map["W_mj"]) : ones(length(ℓ_template_map))
        nsims_map_done = get(d_map, "nsims_map_done", 0)
        WL_mj_Cℓs = Cℓs(ℓ_template_map, W_mj)
        println("  MAP W_L: $nsims_map_done sims from $map_wl_file")
    end

    # ── Projection metadata (from QE/GI phi file) ────────────────────────────
    Cℓn_meta = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    meta_seed = jldopen(phi_maps_file, "r") do f
        first_sim = minimum(parse(Int, match(r"^sim_(\d+)$", k).captures[1])
                            for k in keys(f) if occursin(r"^sim_\d+$", k))
        read(f, "sim_$first_sim/seed")
    end
    (; ϕ) = load_sim(; seed=meta_seed, Cℓ=Cℓ, Cℓn=Cℓn_meta, θpix=θpix_sim,
                       T=Float64, Nside=Nside_sim, beamFWHM=beamFWHM, pol=pol,
                       bandpass_mask=LowPass(Lmax),
                       pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))
    ϕ_ref = Map(ϕ)
    wrap(arr) = typeof(ϕ_ref)(Float64.(arr), ϕ_ref.metadata)

    # ── Per-sim spectra: QE/GI from 1000-sim phi file ─────────────────────────
    Cl_auto_qe_wf_sims  = Vector{Vector{Float64}}()
    Cl_auto_qe_ul_sims  = Vector{Vector{Float64}}()
    Cl_auto_gi_b_sims   = Vector{Vector{Float64}}()
    Cl_auto_gi_c_sims   = Vector{Vector{Float64}}()
    Cl_cross_qe_wf_sims = Vector{Vector{Float64}}()
    Cl_cross_qe_ul_sims = Vector{Vector{Float64}}()
    Cl_cross_gi_b_sims  = Vector{Vector{Float64}}()
    Cl_cross_gi_c_sims  = Vector{Vector{Float64}}()
    Cl_true_sims        = Vector{Vector{Float64}}()
    ℓ_kk                = nothing

    qegi_sims = Int[]
    jldopen(phi_maps_file, "r") do f
        for key in keys(f)
            m = match(r"^sim_(\d+)$", key)
            m !== nothing && push!(qegi_sims, parse(Int, m.captures[1]))
        end
        sort!(qegi_sims)
        for (i, s) in enumerate(qegi_sims)
            ϕ_true_field = wrap(read(f, "sim_$s/ϕ_true"))
            if haskey(f, "sim_$s/ϕ_qe_wf") && WL_qe_wf_Cℓs !== nothing
                ϕ_wf_deb = debias_phi_with_WL(wrap(read(f, "sim_$s/ϕ_qe_wf")), WL_qe_wf_Cℓs; minW=minW)
                cl = get_Cℓ(ϕ_wf_deb; Δℓ=Δℓ_spec); cx = get_Cℓ(ϕ_true_field, ϕ_wf_deb; Δℓ=Δℓ_spec)
                ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl.ℓ)))
                kfac = @. (ℓ_kk^2 / 2)^2
                push!(Cl_auto_qe_wf_sims, kfac .* Float64.(cl.Cℓ)); push!(Cl_cross_qe_wf_sims, kfac .* Float64.(cx.Cℓ))
            end
            if haskey(f, "sim_$s/ϕ_qe_ul") && WL_qe_ul_Cℓs !== nothing
                ϕ_ul_deb = debias_phi_with_WL(wrap(read(f, "sim_$s/ϕ_qe_ul")), WL_qe_ul_Cℓs; minW=minW)
                cl = get_Cℓ(ϕ_ul_deb; Δℓ=Δℓ_spec); cx = get_Cℓ(ϕ_true_field, ϕ_ul_deb; Δℓ=Δℓ_spec)
                ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl.ℓ)))
                kfac = @. (ℓ_kk^2 / 2)^2
                push!(Cl_auto_qe_ul_sims, kfac .* Float64.(cl.Cℓ)); push!(Cl_cross_qe_ul_sims, kfac .* Float64.(cx.Cℓ))
            end
            if haskey(f, "sim_$s/ϕ_gi_b") && WL_gi_b_Cℓs !== nothing
                ϕ_gi_b_deb = debias_phi_with_WL(wrap(read(f, "sim_$s/ϕ_gi_b")), WL_gi_b_Cℓs; minW=minW)
                cl = get_Cℓ(ϕ_gi_b_deb; Δℓ=Δℓ_spec); cx = get_Cℓ(ϕ_true_field, ϕ_gi_b_deb; Δℓ=Δℓ_spec)
                ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl.ℓ)))
                kfac = @. (ℓ_kk^2 / 2)^2
                push!(Cl_auto_gi_b_sims, kfac .* Float64.(cl.Cℓ)); push!(Cl_cross_gi_b_sims, kfac .* Float64.(cx.Cℓ))
            end
            if haskey(f, "sim_$s/ϕ_gi_c") && WL_gi_c_Cℓs !== nothing
                ϕ_gi_c_deb = debias_phi_with_WL(wrap(read(f, "sim_$s/ϕ_gi_c")), WL_gi_c_Cℓs; minW=minW)
                cl = get_Cℓ(ϕ_gi_c_deb; Δℓ=Δℓ_spec); cx = get_Cℓ(ϕ_true_field, ϕ_gi_c_deb; Δℓ=Δℓ_spec)
                ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl.ℓ)))
                kfac = @. (ℓ_kk^2 / 2)^2
                push!(Cl_auto_gi_c_sims, kfac .* Float64.(cl.Cℓ)); push!(Cl_cross_gi_c_sims, kfac .* Float64.(cx.Cℓ))
            end
            cl_tt = get_Cℓ(ϕ_true_field; Δℓ=Δℓ_spec)
            ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl_tt.ℓ)))
            push!(Cl_true_sims, @. (ℓ_kk^2/2)^2 * Float64.(cl_tt.Cℓ))
            print("\r  QE/GI sim $i/$(length(qegi_sims))"); flush(stdout)
        end
    end
    println()
    println("  phi maps: $(length(qegi_sims)) QE/GI sims from $phi_maps_file")

    # ── Per-sim spectra: MAP joint from MAP phi file ───────────────────────────
    Cl_auto_mj_sims  = Vector{Vector{Float64}}()
    Cl_cross_mj_sims = Vector{Vector{Float64}}()

    if has_map
        map_sims = Int[]
        jldopen(map_phi_file, "r") do f
            for key in keys(f)
                m = match(r"^sim_(\d+)$", key)
                m !== nothing && push!(map_sims, parse(Int, m.captures[1]))
            end
            sort!(map_sims)
            for (i, s) in enumerate(map_sims)
                haskey(f, "sim_$s/ϕ_mj") || continue
                ϕ_true_field = wrap(read(f, "sim_$s/ϕ_true"))
                ϕ_mj_deb = debias_phi_with_WL(wrap(read(f, "sim_$s/ϕ_mj")), WL_mj_Cℓs; minW=minW)
                cl = get_Cℓ(ϕ_mj_deb; Δℓ=Δℓ_spec); cx = get_Cℓ(ϕ_true_field, ϕ_mj_deb; Δℓ=Δℓ_spec)
                kfac = @. (ℓ_kk^2 / 2)^2
                push!(Cl_auto_mj_sims, kfac .* Float64.(cl.Cℓ)); push!(Cl_cross_mj_sims, kfac .* Float64.(cx.Cℓ))
                print("\r  MAP sim $i/$(length(map_sims))"); flush(stdout)
            end
        end
        println()
        println("  phi maps: $(length(map_sims)) MAP sims from $map_phi_file")
    end

    # ── Coarsen ───────────────────────────────────────────────────────────────
    # σ_scale for this dataset: rescale patch σ to f_sky=0.4.
    θpix_rad_sim  = θpix_sim * π / (180 * 60)
    f_sky_sim     = (Nside_sim * θpix_rad_sim)^2 / (4π)
    σ_scale_sim   = sqrt(f_sky_sim / f_sky_paper)
    println("  f_sky_sim=$(round(f_sky_sim; sigdigits=4))  σ_scale_sim=$(round(σ_scale_sim; sigdigits=4))")

    Tmat = reduce(hcat, Cl_true_sims)
    Lc, C̄_true, _ = coarsen(ℓ_kk, Tmat; edges=proc_edges)
    Neff = @. (2Lc + 1) * ΔL * f_sky_paper

    function proc(sims_a, sims_x)
        isempty(sims_a) && return (fill(NaN,length(Lc)), fill(NaN,length(Lc)),
                                   fill(NaN,length(Lc)), fill(NaN,length(Lc)),
                                   fill(NaN,length(Lc)), fill(NaN,length(Lc)))
        A = reduce(hcat, sims_a); X = reduce(hcat, sims_x)
        _, C̄a, σa = coarsen(ℓ_kk, A; edges=proc_edges); σa .*= σ_scale_sim
        _, C̄x, σx = coarsen(ℓ_kk, X; edges=proc_edges); σx .*= σ_scale_sim
        σ_th_a = @. sqrt(2 / Neff) * abs(C̄a)
        σ_th_x = @. sqrt(max(abs(C̄_true)*abs(C̄a) + C̄x^2, 0.0) / Neff)
        C̄a, σa, σ_th_a, C̄x, σx, σ_th_x
    end

    C̄_aqe_wf, σ_aqe_wf, σ_th_aqe_wf, C̄_xqe_wf, σ_xqe_wf, σ_th_xqe_wf = proc(Cl_auto_qe_wf_sims, Cl_cross_qe_wf_sims)
    C̄_aqe_ul, σ_aqe_ul, σ_th_aqe_ul, C̄_xqe_ul, σ_xqe_ul, σ_th_xqe_ul = proc(Cl_auto_qe_ul_sims, Cl_cross_qe_ul_sims)
    C̄_agi_b,  σ_agi_b,  σ_th_agi_b,  C̄_xgi_b,  σ_xgi_b,  σ_th_xgi_b  = proc(Cl_auto_gi_b_sims,  Cl_cross_gi_b_sims)
    C̄_agi_c,  σ_agi_c,  σ_th_agi_c,  C̄_xgi_c,  σ_xgi_c,  σ_th_xgi_c  = proc(Cl_auto_gi_c_sims,  Cl_cross_gi_c_sims)
    C̄_amj,    σ_amj,    σ_th_amj,    C̄_xmj,    σ_xmj,    σ_th_xmj    = proc(Cl_auto_mj_sims,    Cl_cross_mj_sims)

    has_map_data   = !isempty(Cl_auto_mj_sims)
    has_qe_ul_data = !isempty(Cl_auto_qe_ul_sims)

    println("\n  ΔL=$(Int(ΔL)), empirical σ (rescaled to f_sky=0.4):")
    println("  L       σ_QE_WF    σ_GI_B      σ_GI_C      σ_MAP_J")
    for b in eachindex(Lc)
        isnan(Lc[b]) && continue
        @printf "  L=%-5d %9.2e %9.2e %9.2e %9.2e\n" round(Int,Lc[b]) σ_aqe_wf[b] σ_agi_b[b] σ_agi_c[b] σ_amj[b]
    end

    # ── SNR table (from simulation covariance) ────────────────────────────────
    Lc_edges      = [0.5*(proc_edges[b]+proc_edges[b+1]) for b in 1:(length(proc_edges)-1)]
    T_bands       = bands_per_sim(ℓ_kk, Cl_true_sims,        proc_edges)
    B_auto_gi_b   = bands_per_sim(ℓ_kk, Cl_auto_gi_b_sims,   proc_edges)
    B_auto_gi_c   = bands_per_sim(ℓ_kk, Cl_auto_gi_c_sims,   proc_edges)
    B_auto_qe_wf  = bands_per_sim(ℓ_kk, Cl_auto_qe_wf_sims,  proc_edges)
    B_auto_qe_ul  = bands_per_sim(ℓ_kk, Cl_auto_qe_ul_sims,  proc_edges)
    B_cross_gi_b  = bands_per_sim(ℓ_kk, Cl_cross_gi_b_sims,  proc_edges)
    B_cross_gi_c  = bands_per_sim(ℓ_kk, Cl_cross_gi_c_sims,  proc_edges)
    B_cross_qe_wf = bands_per_sim(ℓ_kk, Cl_cross_qe_wf_sims, proc_edges)
    B_cross_qe_ul = bands_per_sim(ℓ_kk, Cl_cross_qe_ul_sims, proc_edges)
    B_auto_mj     = bands_per_sim(ℓ_kk, Cl_auto_mj_sims,     proc_edges)
    B_cross_mj    = bands_per_sim(ℓ_kk, Cl_cross_mj_sims,    proc_edges)
    snr_a_gi_b   = snr_compute(B_auto_gi_b,   T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_a_gi_c   = snr_compute(B_auto_gi_c,   T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_a_qe_wf  = snr_compute(B_auto_qe_wf,  T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_a_qe_ul  = snr_compute(B_auto_qe_ul,  T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_x_gi_b   = snr_compute(B_cross_gi_b,  T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_x_gi_c   = snr_compute(B_cross_gi_c,  T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_x_qe_wf  = snr_compute(B_cross_qe_wf, T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_x_qe_ul  = snr_compute(B_cross_qe_ul, T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_a_mj     = snr_compute(B_auto_mj,     T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    snr_x_mj     = snr_compute(B_cross_mj,    T_bands, Lc_edges, snr_L_lo, snr_L_hi, σ_scale_sim)
    _sf(x) = isnan(x) ? "     -" : @sprintf("%6.1f", x)
    println("\n  SNR (L=$(Int(snr_L_lo))-$(Int(snr_L_hi)), from sims):")
    println("  $(rpad("QE (unl, WF)", 16))  Auto=$(_sf(snr_a_qe_wf))  Cross=$(_sf(snr_x_qe_wf))")
    has_qe_ul_data && println("  $(rpad("QE unlensed wts", 16))  Auto=$(_sf(snr_a_qe_ul))  Cross=$(_sf(snr_x_qe_ul))")
    !isempty(Cl_auto_gi_b_sims) && println("  $(rpad("GI Boryana",       16))  Auto=$(_sf(snr_a_gi_b))  Cross=$(_sf(snr_x_gi_b))")
    !isempty(Cl_auto_gi_c_sims) && println("  $(rpad("GI corrected",      16))  Auto=$(_sf(snr_a_gi_c))  Cross=$(_sf(snr_x_gi_c))")
    has_map_data   && println("  $(rpad("MAP joint",        16))  Auto=$(_sf(snr_a_mj))  Cross=$(_sf(snr_x_mj))")

    return (Lc=Lc, xlim=xlim_plot, snr_L_range=(snr_L_lo, snr_L_hi), Lc_edges=Lc_edges,
            C̄_xgi_b=C̄_xgi_b,   σ_xgi_b=σ_xgi_b,    σ_th_xgi_b=σ_th_xgi_b,
            C̄_agi_b=C̄_agi_b,   σ_agi_b=σ_agi_b,    σ_th_agi_b=σ_th_agi_b,
            C̄_xgi_c=C̄_xgi_c,   σ_xgi_c=σ_xgi_c,    σ_th_xgi_c=σ_th_xgi_c,
            C̄_agi_c=C̄_agi_c,   σ_agi_c=σ_agi_c,    σ_th_agi_c=σ_th_agi_c,
            C̄_xqe_wf=C̄_xqe_wf, σ_xqe_wf=σ_xqe_wf, σ_th_xqe_wf=σ_th_xqe_wf,
            C̄_aqe_wf=C̄_aqe_wf, σ_aqe_wf=σ_aqe_wf, σ_th_aqe_wf=σ_th_aqe_wf,
            C̄_xqe_ul=C̄_xqe_ul, σ_xqe_ul=σ_xqe_ul, σ_th_xqe_ul=σ_th_xqe_ul,
            C̄_aqe_ul=C̄_aqe_ul, σ_aqe_ul=σ_aqe_ul, σ_th_aqe_ul=σ_th_aqe_ul,
            C̄_xmj=C̄_xmj,       σ_xmj=σ_xmj,        σ_th_xmj=σ_th_xmj,
            C̄_amj=C̄_amj,       σ_amj=σ_amj,        σ_th_amj=σ_th_amj,
            C̄_true=C̄_true,
            nsims_qegi=length(qegi_sims), nsims_map=length(Cl_auto_mj_sims),
            has_map=has_map_data, has_qe_ul=has_qe_ul_data,
            has_gi_b=!isempty(Cl_auto_gi_b_sims),
            has_gi_c=!isempty(Cl_auto_gi_c_sims),
            W_gi_b=W_gi_b, W_gi_c=W_gi_c,
            W_qe_wf=(has_wf ? W_qe_wf : nothing),
            W_qe_ul=(has_ul ? W_qe_ul : nothing),
            W_mj=W_mj, ℓ_wl=ℓ_template, ℓ_wl_map=ℓ_template_map,
            B_auto_gi_b=B_auto_gi_b, B_auto_gi_c=B_auto_gi_c,
            B_auto_qe_wf=B_auto_qe_wf,
            B_cross_gi_b=B_cross_gi_b, B_cross_gi_c=B_cross_gi_c,
            B_cross_qe_wf=B_cross_qe_wf,
            B_auto_mj=B_auto_mj, B_cross_mj=B_cross_mj,
            snr=(a_gi_b=snr_a_gi_b, a_gi_c=snr_a_gi_c,
                 a_qe_wf=snr_a_qe_wf, a_qe_ul=snr_a_qe_ul, a_mj=snr_a_mj,
                 x_gi_b=snr_x_gi_b, x_gi_c=snr_x_gi_c,
                 x_qe_wf=snr_x_qe_wf, x_qe_ul=snr_x_qe_ul, x_mj=snr_x_mj))
end

s4 = process_noise_level(
    "results/WL_qe_gi_12000.jld2",
    "results/phi_maps_qe_gi_12000.jld2",
    "S4-like (1µK-arcmin, Lmax=12000)";
    Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
    map_wl_file="results/WL_map_12000.jld2",
    map_phi_file="results/phi_maps_map_12000.jld2",
    proc_edges=collect(4500:Int(ΔL):11500),   # plot 5000-11000
    snr_L_lo=5000.0, snr_L_hi=11000.0,
    xlim_plot=(5000.0, 11000.0))

has_ul = isfile("results/WL_qe_gi_12000_ul.jld2") &&
         isfile("results/phi_maps_qe_gi_12000_ul.jld2")
if has_ul
    ul = process_noise_level(
        "results/WL_qe_gi_12000_ul.jld2",
        "results/phi_maps_qe_gi_12000_ul.jld2",
        "Ultra-low std (0.1µK-arcmin, Lmax=12000)";
        Lmax=12000, beamFWHM=0.3, μKarcminT=0.1,
        proc_edges=collect(4500:Int(ΔL):11500),   # same range as S4
        snr_L_lo=5000.0, snr_L_hi=11000.0,
        xlim_plot=(5000.0, 11000.0))
end

# ── Plot ──────────────────────────────────────────────────────────────────────
const ticker = PythonPlot.matplotlib.ticker

function set_log_ticks(ax, ymin::Real, ymax::Real)
    lo = floor(Int, log10(ymin))
    hi = ceil(Int, log10(ymax))
    decades = [10.0^k for k in lo:hi]
    ax.yaxis.set_major_locator(ticker.FixedLocator(decades))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9)))
    ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation(base=10, labelOnlyBase=true))
    ax.set_ylim(10.0^lo * 0.7, 10.0^hi * 1.5)
end

PythonPlot.rc("font",        family="serif", size=11)
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)
PythonPlot.rc("xtick.minor", width=0.5, size=2.5, visible=true)
PythonPlot.rc("ytick.minor", width=0.5, size=2.5, visible=true)

colours = Dict("gi_o" => "#D62728", "gi_b" => "#1F77B4", "gi_c" => "#E377C2",
               "qe_wf" => "#2CA02C", "qe_ul" => "#FF7F0E", "mj" => "#9467BD")
labels  = Dict("gi_o" => "GI orig (Lhp=4k)", "gi_b" => "GI (Boryana)", "gi_c" => "GI corrected (Wiener)",
               "qe_wf" => "QE (unlensed, WF)", "qe_ul" => "QE unlensed wts", "mj" => "MAP joint")

datasets = Pair{String, Any}[]
push!(datasets, "S4" => s4)
has_ul && push!(datasets, "UL" => ul)
ncols = length(datasets)

fig, axs_arr = PythonPlot.subplots(4, ncols;
    figsize=(6.5*ncols, 14.0), sharex="col", constrained_layout=true)

getax(axs, row, col) = ncols == 1 ? axs[row] : axs[row, col]

function fill_sigma_panel(ax, data, is_auto; title_str="", show_legend=false, sigma_ymin=nothing)
    Lc = data.Lc
    entries = is_auto ?
        [("gi_b", data.σ_agi_b, data.σ_th_agi_b),
         ("gi_c", data.σ_agi_c, data.σ_th_agi_c), ("qe_wf", data.σ_aqe_wf, data.σ_th_aqe_wf),
         ("qe_ul", data.σ_aqe_ul, data.σ_th_aqe_ul)] :
        [("gi_b", data.σ_xgi_b, data.σ_th_xgi_b),
         ("gi_c", data.σ_xgi_c, data.σ_th_xgi_c), ("qe_wf", data.σ_xqe_wf, data.σ_th_xqe_wf),
         ("qe_ul", data.σ_xqe_ul, data.σ_th_xqe_ul)]
    for (key, σ_sim, σ_th) in entries
        msk = @. !isnan(Lc) & isfinite(σ_sim) & (σ_sim > 0)
        !any(msk) && continue
        ax.semilogy(Lc[msk], σ_sim[msk]; color=colours[key], linewidth=2.0, label=labels[key])
        msk_th = @. !isnan(Lc) & isfinite(σ_th) & (σ_th > 0)
        any(msk_th) && ax.semilogy(Lc[msk_th], σ_th[msk_th];
            color=colours[key], linestyle="--", linewidth=1.2, alpha=0.7)
    end
    if data.has_map
        σ_sym    = is_auto ? data.σ_amj    : data.σ_xmj
        σ_th_sym = is_auto ? data.σ_th_amj : data.σ_th_xmj
        msk = @. !isnan(Lc) & isfinite(σ_sym) & (σ_sym > 0)
        if any(msk)
            ax.semilogy(Lc[msk], σ_sym[msk]; color=colours["mj"], linewidth=2.0, label=labels["mj"])
            msk_th = @. !isnan(Lc) & isfinite(σ_th_sym) & (σ_th_sym > 0)
            any(msk_th) && ax.semilogy(Lc[msk_th], σ_th_sym[msk_th];
                color=colours["mj"], linestyle="--", linewidth=1.2, alpha=0.7)
        end
    end
    ax.set_xlim(data.xlim...)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    if is_auto
        ylo = sigma_ymin !== nothing ? sigma_ymin : 1e-13
        set_log_ticks(ax, ylo, 1e-9)
    else
        _σ_all = vcat([filter(v -> isfinite(v) && v > 0, σ) for (_, σ, _) in entries]...)
        raw_ylo = isempty(_σ_all) ? 1e-13 : minimum(_σ_all)
        ylo = sigma_ymin !== nothing ? max(raw_ylo, sigma_ymin) : raw_ylo
        set_log_ticks(ax, ylo, isempty(_σ_all) ? 1e-10 : maximum(_σ_all))
    end
    isempty(title_str) || ax.set_title(title_str, fontsize=9)
    show_legend && ax.legend(loc="upper left", frameon=false, fontsize=9,
        title="solid=empirical, dashed=Knox", title_fontsize=7)
end

function fill_mean_cross_panel(ax, data; show_legend=false)
    Lc = data.Lc
    # True signal
    msk_t = @. !isnan(Lc) & isfinite(data.C̄_true) & (data.C̄_true > 0)
    any(msk_t) && ax.semilogy(Lc[msk_t], data.C̄_true[msk_t];
        color="black", linestyle="--", linewidth=1.8, label=L"C_L^{\kappa\kappa}\,(true)")
    # Cross-spectra (solid) and auto-spectra (dotted) for each estimator
    cross_entries = [("gi_b", data.C̄_xgi_b, data.C̄_agi_b),
                     ("qe_wf", data.C̄_xqe_wf, data.C̄_aqe_wf)]
    data.has_map && push!(cross_entries, ("mj", data.C̄_xmj, data.C̄_amj))
    for (key, C̄_x, C̄_a) in cross_entries
        msk_x = @. !isnan(Lc) & isfinite(C̄_x) & (abs(C̄_x) > 0)
        msk_a = @. !isnan(Lc) & isfinite(C̄_a) & (abs(C̄_a) > 0)
        if any(msk_x)
            ax.semilogy(Lc[msk_x], abs.(C̄_x[msk_x]);
                color=colours[key], linewidth=2.0, label=labels[key]*" (cross)")
        end
        if any(msk_a)
            ax.semilogy(Lc[msk_a], abs.(C̄_a[msk_a]);
                color=colours[key], linestyle=":", linewidth=1.5,
                label=labels[key]*" (auto)")
        end
    end
    ax.set_xlim(data.xlim...)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    _c_all = Float64[]
    for (_, C̄_x, C̄_a) in cross_entries
        append!(_c_all, filter(v -> isfinite(v) && v > 0, abs.(C̄_x)))
        append!(_c_all, filter(v -> isfinite(v) && v > 0, abs.(C̄_a)))
    end
    any(msk_t) && append!(_c_all, data.C̄_true[msk_t])
    set_log_ticks(ax, isempty(_c_all) ? 1e-13 : minimum(_c_all),
                      isempty(_c_all) ? 1e-10 : maximum(_c_all))
    show_legend && ax.legend(loc="upper right", frameon=false, fontsize=7,
                             ncol=2)
end

function fill_WL_panel(ax, data; show_legend=false)
    ℓ = data.ℓ_wl
    xlo, xhi = data.xlim
    for (key, W) in [("gi_b", data.W_gi_b), ("gi_c", data.W_gi_c),
                      ("qe_wf", data.W_qe_wf), ("qe_ul", data.W_qe_ul)]
        W === nothing && continue
        msk = @. !isnan(ℓ) & isfinite(W) & (ℓ >= xlo) & (ℓ <= xhi)
        !any(msk) && continue
        ax.plot(ℓ[msk], W[msk]; color=colours[key], linewidth=2.0, label=labels[key])
    end
    if data.has_map && data.W_mj !== nothing
        W = data.W_mj; ℓm = data.ℓ_wl_map
        msk = @. !isnan(ℓm) & isfinite(W) & (ℓm >= xlo) & (ℓm <= xhi)
        any(msk) && ax.plot(ℓm[msk], W[msk]; color=colours["mj"], linewidth=2.0, label=labels["mj"])
    end
    ax.axhline(1.0; color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0; color="k",    linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(data.xlim...)
    ax.set_ylim(-0.2, 1.5)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    show_legend && ax.legend(loc="upper right", frameon=false, fontsize=9)
end

# Column titles
col_titles = Dict(
    "S4" => "1 µK-arcmin",
    "UL" => "0.1 µK-arcmin",
)

for (col, (lbl, d)) in enumerate(datasets)
    col0      = col - 1   # 0-based for getax
    is_first  = col == 1
    nsims_str = "$(d.nsims_qegi) sims" * (d.has_map ? ", MAP: $(d.nsims_map) sims" : "")
    title_str = "$(get(col_titles, lbl, lbl))  ($nsims_str)"

    ymin_ul = lbl == "UL" ? 1e-13 : nothing
    fill_sigma_panel(getax(axs_arr, 0, col0), d, true;  title_str=title_str, show_legend=is_first, sigma_ymin=ymin_ul)
    fill_sigma_panel(getax(axs_arr, 1, col0), d, false; show_legend=false, sigma_ymin=ymin_ul)
    fill_mean_cross_panel(getax(axs_arr, 2, col0), d; show_legend=is_first)
    fill_WL_panel(getax(axs_arr, 3, col0), d; show_legend=is_first)
    getax(axs_arr, 3, col0).set_xlabel(L"L", fontsize=12)
end

# y-axis labels on leftmost column only
getax(axs_arr, 0, 0).set_ylabel(L"\sigma[C_L^{\hat{\kappa}\hat{\kappa}}]", fontsize=12)
getax(axs_arr, 1, 0).set_ylabel(L"\sigma[C_L^{\kappa\hat{\kappa}}]", fontsize=12)
getax(axs_arr, 2, 0).set_ylabel(L"\bar{C}_L^{\kappa\kappa},\,\bar{C}_L^{\kappa\hat{\kappa}}", fontsize=12)
getax(axs_arr, 3, 0).set_ylabel(
    L"W_L = \left\langle C_L^{\phi_{\rm true},\hat\phi} / C_L^{\phi_{\rm true},\phi_{\rm true}} \right\rangle",
    fontsize=10)

# ── SNR summary table ─────────────────────────────────────────────────────────
println("\n" * "="^80)
_sf2(x) = isnan(x) ? "         -" : @sprintf("%9.1f", x)
println("SNR Table  (f_sky=0.4, from sims)")
println("="^80)
println("                              Cross QE(L)  Cross QE(UL)  Cross GI_B  Cross GI_C  Cross MAP   Auto QE(L)  Auto QE(UL)  Auto GI_B  Auto GI_C  Auto MAP")
function print_snr_row(io, tag, qe_snr, gi_snr)
    println(io, "$tag  $(_sf2(qe_snr.x_qe_wf))  $(_sf2(qe_snr.x_qe_ul))  $(_sf2(gi_snr.x_gi_b))  $(_sf2(gi_snr.x_gi_c))  $(_sf2(gi_snr.x_mj))  $(_sf2(qe_snr.a_qe_wf))  $(_sf2(qe_snr.a_qe_ul))  $(_sf2(gi_snr.a_gi_b))  $(_sf2(gi_snr.a_gi_c))  $(_sf2(gi_snr.a_mj))")
end
for (lbl, d) in datasets
    lo, hi = Int.(d.snr_L_range)
    tag = rpad("$lbl (L=$lo-$hi)", 30)
    print_snr_row(stdout, tag, d.snr, d.snr)
end
println("="^80)
println("Paper (UL/S4/SO)  Cross QE: 710/550/195  Cross GI: 4100/1440/270  Auto QE: 205/100/7  Auto GI: 1515/360/30")

# Save SNR table to file
open("results/snr_table.txt", "w") do io
    println(io, "SNR Table  (f_sky=0.4, from sims)")
    println(io, "="^80)
    println(io, "                              Cross QE(L)  Cross QE(UL)  Cross GI_B  Cross GI_C  Cross MAP   Auto QE(L)  Auto QE(UL)  Auto GI_B  Auto GI_C  Auto MAP")
    for (lbl, d) in datasets
        lo, hi = Int.(d.snr_L_range)
        tag = rpad("$lbl (L=$lo-$hi)", 30)
        print_snr_row(io, tag, d.snr, d.snr)
    end
    println(io, "="^80)
    println(io, "Paper (UL/S4/SO)  Cross QE: 710/550/195  Cross GI: 4100/1440/270  Auto QE: 205/100/7  Auto GI: 1515/360/30")
end
println("Saved results/snr_table.txt")

savepath = "results/qe_gi_map_sigma_12000.png"
fig.savefig(savepath; dpi=200)
println("\nSaved $savepath")
PythonPlot.plotclose("all")

# ── Covariance correlation matrices ───────────────────────────────────────────
# ρ_ij = Cov(C_Li, C_Lj) / (σ_i σ_j)  — shows off-diagonal bandpower correlations.
# GI should have lower off-diagonal ρ than QE (less mode coupling).
function corr_matrix(B::Matrix)
    nb, ns = size(B)
    ns < 3 && return fill(NaN, nb, nb)
    A_c = B .- mean(B; dims=2)
    Cov = (A_c * A_c') ./ (ns - 1)
    σ = sqrt.(abs.(diag(Cov)))
    σ[σ .< 1e-100] .= 1.0
    return Cov ./ (σ * σ')
end

function fill_corr_ax(ax, B::Matrix, Lc_edges::Vector; title="")
    # Keep only bands with a valid label, all-finite values, and nonzero variance
    valid = findall(i -> !isnan(Lc_edges[i]) &&
                         all(isfinite.(B[i, :])) &&
                         std(B[i, :]) > 1e-30,
                    1:min(size(B,1), length(Lc_edges)))
    isempty(valid) && return
    C = corr_matrix(B[valid, :])
    Lv = Lc_edges[valid]
    im = ax.imshow(C; cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                   extent=[Lv[1], Lv[end], Lv[1], Lv[end]], aspect="equal",
                   interpolation="nearest")
    ax.set_xlabel(L"L'", fontsize=9); ax.set_ylabel(L"L", fontsize=9)
    ax.set_title(title, fontsize=8)
    return im
end

datasets_cov = datasets  # reuse same ordered list

# 3 panels per dataset: GI, QE, MAP (MAP hidden if not available)
n_cov_per = 3
n_cov_cols = n_cov_per * length(datasets_cov)
fig_cov, axs_cov = PythonPlot.subplots(2, n_cov_cols;
    figsize=(4.5 * n_cov_cols, 9.0), constrained_layout=true)
getac(r, c) = n_cov_cols == 1 ? axs_cov[r] : axs_cov[r, c]

for (col_pair, (lbl, d)) in enumerate(datasets_cov)
    c_gi = n_cov_per*(col_pair-1)
    c_qe = n_cov_per*(col_pair-1) + 1
    c_mj = n_cov_per*(col_pair-1) + 2
    for (row, spec) in enumerate(["auto", "cross"])
        row0 = row - 1
        B_gi = spec == "auto" ? d.B_auto_gi_b  : d.B_cross_gi_b
        B_qe = spec == "auto" ? d.B_auto_qe_wf : d.B_cross_qe_wf
        B_mj = spec == "auto" ? d.B_auto_mj    : d.B_cross_mj
        im_gi = fill_corr_ax(getac(row0, c_gi), B_gi, d.Lc_edges; title="$lbl  GI $spec")
        im_qe = fill_corr_ax(getac(row0, c_qe), B_qe, d.Lc_edges; title="$lbl  QE $spec")
        im_gi !== nothing && fig_cov.colorbar(im_gi; ax=getac(row0, c_gi), fraction=0.046, pad=0.04)
        im_qe !== nothing && fig_cov.colorbar(im_qe; ax=getac(row0, c_qe), fraction=0.046, pad=0.04)
        if d.has_map
            im_mj = fill_corr_ax(getac(row0, c_mj), B_mj, d.Lc_edges; title="$lbl  MAP joint $spec")
            im_mj !== nothing && fig_cov.colorbar(im_mj; ax=getac(row0, c_mj), fraction=0.046, pad=0.04)
        else
            getac(row0, c_mj).set_visible(false)
        end
    end
end
fig_cov.suptitle(L"Bandpower correlation  $\rho_{LL'} = \mathrm{Cov}(C_L,C_{L'}) / (\sigma_L\,\sigma_{L'})$",
    fontsize=11)
cov_path = "results/covariance_correlation.png"
fig_cov.savefig(cov_path; dpi=200)
println("Saved $cov_path")
PythonPlot.plotclose("all")
