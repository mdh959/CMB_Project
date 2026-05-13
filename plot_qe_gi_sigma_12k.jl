#!/usr/bin/env julia
"""
plot_qe_gi_sigma_12k.jl — reproduces figures from Hadzhiyska et al. (2019)

Debiasing strategy for each estimator:
  QE  auto:  C_auto = [C(ϕ̂_raw) − ⟨N^(0)⟩] / W_QE²(ℓ)   (spectrum-level; N^(0) = mean RDN0)
  QE  cross: C_cross = C(ϕ_true, ϕ̂_deb)                  (field-level; ϕ̂_deb = ϕ̂ / W_QE)
  GI  auto:  C_auto  = [C(ϕ̂_raw) − ⟨N^(0)⟩] / W_GI²(ℓ)   (spectrum-level; N^(0) = mean MC N0)
  GI  cross: C_cross = C(ϕ_true, ϕ̂_deb)
  MAP:       same pattern as GI (field-level debiasing only)

W_L is smoothed with a 9-bin running mean before applying to reduce noise scatter.
Bandpower width ΔL=2000 matches Hadzhiyska+2019; bin centres at [4000,6000,...,12000].
σ is rescaled from the simulation patch to the paper's f_sky=0.4 survey area.
Output: results/Boryana's paper/
"""

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using Statistics: mean, std, median, quantile
using PythonPlot
using Printf
using JLD2

include("utils.jl")
using .Utils

# ── Fiducial model ─────────────────────────────────────────────────────────────
const Cℓ_theory   = camb(r=0.05, ℓmax=170000)  # high-ℓ CAMB run for accurate theory curves

# ── Patch geometry ─────────────────────────────────────────────────────────────
const θpix        = 0.7438046267475303           # pixel width (arcmin)
const Nside       = 512
const pol         = :I
const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π) # sky fraction of the simulated patch
const f_sky_paper = 0.4                          # Hadzhiyska+2019 survey sky fraction;
                                                  # σ is rescaled as √(f_sky_sim/f_sky_paper)

# ── Debiasing floors ──────────────────────────────────────────────────────────
const minW        = 1e-8   # field-level φ debiasing: modes with W_L < minW are zeroed
const minW_auto   = 0.5    # spectrum-level floor for W_L² (QE); W_QE ≈ 1 analytically,
                           # values < 0.5 are noise artefacts that would catastrophically
                           # amplify the QE auto-spectrum estimate
const minW_auto_gi = 0.2   # spectrum-level floor for GI W_L²; limits 1/W² amplification
                           # to ≤25×; bins where W_GI < 0.2 are unreliable and masked

# ── Bandpower binning ─────────────────────────────────────────────────────────
const ΔL          = 2000   # bin width, matching Hadzhiyska+2019
const Δℓ_spec     = 30     # fine ℓ-bin for per-sim spectra before coarsening to ΔL
const proc_edges  = collect(3000.0:ΔL:13001.0)   # centres at [4000,6000,...,12000]
const OUT_DIR     = "results/Boryana's paper"

mkpath(OUT_DIR)
println("f_sky_patch = $(round(f_sky_patch; sigdigits=4))")
println("proc_edges  = $(proc_edges)")

# helpers
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

# main processing function
function process_noise_level(WL_file, phi_maps_file, label;
                             Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
                             map_wl_file=nothing, map_phi_file=nothing,
                             snr_L_lo=5000.0, snr_L_hi=11000.0,
                             xlim_plot=(4000.0, 12000.0),
                             θpix_sim::Float64=θpix, Nside_sim::Int=Nside,
                             exclude_sims::Set{Int}=Set{Int}(),
                             skip_rms_outliers::Bool=false)
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

    # Per-sim accumulators
    Cl_auto_qe_sims      = Vector{Vector{Float64}}()
    Cl_auto_qe_full_sims = Vector{Vector{Float64}}()
    Cl_auto_qe_rdn0_sims = Vector{Vector{Float64}}()
    Cl_cross_qe_sims     = Vector{Vector{Float64}}()
    Cl_auto_gi_sims      = Vector{Vector{Float64}}()
    Cl_auto_gi_full_sims = Vector{Vector{Float64}}()
    Cl_auto_gi_fgmc_sims = Vector{Vector{Float64}}()
    Cl_auto_gi_rdn0_sims = Vector{Vector{Float64}}()
    Cl_auto_gi_linrd_sims = Vector{Vector{Float64}}()
    Cl_cross_gi_sims     = Vector{Vector{Float64}}()
    Cl_true_sims         = Vector{Vector{Float64}}()
    ρ_qe_sims            = Vector{Vector{Float64}}()
    ρ_gi_sims            = Vector{Vector{Float64}}()
    ρ_mj_gi_sims         = Vector{Vector{Float64}}()   # ρ(ϕ_MAP_deb, ϕ_GI_deb) per sim
    ℓ_kk                 = nothing

    qegi_sims  = Int[]
    mean_N0_qe = nothing

    jldopen(phi_maps_file, "r") do f
        # 1. Scan sim keys
        for key in keys(f)
            m = match(r"^sim_(\d+)$", key)
            m !== nothing && push!(qegi_sims, parse(Int, m.captures[1]))
        end
        sort!(qegi_sims)

        # 2. RMS outlier exclusion (exclude sims where QE pixel RMS > 15× median)
        if has_qe && !skip_rms_outliers
            rms_vals = Dict{Int,Float64}()
            for s in qegi_sims
                haskey(f, "sim_$s/ϕ_qe_raw") || continue
                rms_vals[s] = sqrt(mean(Map(wrap(read(f, "sim_$s/ϕ_qe_raw"))).arr .^ 2))
            end
            if !isempty(rms_vals)
                med_rms = median(values(rms_vals))
                thresh  = 15.0 * med_rms
                rms_bad = Set(s for (s, r) in rms_vals if r > thresh)
                if !isempty(rms_bad)
                    println("  RMS outliers (QE RMS > 15×median=$(round(thresh,sigdigits=3))): $(sort(collect(rms_bad)))")
                    filter!(s -> s ∉ rms_bad, qegi_sims)
                end
            end
        elseif skip_rms_outliers
            println("  RMS outlier exclusion skipped (skip_rms_outliers=true)")
        end
        if !isempty(exclude_sims)
            filter!(s -> s ∉ exclude_sims, qegi_sims)
            println("  Additional excluded sims: $(sort(collect(exclude_sims)))")
        end

        # 3. Accumulate mean QE RDN0 (lower scatter than per-sim at UL noise)
        if has_qe
            ℓ_kk_tmp = nothing; N0_accum = nothing; N0_count = 0
            for s in qegi_sims
                haskey(f, "sim_$s/ϕ_qe_raw") || continue
                if ℓ_kk_tmp === nothing
                    cl = get_Cℓ(wrap(read(f, "sim_$s/ϕ_qe_raw")); Δℓ=Δℓ_spec)
                    ℓ_kk_tmp = Float64.(collect(cl.ℓ))
                end
                haskey(f, "sim_$s/N0_rdn0") || continue
                N0_stored = Float64.(read(f, "sim_$s/N0_rdn0"))
                N0_safe   = [isfinite(v) && v >= 0.0 ? v : 0.0 for v in N0_stored]
                n_n0 = length(N0_safe); n_kk = length(ℓ_kk_tmp)
                N0_this = if n_n0 == n_kk
                    N0_safe
                elseif n_n0 > 0
                    ℓkk_cl = clamp.(ℓ_kk_tmp, ℓ_template[1], ℓ_template[end])
                    max.(Float64.(Cℓs(ℓ_template, N0_safe).(ℓkk_cl)), 0.0)
                else
                    nothing
                end
                N0_this === nothing && continue
                N0_accum = N0_accum === nothing ? copy(N0_this) : N0_accum .+ N0_this
                N0_count += 1
            end
            if N0_count > 0
                println("  Mean QE RDN0 from $N0_count sims")
                mean_N0_qe = N0_accum ./ N0_count
            end
        end

        # 4. Main per-sim spectrum loop
        for (i, s) in enumerate(qegi_sims)
            ϕt = wrap(read(f, "sim_$s/ϕ_true"))
            cl_tt = get_Cℓ(ϕt; Δℓ=Δℓ_spec)
            ℓ_kk === nothing && (ℓ_kk = Float64.(collect(cl_tt.ℓ)))
            kfac = @. (ℓ_kk^2 / 2)^2
            push!(Cl_true_sims, kfac .* Float64.(cl_tt.Cℓ))

            if has_qe && haskey(f, "sim_$s/ϕ_qe_raw")
                ϕr     = wrap(read(f, "sim_$s/ϕ_qe_raw"))
                ϕr_deb = debias_phi_with_WL(ϕr, WL_qe; minW=minW)
                cl_aa  = get_Cℓ(ϕr_deb; Δℓ=Δℓ_spec)          # C(ϕ̂/W_L, ϕ̂/W_L)
                cl_xa  = get_Cℓ(ϕt, ϕr_deb; Δℓ=Δℓ_spec)
                Cl_aa_vals = Float64.(cl_aa.Cℓ)                # already debiased
                W_at_ℓkk   = Float64.(WL_qe.(ℓ_kk))
                W2_floor    = @. max(W_at_ℓkk^2, minW_auto^2)
                push!(Cl_auto_qe_full_sims, kfac .* Cl_aa_vals)
                # RDN0: N0 is in raw estimator units → divide by W² before subtracting
                if haskey(f, "sim_$s/N0_rdn0")
                    N0_ps = Float64.(read(f, "sim_$s/N0_rdn0"))
                    N0_ps = [isfinite(v) && v >= 0.0 ? v : 0.0 for v in N0_ps]
                    if length(N0_ps) == length(ℓ_kk)
                        push!(Cl_auto_qe_rdn0_sims,
                              kfac .* (@. Cl_aa_vals - N0_ps / W2_floor))
                    end
                end
                N0_mean_deb = (mean_N0_qe !== nothing && length(mean_N0_qe) == length(ℓ_kk)) ?
                              mean_N0_qe ./ W2_floor : zeros(length(ℓ_kk))
                push!(Cl_auto_qe_sims, kfac .* (Cl_aa_vals .- N0_mean_deb))
                push!(Cl_cross_qe_sims, kfac .* Float64.(cl_xa.Cℓ))
                denom = sqrt.(max.(Float64.(cl_tt.Cℓ) .* max.(Cl_aa_vals, 0.0), 0.0))
                push!(ρ_qe_sims, clamp.(Float64.(cl_xa.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
            end

            if has_gi && haskey(f, "sim_$s/ϕ_gi_b")
                ϕg_raw = wrap(read(f, "sim_$s/ϕ_gi_b"))
                ϕg_deb = debias_phi_with_WL(ϕg_raw, WL_gi; minW=minW)
                cl_gi_a      = get_Cℓ(ϕg_deb; Δℓ=Δℓ_spec)    # C(ϕ̂/W_L, ϕ̂/W_L)
                cl_gi_a_vals = Float64.(cl_gi_a.Cℓ)            # already debiased
                W_gi_at_ℓkk  = Float64.(WL_gi.(ℓ_kk))
                W2_gi_floor  = @. max(W_gi_at_ℓkk^2, minW_auto_gi^2)
                push!(Cl_auto_gi_full_sims, kfac .* cl_gi_a_vals)
                push!(Cl_auto_gi_sims,      kfac .* cl_gi_a_vals)
                # N0 methods: stored in raw estimator units → divide by W² before subtracting
                if haskey(f, "sim_$s/N0_gi_fgmc")
                    N0_fgmc_ell  = Float64.(read(f, "sim_$s/N0_gi_fgmc_ell"))
                    N0_fgmc_raw  = Float64.(read(f, "sim_$s/N0_gi_fgmc"))
                    N0_fgmc_safe = [isfinite(v) && v >= 0.0 ? v : 0.0 for v in N0_fgmc_raw]
                    itp_fg       = Cℓs(N0_fgmc_ell, N0_fgmc_safe)
                    N0_fg_at_kk  = [ℓ < N0_fgmc_ell[1] || ℓ > N0_fgmc_ell[end] ? 0.0 :
                                    max(Float64(itp_fg(ℓ)), 0.0) for ℓ in ℓ_kk]
                    push!(Cl_auto_gi_fgmc_sims,
                          kfac .* (@. cl_gi_a_vals - N0_fg_at_kk / W2_gi_floor))
                end
                if haskey(f, "sim_$s/N0_gi_rdn0_v2")
                    N0_rd_ell   = Float64.(read(f, "sim_$s/N0_gi_rdn0_v2_ell"))
                    N0_rd_raw   = Float64.(read(f, "sim_$s/N0_gi_rdn0_v2"))
                    itp_rd      = Cℓs(N0_rd_ell, N0_rd_raw)
                    N0_rd_at_kk = [ℓ < N0_rd_ell[1] || ℓ > N0_rd_ell[end] ? 0.0 :
                                   Float64(itp_rd(ℓ)) for ℓ in ℓ_kk]
                    push!(Cl_auto_gi_rdn0_sims,
                          kfac .* (@. cl_gi_a_vals - N0_rd_at_kk / W2_gi_floor))
                end
                if haskey(f, "sim_$s/N0_gi_linrd")
                    N0_lr_ell   = Float64.(read(f, "sim_$s/N0_gi_linrd_ell"))
                    N0_lr_raw   = Float64.(read(f, "sim_$s/N0_gi_linrd"))
                    itp_lr      = Cℓs(N0_lr_ell, N0_lr_raw)
                    N0_lr_at_kk = [ℓ < N0_lr_ell[1] || ℓ > N0_lr_ell[end] ? 0.0 :
                                   Float64(itp_lr(ℓ)) for ℓ in ℓ_kk]
                    push!(Cl_auto_gi_linrd_sims,
                          kfac .* (@. cl_gi_a_vals - N0_lr_at_kk / W2_gi_floor))
                end
                cl_gi_x = get_Cℓ(ϕt, ϕg_deb; Δℓ=Δℓ_spec)
                push!(Cl_cross_gi_sims, kfac .* Float64.(cl_gi_x.Cℓ))
                denom = sqrt.(max.(Float64.(cl_tt.Cℓ) .* max.(cl_gi_a_vals, 0.0), 0.0))
                push!(ρ_gi_sims, clamp.(Float64.(cl_gi_x.Cℓ) ./ max.(denom, 1e-30), -1.0, 1.0))
            end
            print("\r  QE/GI sim $i/$(length(qegi_sims))"); flush(stdout)
        end
    end
    println()
    println("  $(length(qegi_sims)) QE/GI sims  |  debiasing method counts:")
    println("    QE RDN0  : $(length(Cl_auto_qe_rdn0_sims)) sims")
    println("    GI fg-MC : $(length(Cl_auto_gi_fgmc_sims)) sims")
    println("    GI RDN0  : $(length(Cl_auto_gi_rdn0_sims)) sims")
    println("    GI lin-RD: $(length(Cl_auto_gi_linrd_sims)) sims")
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
            # ── RMS-based outlier exclusion for MAP sims ───────────────────────
            # Same logic as QE: exclude sims where ϕ_mj pixel RMS > 15 × median.
            # Runaway MAP solutions (shell-crossing divergence) have RMS ~1000×
            # larger than converged sims and would otherwise dominate σ.
            map_rms_vals = Dict{Int,Float64}()
            for s in map_sims
                haskey(f, "sim_$s/ϕ_mj") || continue
                map_rms_vals[s] = sqrt(mean(Map(wrap(read(f, "sim_$s/ϕ_mj"))).arr .^ 2))
            end
            if !isempty(map_rms_vals)
                med_rms_mj = median(values(map_rms_vals))
                thresh_mj  = 15.0 * med_rms_mj
                rms_bad_mj = Set(s for (s, r) in map_rms_vals if r > thresh_mj)
                if !isempty(rms_bad_mj)
                    println("  MAP RMS outliers (ϕ_mj RMS > 15×median=$(round(thresh_mj,sigdigits=3))): $(sort(collect(rms_bad_mj)))")
                    filter!(s -> s ∉ rms_bad_mj, map_sims)
                end
            end
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
                # ρ(ϕ_MAP_deb, ϕ_GI_deb): ρ is invariant to per-mode W_L rescaling,
                # so this equals ρ of the debiased fields.
                if has_gi
                    jldopen(phi_maps_file, "r") do fqegi
                        if haskey(fqegi, "sim_$s/ϕ_gi_b")
                            ϕg_r  = wrap(read(fqegi, "sim_$s/ϕ_gi_b"))
                            ϕg_d  = debias_phi_with_WL(ϕg_r, WL_gi; minW=minW)
                            cl_mg = get_Cℓ(ϕm_d, ϕg_d; Δℓ=Δℓ_spec)
                            cl_g  = get_Cℓ(ϕg_d; Δℓ=Δℓ_spec)
                            denom_mg = sqrt.(max.(Float64.(cl_a.Cℓ) .* Float64.(cl_g.Cℓ), 0.0))
                            push!(ρ_mj_gi_sims, clamp.(Float64.(cl_mg.Cℓ) ./ max.(denom_mg, 1e-30), -1.0, 1.0))
                        end
                    end
                end
                print("\r  MAP sim $i/$(length(map_sims))"); flush(stdout)
            end
        end
        println(); println("  $(length(Cl_auto_mj_sims)) MAP sims")
    end

    # ── Near-zero outlier diagnostic ──────────────────────────────────────────
    # The 15×median RMS filter catches high-RMS runaways but not near-zero sims
    # (collapsed/non-converged solutions). Check coarse bandpowers for anomalies.
    function _check_auto_outliers(sims, name, edges, ℓv)
        isempty(sims) && return
        M = bands_per_sim(ℓv, sims, edges)  # (nbands × nsims)
        ns = size(M, 2); nb = size(M, 1)
        all_vals = filter(isfinite, vec(M))
        isempty(all_vals) && return
        med_bp = median(all_vals)
        thresh_lo = 0.05 * med_bp   # flag if < 5% of median (catches near-zero)
        bad = [j for j in 1:ns if any(isfinite(M[b,j]) && M[b,j] < thresh_lo for b in 1:nb)]
        println("  [$name] per-sim auto bandpower check: median=$(round(med_bp,sigdigits=3)), " *
                "5%-of-median threshold=$(round(thresh_lo,sigdigits=3))")
        if isempty(bad)
            println("    No near-zero outlier sims detected.")
        else
            println("    Near-zero outlier sim indices: $bad")
            for j in bad
                bp_str = join([@sprintf("%.2e", isfinite(M[b,j]) ? M[b,j] : NaN) for b in 1:nb], ", ")
                println("      sim[$j] bandpowers: [$bp_str]")
            end
        end
        # Also print full per-sim bandpower table for inspection
        println("  [$name] per-sim bandpowers (rows=band, cols=sim):")
        Lc_diag = [0.5*(edges[b]+edges[b+1]) for b in 1:(length(edges)-1)]
        for b in 1:nb
            isnan(Lc_diag[b]) && continue
            row = [@sprintf("%8.2e", isfinite(M[b,j]) ? M[b,j] : NaN) for j in 1:ns]
            println("    L≈$(round(Int,Lc_diag[b])): $(join(row, " "))")
        end
    end
    # σ rescaling: sample variance ∝ 1/N_modes ∝ 1/(f_sky ΔL (2L+1)), so σ ∝ 1/√f_sky.
    # Multiplying by √(f_sky_sim/f_sky_paper) converts simulation σ to the survey area
    # of Hadzhiyska+2019 (f_sky=0.4) for a direct comparison with their figures.
    θpr        = θpix_sim * π / (180 * 60)
    f_sky_sim  = (Nside_sim * θpr)^2 / (4π)
    σ_sc       = sqrt(f_sky_sim / f_sky_paper)

    ℓv = ℓ_kk !== nothing ? ℓ_kk : Float64[]

    println("\n── Auto bandpower outlier diagnostics for: $label ──")
    _check_auto_outliers(Cl_auto_qe_rdn0_sims, "QE RDN0", proc_edges, ℓv)
    _check_auto_outliers(Cl_auto_mj_sims,      "MAP",     proc_edges, ℓv)
    Tmat = reduce(hcat, Cl_true_sims)
    Lc, C̄_true, _ = coarsen(ℓv, Tmat; edges=proc_edges)
    Neff_v = @. (2*Lc + 1) * ΔL * f_sky_paper

    # sa      = N0-subtracted signal auto (plotted curves, SNR)
    # sa_full = full auto including N0, W_L²-debiased (Knox formula denominator)
    # MAP does not subtract N0 so sa_full == sa for MAP
    function proc(sa, sa_full, sx)
        nb = length(Lc)
        z = fill(NaN, nb)
        isempty(sa) && return z,z,z,z,z,z
        A      = reduce(hcat, sa);      X = reduce(hcat, sx)
        A_full = reduce(hcat, sa_full)
        _, C̄a,      σa = coarsen(ℓv, A;      edges=proc_edges); σa .*= σ_sc
        _, C̄a_full, _  = coarsen(ℓv, A_full; edges=proc_edges)
        _, C̄x,      σx = coarsen(ℓv, X;      edges=proc_edges); σx .*= σ_sc
        # Knox formula uses C_total (signal + noise), not the bias-subtracted signal
        σ_th_a = @. sqrt(2 / Neff_v) * abs(C̄a_full)
        σ_th_x = @. sqrt(max(abs(C̄_true)*abs(C̄a_full) + C̄x^2, 0.0) / Neff_v)
        C̄a, σa, σ_th_a, C̄x, σx, σ_th_x
    end

    C̄_a_qe, σ_a_qe, σ_th_a_qe, C̄_x_qe, σ_x_qe, σ_th_x_qe = proc(Cl_auto_qe_rdn0_sims, Cl_auto_qe_full_sims, Cl_cross_qe_sims)
    C̄_a_gi, σ_a_gi, σ_th_a_gi, C̄_x_gi, σ_x_gi, σ_th_x_gi = proc(Cl_auto_gi_sims, Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_gi_fgmc, σ_a_gi_fgmc, σ_th_a_gi_fgmc, _, _, _ = proc(Cl_auto_gi_fgmc_sims, Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_gi_rdn0,  σ_a_gi_rdn0,  σ_th_a_gi_rdn0,  _, _, _ = proc(Cl_auto_gi_rdn0_sims,  Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_gi_linrd, σ_a_gi_linrd, σ_th_a_gi_linrd, _, _, _ = proc(Cl_auto_gi_linrd_sims, Cl_auto_gi_full_sims, Cl_cross_gi_sims)
    C̄_a_mj, σ_a_mj, σ_th_a_mj, C̄_x_mj, σ_x_mj, σ_th_x_mj = proc(Cl_auto_mj_sims, Cl_auto_mj_sims, Cl_cross_mj_sims)

    ρ_proc(sims) = begin
        isempty(sims) && return (fill(NaN,length(Lc)), fill(NaN,length(Lc)))
        R = reduce(hcat, sims)
        _, ρ̄, σρ = coarsen(ℓv, R; edges=proc_edges); ρ̄, σρ
    end
    ρ̄_qe, σρ_qe       = ρ_proc(ρ_qe_sims)
    ρ̄_gi, σρ_gi       = ρ_proc(ρ_gi_sims)
    ρ̄_mj, σρ_mj       = ρ_proc(ρ_mj_sims)
    ρ̄_mj_gi, σρ_mj_gi = ρ_proc(ρ_mj_gi_sims)

    # SNR
    T_b   = bands_per_sim(ℓv, Cl_true_sims,            proc_edges)
    Ba_qe      = bands_per_sim(ℓv, Cl_auto_qe_rdn0_sims,  proc_edges)
    Bx_qe      = bands_per_sim(ℓv, Cl_cross_qe_sims,      proc_edges)
    Ba_gi_fgmc = bands_per_sim(ℓv, Cl_auto_gi_fgmc_sims,  proc_edges)
    Ba_gi_rdn0  = bands_per_sim(ℓv, Cl_auto_gi_rdn0_sims,  proc_edges)
    Ba_gi_linrd = bands_per_sim(ℓv, Cl_auto_gi_linrd_sims, proc_edges)
    Bx_gi       = bands_per_sim(ℓv, Cl_cross_gi_sims,      proc_edges)
    Ba_mj  = bands_per_sim(ℓv, Cl_auto_mj_sims,   proc_edges)
    Bx_mj  = bands_per_sim(ℓv, Cl_cross_mj_sims,  proc_edges)
    Lc_e   = [0.5*(proc_edges[b]+proc_edges[b+1]) for b in 1:(length(proc_edges)-1)]

    snr(M,) = snr_compute(M, T_b, Lc_e, snr_L_lo, snr_L_hi, σ_sc)
    snr_a_qe      = snr(Ba_qe);      snr_x_qe  = snr(Bx_qe)
    snr_a_gi_fgmc = snr(Ba_gi_fgmc); snr_x_gi  = snr(Bx_gi)
    snr_a_gi_rdn0  = snr(Ba_gi_rdn0)
    snr_a_gi_linrd = snr(Ba_gi_linrd)
    snr_a_mj = snr(Ba_mj); snr_x_mj   = snr(Bx_mj)

    _sf(x) = isnan(x) ? "     -" : @sprintf("%6.1f", x)
    println("  SNR (L=$(Int(snr_L_lo))-$(Int(snr_L_hi))):")
    println("    QE RDN0          Auto=$(_sf(snr_a_qe))  Cross=$(_sf(snr_x_qe))")
    !isempty(Cl_auto_gi_fgmc_sims)  && println("    GI (fg-MC N0)    Auto=$(_sf(snr_a_gi_fgmc))  Cross=$(_sf(snr_x_gi))")
    !isempty(Cl_auto_gi_rdn0_sims)  && println("    GI (RDN0)        Auto=$(_sf(snr_a_gi_rdn0))")
    !isempty(Cl_auto_gi_linrd_sims) && println("    GI (lin-RD N0)   Auto=$(_sf(snr_a_gi_linrd))")
    !isempty(Cl_auto_mj_sims)       && println("    MAP joint        Auto=$(_sf(snr_a_mj))  Cross=$(_sf(snr_x_mj))")

    (label=label, Lmax=Lmax, Lc=Lc, Neff_v=Neff_v, xlim=xlim_plot, C̄_true=C̄_true,
     C̄_a_qe=C̄_a_qe, σ_a_qe=σ_a_qe, σ_th_a_qe=σ_th_a_qe,
     C̄_x_qe=C̄_x_qe, σ_x_qe=σ_x_qe, σ_th_x_qe=σ_th_x_qe,
     C̄_a_gi=C̄_a_gi, σ_a_gi=σ_a_gi, σ_th_a_gi=σ_th_a_gi,
     C̄_a_gi_fgmc=C̄_a_gi_fgmc, σ_a_gi_fgmc=σ_a_gi_fgmc, σ_th_a_gi_fgmc=σ_th_a_gi_fgmc,
     C̄_a_gi_rdn0=C̄_a_gi_rdn0, σ_a_gi_rdn0=σ_a_gi_rdn0, σ_th_a_gi_rdn0=σ_th_a_gi_rdn0,
     C̄_a_gi_linrd=C̄_a_gi_linrd, σ_a_gi_linrd=σ_a_gi_linrd, σ_th_a_gi_linrd=σ_th_a_gi_linrd,
     has_gi_rdn0=!isempty(Cl_auto_gi_rdn0_sims),
     has_gi_linrd=!isempty(Cl_auto_gi_linrd_sims),
     nsims_gi_linrd=length(Cl_auto_gi_linrd_sims),
     Cl_auto_gi_linrd_sims=Cl_auto_gi_linrd_sims,
     C̄_x_gi=C̄_x_gi, σ_x_gi=σ_x_gi, σ_th_x_gi=σ_th_x_gi,
     C̄_a_mj=C̄_a_mj, σ_a_mj=σ_a_mj, σ_th_a_mj=σ_th_a_mj,
     C̄_x_mj=C̄_x_mj, σ_x_mj=σ_x_mj, σ_th_x_mj=σ_th_x_mj,
     ρ̄_qe=ρ̄_qe, σρ_qe=σρ_qe, ρ̄_gi=ρ̄_gi, σρ_gi=σρ_gi, ρ̄_mj=ρ̄_mj, σρ_mj=σρ_mj,
     ρ̄_mj_gi=ρ̄_mj_gi, σρ_mj_gi=σρ_mj_gi,
     W_qe_raw=W_qe_raw, W_gi_b=W_gi_b, W_qe_s=W_qe_s, W_gi_s=W_gi_s,
     W_mj_raw=W_mj_raw,
     ℓ_wl=ℓ_template, ℓ_wl_map=ℓ_template_map,
     snr=(a_qe=snr_a_qe, x_qe=snr_x_qe,
          a_gi_fgmc=snr_a_gi_fgmc, a_gi_rdn0=snr_a_gi_rdn0,
          x_gi=snr_x_gi, a_mj=snr_a_mj, x_mj=snr_x_mj),
     snr_L_range=(snr_L_lo, snr_L_hi), Lc_edges=Lc_e,
     nsims_qegi=length(qegi_sims), nsims_map=length(Cl_auto_mj_sims),
     nsims_gi_fgmc=length(Cl_auto_gi_fgmc_sims),
     nsims_gi_rdn0=length(Cl_auto_gi_rdn0_sims),
     nsims_qe_rdn0=length(Cl_auto_qe_rdn0_sims),
     has_map=!isempty(Cl_auto_mj_sims),
     has_qe=!isempty(Cl_auto_qe_sims), has_gi=!isempty(Cl_auto_gi_sims),
     Ba_qe=Ba_qe, Bx_qe=Bx_qe, Ba_gi=Ba_gi_fgmc, Bx_gi=Bx_gi,
     Ba_mj=Ba_mj, Bx_mj=Bx_mj, T_b=T_b,
     Cl_auto_qe_sims=Cl_auto_qe_sims, Cl_cross_qe_sims=Cl_cross_qe_sims,
     Cl_auto_qe_full_sims=Cl_auto_qe_full_sims, Cl_auto_qe_rdn0_sims=Cl_auto_qe_rdn0_sims,
     Cl_auto_gi_sims=Cl_auto_gi_sims, Cl_auto_gi_fgmc_sims=Cl_auto_gi_fgmc_sims,
     Cl_auto_gi_rdn0_sims=Cl_auto_gi_rdn0_sims,
     Cl_cross_gi_sims=Cl_cross_gi_sims, Cl_auto_gi_full_sims=Cl_auto_gi_full_sims,
     Cl_auto_mj_sims=Cl_auto_mj_sims, Cl_cross_mj_sims=Cl_cross_mj_sims,
     Cl_true_sims=Cl_true_sims, ℓ_kk=ℓv,
     σ_scale=σ_sc, f_sky_sim=f_sky_sim,
     phi_maps_file=phi_maps_file)
end

# run both noise levels
# No hardcoded exclusions — auto-detected RMS outliers (15×median) are sufficient.

s4 = process_noise_level(
    "results/WL_qe_gi_12000.jld2",
    "results/phi_maps_qe_gi_12000.jld2",
    "S4-like (1 µK-arcmin)";
    Lmax=12000, beamFWHM=1.0, μKarcminT=1.0,
    map_wl_file  = isfile("results/WL_map_12000.jld2")  ? "results/WL_map_12000.jld2"  : nothing,
    map_phi_file = isfile("results/phi_maps_map_12000.jld2") ? "results/phi_maps_map_12000.jld2" : nothing,
    snr_L_lo=4000.0, snr_L_hi=12000.0,
    xlim_plot=(5000.0, 11000.0),
    exclude_sims=Set{Int}(),
    )

ul_files_exist = isfile("results/WL_qe_gi_12000_ul.jld2") &&
                 isfile("results/phi_maps_qe_gi_12000_ul.jld2")
# Primary UL MAP: plain run (_ul). Secondary: prior weakening ×10 (_ul_pw).
_ul_map_wl   = isfile("results/WL_map_12000_ul.jld2")             ? "results/WL_map_12000_ul.jld2"             : nothing
_ul_map_phi  = isfile("results/phi_maps_map_12000_ul.jld2")       ? "results/phi_maps_map_12000_ul.jld2"       : nothing
ul = ul_files_exist ? process_noise_level(
    "results/WL_qe_gi_12000_ul.jld2",
    "results/phi_maps_qe_gi_12000_ul.jld2",
    "UL (0.1 µK-arcmin, Lmax=12k)";
    Lmax=12000, beamFWHM=0.3, μKarcminT=0.1,
    map_wl_file   = _ul_map_wl,
    map_phi_file  = _ul_map_phi,
    snr_L_lo=4000.0, snr_L_hi=12000.0,
    xlim_plot=(5000.0, 11000.0),
    exclude_sims=Set{Int}(),
    ) : nothing

datasets = filter(!isnothing, Any[s4, ul])

# UL hess2 MAP variant — same QE/GI files, different MAP files (nburnin_hessian=2)
_ul_hess2_wl  = isfile("results/WL_map_12000_ul_hess2.jld2")       ? "results/WL_map_12000_ul_hess2.jld2"       : nothing
_ul_hess2_phi = isfile("results/phi_maps_map_12000_ul_hess2.jld2") ? "results/phi_maps_map_12000_ul_hess2.jld2" : nothing
ul_hess2 = (ul_files_exist && _ul_hess2_wl !== nothing) ? process_noise_level(
    "results/WL_qe_gi_12000_ul.jld2",
    "results/phi_maps_qe_gi_12000_ul.jld2",
    "UL (0.1 µK-arcmin, Lmax=12k, Hess-2)";
    Lmax=12000, beamFWHM=0.3, μKarcminT=0.1,
    map_wl_file  = _ul_hess2_wl,
    map_phi_file = _ul_hess2_phi,
    snr_L_lo=4000.0, snr_L_hi=12000.0,
    xlim_plot=(5000.0, 11000.0),
    exclude_sims=Set{Int}(),
    ) : nothing
# datasets_aux: per-dataset auxiliary result (hess2 MAP overlay for UL; nothing for S4)
datasets_aux = length(datasets) == 2 ? Any[nothing, ul_hess2] :
               length(datasets) == 1 && any(d->occursin("UL",d.label), datasets) ? Any[ul_hess2] :
               Any[nothing for _ in datasets]

# plot style
const ticker = PythonPlot.matplotlib.ticker

PythonPlot.rc("font",        family="serif", size=11)
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)
PythonPlot.rc("xtick.minor", width=0.5, size=2.5, visible=true)
PythonPlot.rc("ytick.minor", width=0.5, size=2.5, visible=true)

CLR = Dict("qe"=>"#D62728", "gi"=>"#1F77B4", "gi_fgmc"=>"#17BECF", "gi_rdn0"=>"#2CA02C", "gi_linrd"=>"#8C564B", "mj"=>"#9467BD", "mj_hess2"=>"#FF7F0E")
LBL = Dict("qe"=>"QE RDN0", "gi"=>"GI (global MC N0)", "gi_fgmc"=>"GI (fg-MC N0)", "gi_rdn0"=>"GI (RDN0)", "gi_linrd"=>"GI (lin-RD N0)", "mj"=>"MAP joint", "mj_hess2"=>"MAP (Hess-2)")

function set_log_ticks(ax, ymin, ymax)
    lo = floor(Int, log10(max(ymin, 1e-100)))
    hi = ceil(Int,  log10(max(ymax, 1e-100)))
    lo >= hi && (hi = lo + 1)
    ax.set_ylim(10.0^lo * 0.5, 10.0^hi * 2.5)
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=collect(2:9), numticks=100))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
end

# fig3: sigma panels — parameter summary row + auto σ + cross σ
# Error bands show ±σ/√(2(N-1)): statistical uncertainty on the estimated σ.
let
    ncols = length(datasets)

    # helper: evaluate smoothed W_L at a given L (nearest-neighbour)
    function wl_at(W, ℓ, L)
        (W === nothing || isempty(W)) && return NaN
        W[argmin(abs.(ℓ .- Float64(L)))]
    end

    # ── Layout: 3 rows [param panel, auto σ, cross σ] ─────────────────────────
    fig, axs = PythonPlot.subplots(3, ncols;
        figsize=(5.5*ncols, 12.5),
        gridspec_kw=Dict("height_ratios"=>[0.55, 1.0, 1.0], "hspace"=>0.06),
        constrained_layout=false)
    PythonPlot.subplots_adjust(left=0.09, right=0.97, top=0.97, bottom=0.06)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    # ── Row 0: parameter summary ───────────────────────────────────────────────
    for (ci, d) in enumerate(datasets)
        ax = getax(0, ci - 1)
        ax.axis("off")

        # Determine MAP settings from label
        is_ul = occursin("UL", d.label) || occursin("0.1", d.label)
        map_line = if d.has_map
            is_ul ? "MAP: QE-WF start, αmax=0.05, 40 steps, CG nsteps=500" :
                    "MAP: QE-WF start, αmax=0.3,  40 steps, CG nsteps=200"
        else
            "MAP: not run"
        end
        noise_line = is_ul ? "σ_N=0.1 µK·arcmin, beam=0.3'" :
                              "σ_N=1.0 µK·arcmin, beam=1.0'"

        # Transfer function W_L at reference L values
        L_refs  = [6000, 8000, 10000]
        hdr  = @sprintf("  %-12s  %7s  %7s  %7s", "Estimator", "L=6k", "L=8k", "L=10k")
        sep  = "  " * "-"^(length(hdr)-2)
        rows = String[hdr, sep]
        for (lbl, W, ℓ) in [("QE",       d.W_qe_s,  d.ℓ_wl),
                              ("GI",       d.W_gi_s,  d.ℓ_wl),
                              ("MAP joint", d.W_mj_raw !== nothing ?
                                  smooth_wl(Float64.(d.W_mj_raw)) : nothing,
                                  d.ℓ_wl_map)]
            ws = [isnan(wl_at(W, ℓ, L)) ? "   —  " : @sprintf("%7.3f", wl_at(W, ℓ, L))
                  for L in L_refs]
            push!(rows, @sprintf("  %-12s  %s  %s  %s", lbl, ws[1], ws[2], ws[3]))
        end

        txt = join([
            "$(d.label)",
            "$noise_line,  Lmax=$(d.Lmax),  Nside=512 (0.74\"/pix),  f_sky_paper=0.4",
            "QE/GI sims: $(d.nsims_qegi)" *
                (d.has_map ? ",   MAP sims: $(d.nsims_map)" : ""),
            map_line,
            "",
            "Transfer function W_L = ⟨C_L^{ϕ_true,ϕ̂}⟩ / C_L^{ϕϕ}  (smoothed, 9-bin running mean):",
            rows...,
        ], "\n")

        ax.text(0.03, 0.98, txt;
            transform=ax.transAxes, va="top", ha="left",
            fontsize=7.2, fontfamily="monospace",
            bbox=Dict("boxstyle"=>"round,pad=0.5", "facecolor"=>"#f7f7f7",
                      "edgecolor"=>"#bbbbbb", "linewidth"=>0.8))
    end

    # ── Rows 1–2: σ panels ────────────────────────────────────────────────────
    row0_vals = Float64[]
    row1_vals = Float64[]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1
        da = ci <= length(datasets_aux)    ? datasets_aux[ci]    : nothing
        title = "$(d.label)  (QE/GI: $(d.nsims_qegi) sims" *
                (d.has_map ? ", MAP: $(d.nsims_map) sims)" : ")")

        auto_pairs = Tuple{String,Vector{Float64},Vector{Float64}}[
            ("qe",       d.σ_a_qe,      d.σ_th_a_qe),
            ("gi_fgmc",  d.σ_a_gi_fgmc,  d.σ_th_a_gi_fgmc),
            ("gi_linrd", d.σ_a_gi_linrd, d.σ_th_a_gi_linrd),
            ("mj",       d.σ_a_mj,       d.σ_th_a_mj),
        ]
        da !== nothing && da.has_map &&
            push!(auto_pairs, ("mj_hess2", da.σ_a_mj, da.σ_th_a_mj))

        cross_pairs = Tuple{String,Vector{Float64},Vector{Float64}}[
            ("qe",  d.σ_x_qe,  d.σ_th_x_qe),
            ("gi",  d.σ_x_gi,   d.σ_th_x_gi),
            ("mj",  d.σ_x_mj,   d.σ_th_x_mj),
        ]
        da !== nothing && da.has_map &&
            push!(cross_pairs, ("mj_hess2", da.σ_x_mj, da.σ_th_x_mj))

        for (row, pairs) in [(1, auto_pairs), (2, cross_pairs)]
            ax = getax(row, c)
            for (key, σ, σ_th) in pairs
                key == "mj"       && !d.has_map      && continue
                key == "mj_hess2" && (da === nothing || !da.has_map) && continue
                key == "gi_rdn0"  && !d.has_gi_rdn0  && continue
                key == "gi_linrd" && !d.has_gi_linrd && continue
                msk = @. !isnan(Lc) & isfinite(σ) & (σ > 0)
                !any(msk) && continue

                # ±σ/√(2(N-1)) uncertainty band on the estimated σ
                N_sims = if key == "mj";           d.nsims_map
                         elseif key == "mj_hess2"; da.nsims_map
                         elseif key == "gi_fgmc";  d.nsims_gi_fgmc
                         elseif key == "gi_rdn0";  d.nsims_gi_rdn0
                         elseif key == "gi_linrd"; d.nsims_gi_linrd
                         elseif key == "qe";       row == 2 ? d.nsims_qegi : d.nsims_qe_rdn0
                         else d.nsims_qegi
                         end
                err_frac = 1 / sqrt(2 * max(N_sims - 1, 1))
                ax.fill_between(Lc[msk],
                    σ[msk] .* (1 - err_frac), σ[msk] .* (1 + err_frac);
                    color=CLR[key], alpha=0.18, linewidth=0)

                lbl = (row == 2 && key == "gi") ? "GI" : LBL[key]
                ax.semilogy(Lc[msk], σ[msk]; color=CLR[key], lw=2, label=lbl)
                row == 1 ? append!(row0_vals, σ[msk]) : append!(row1_vals, σ[msk])

                msk_th = @. !isnan(Lc) & isfinite(σ_th) & (σ_th > 0)
                any(msk_th) && ax.semilogy(Lc[msk_th], σ_th[msk_th];
                    color=CLR[key], ls="--", lw=1.2, alpha=0.7)
            end

            ax.set_xlim(d.xlim...)
            if row == 1
                ax.set_title(title, fontsize=9)
                ax.tick_params(labelbottom=false)
                ci == 1 && ax.legend(loc="upper left", frameon=false, fontsize=8,
                    title="solid=sim  ±σ/√(2N),  dashed=Knox", title_fontsize=6.5)
            else
                ax.set_xlabel(L"$L$", fontsize=12)
                ci == 1 && ax.legend(frameon=false, fontsize=8)
            end
        end
    end

    # Share y-axis within each row and set fixed visible limits.
    # Filter floor at 1e-20: MAP with poor convergence (W_L≈0) produces σ≈1e-40
    # from std of near-identical near-zero maps; those are numerical artefacts.
    ylims_fixed = [(1, (1e-13, 1e-8)), (2, (1e-14, 1e-12))]
    for (row, (ylo, yhi)) in ylims_fixed
        ax0 = getax(row, 0)
        ax0.set_ylim(ylo, yhi)
        ax0.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
        ax0.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=collect(2:9), numticks=100))
        ax0.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        for ci in 2:ncols; getax(row, ci-1).sharey(ax0); end
        for ci in 2:ncols; getax(row, ci-1).sharex(getax(row, 0)); end
        for ci in 2:ncols; getax(row, ci-1).tick_params(labelleft=false); end
    end

    getax(1, 0).set_ylabel(L"$\sigma[C_L^{\hat\kappa\hat\kappa}]$", fontsize=12)
    getax(2, 0).set_ylabel(L"$\sigma[C_L^{\kappa\hat\kappa}]$", fontsize=12)

    fig.text(0.5, 0.005,
        L"Shaded bands: $\pm\,\hat\sigma/\!\sqrt{2(N_{\rm sims}-1)}$ " *
        "(statistical uncertainty on the estimated \$\\hat\\sigma\$).  " *
        "Dashed lines: Knox formula.",
        ha="center", fontsize=7, style="italic")

    fig.savefig("$OUT_DIR/fig3_sigma_panels.png"; dpi=200, bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved fig3_sigma_panels.png")
end

# fig2: mean spectra
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols;
        figsize=(5.5*ncols, 8.0), sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    row_vals_fig2 = [Float64[], Float64[]]   # collect for auto y-scaling

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1

        # Row 0: mean cross
        ax = getax(0, c)
        msk_t = @. !isnan(Lc) & isfinite(d.C̄_true) & (d.C̄_true > 1e-20)
        any(msk_t) && ax.semilogy(Lc[msk_t], d.C̄_true[msk_t];
            color="k", ls="--", lw=1.8, label=L"C_L^{\kappa\kappa}\,(true)")
        append!(row_vals_fig2[1], d.C̄_true[msk_t])
        for (key, C̄x) in [("qe", d.C̄_x_qe), ("gi", d.C̄_x_gi), ("mj", d.C̄_x_mj)]
            key == "mj"  && !d.has_map  && continue
            msk = @. !isnan(Lc) & isfinite(C̄x) & (abs(C̄x) > 1e-20)
            any(msk) && ax.semilogy(Lc[msk], abs.(C̄x[msk]); color=CLR[key], lw=2, label=LBL[key])
            append!(row_vals_fig2[1], abs.(C̄x[msk]))
        end
        ax.set_title(d.label, fontsize=9); ax.set_xlim(d.xlim...)
        ci == 1 && ax.legend(loc="upper right", frameon=false, fontsize=8)

        # Row 1: mean auto
        ax = getax(1, c)
        any(msk_t) && ax.semilogy(Lc[msk_t], d.C̄_true[msk_t];
            color="k", ls="--", lw=1.8, label=L"C_L^{\kappa\kappa}\,(true)")
        append!(row_vals_fig2[2], d.C̄_true[msk_t])
        for (key, C̄a) in [("qe", d.C̄_a_qe), ("gi", d.C̄_a_gi), ("gi_fgmc", d.C̄_a_gi_fgmc), ("mj", d.C̄_a_mj)]
            key == "mj"  && !d.has_map  && continue
            msk = @. !isnan(Lc) & isfinite(C̄a) & (abs(C̄a) > 1e-20)
            any(msk) && ax.semilogy(Lc[msk], abs.(C̄a[msk]); color=CLR[key], lw=2, label=LBL[key])
            append!(row_vals_fig2[2], abs.(C̄a[msk]))
        end
        ax.set_xlim(d.xlim...); ax.set_xlabel(L"L", fontsize=12)
        ci == 1 && ax.legend(loc="upper right", frameon=false, fontsize=8)
    end
    getax(0, 0).set_ylabel(L"\bar{C}_L^{\kappa\hat\kappa}", fontsize=12)
    getax(1, 0).set_ylabel(L"\bar{C}_L^{\hat\kappa\hat\kappa}", fontsize=12)

    # Auto y-scaling per row
    for (row, vals) in enumerate(row_vals_fig2)
        isempty(vals) && continue
        ax0 = getax(row - 1, 0)
        set_log_ticks(ax0, minimum(vals), maximum(vals))
        for ci in 2:ncols; getax(row - 1, ci - 1).sharey(ax0); end
    end

    fig.savefig("$OUT_DIR/fig2_mean_spectra.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig2_mean_spectra.png")
end

# fig4: effective reconstruction noise N_L,eff (paper Eq. 33)
# N_L,eff = σ̂ × sqrt(ΔL(2L+1)f_sky/2) - C^κκ_L,  with (2L+1) ≈ 2(L+0.5)
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
        for (key, σ_a) in [("qe", d.σ_a_qe), ("gi", d.σ_a_gi), ("gi_fgmc", d.σ_a_gi_fgmc), ("mj", d.σ_a_mj)]
            key == "mj"  && !d.has_map  && continue
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

# fig5: improvement ratio σ_QE / σ_estimator
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols; figsize=(5.5*ncols, 7.0),
        sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1
        for (row, (σ_qe, entries)) in enumerate([
                (d.σ_a_qe, [("gi", d.σ_a_gi), ("gi_fgmc", d.σ_a_gi_fgmc), ("mj", d.σ_a_mj)]),
                (d.σ_x_qe, [("gi", d.σ_x_gi), ("mj", d.σ_x_mj)])])
            ax = getax(row - 1, c)
            ax.axhline(1.0; color="grey", ls=":", lw=1)
            for (key, σ) in entries
                key == "mj"  && !d.has_map  && continue
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

# fig6: correlation coefficient ρ_L
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
            key == "mj"  && !d.has_map  && continue
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

# fig6b: ρ(ϕ_MAP_deb, ϕ_GI_deb) — cross-correlation between debiased MAP and GI estimates
let
    ncols = length(datasets)
    any(d.has_map && d.has_gi for d in datasets) || @goto skip_fig6b
    fig, axs = PythonPlot.subplots(1, ncols; figsize=(5.5*ncols, 4.5), constrained_layout=true)
    getax(c) = ncols == 1 ? axs : axs[c]
    for (ci, d) in enumerate(datasets)
        ax = getax(ci - 1); Lc = d.Lc
        (!d.has_map || !d.has_gi) && (ax.set_visible(false); continue)
        msk = @. !isnan(Lc) & isfinite(d.ρ̄_mj_gi)
        any(msk) && ax.plot(Lc[msk], d.ρ̄_mj_gi[msk]; color=CLR["mj"], lw=2, label=LBL["mj"])
        any(msk) && ax.fill_between(Lc[msk],
            d.ρ̄_mj_gi[msk] .- d.σρ_mj_gi[msk],
            d.ρ̄_mj_gi[msk] .+ d.σρ_mj_gi[msk]; color=CLR["mj"], alpha=0.15)
        ax.axhline(1.0; color="grey", ls=":", lw=1, alpha=0.7)
        ax.set_xlabel(L"L", fontsize=12)
        ax.set_title(d.label, fontsize=9)
        ax.set_xlim(d.xlim...); ax.set_ylim(-0.1, 1.15)
        ci == 1 && ax.legend(frameon=false, fontsize=8)
    end
    getax(0).set_ylabel(L"\rho(\hat\phi_{\rm MAP,deb},\,\hat\phi_{\rm GI,deb})", fontsize=11)
    fig.suptitle(L"Cross-correlation between debiased MAP and GI estimates $\rho(\hat\phi_{\rm MAP},\hat\phi_{\rm GI})$", fontsize=10)
    fig.savefig("$OUT_DIR/fig6b_rho_map_gi.png"; dpi=200, bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved fig6b_rho_map_gi.png")
    @label skip_fig6b
end

# fig7: combined Neff + ρ
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

# fig_WL: transfer function W_L
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

# figB: convergence with number of sims
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
                ([("qe", d.Cl_cross_qe_sims), ("gi", d.Cl_cross_gi_sims), ("mj", d.Cl_cross_mj_sims)],
                 L"\sigma[C_L^{\kappa\hat\kappa}]"),
                ([("qe", d.Cl_auto_qe_sims), ("gi", d.Cl_auto_gi_sims), ("mj", d.Cl_auto_mj_sims)],
                 L"\sigma[C_L^{\hat\kappa\hat\kappa}]")])
            ax = getax(row - 1, c)
            for (key, sims) in sims_list
                key == "mj" && !d.has_map && continue
                length(sims) < 4 && continue
                nmax = length(sims)
                steps = unique(vcat([div(nmax, 10) * k for k in 1:10], [nmax]))
                σ_mid = Float64[]
                for n in steps
                    A = reduce(hcat, sims[1:n])
                    _, _, σ_v = coarsen(ℓv, A; edges=proc_edges)
                    σ_sc = d.σ_scale
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

# fig_covariance_correlation
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

# Rebin per-sim spectra to finer ΔL for the correlation matrix plot.
# proc_edges uses ΔL=2000 (4–6 bins) which gives an uninformative tiny matrix.
# ΔL=500 gives ~16 bins over L=4000–12000, revealing off-diagonal structure clearly.
const corr_edges = collect(3000.0:500.0:15001.0)   # ΔL=500 bins for correlation plot
const corr_Lc_e  = [0.5*(corr_edges[b]+corr_edges[b+1]) for b in 1:(length(corr_edges)-1)]

function rebinned_bands(d, sims_key)
    sims = getfield(d, sims_key)
    isempty(sims) && return fill(NaN, length(corr_edges)-1, 0)
    bands_per_sim(d.ℓ_kk, sims, corr_edges)
end

let
    n_per = 3   # QE, GI, MAP per dataset
    ncols = n_per * length(datasets)
    # 3 rows: auto without RDN0 | auto with RDN0 / GI raw | cross
    # "without RDN0" uses full auto (signal+N0)/W² so N0 off-diagonal correlations are visible
    # "with RDN0" uses per-sim QE RDN0, and GI is shown raw because analytical N0 subtraction is disabled
    # cross has no N0 bias so is shown once without subtraction
    fig, axs = PythonPlot.subplots(3, ncols; figsize=(4.0*ncols, 12.0), constrained_layout=true)
    getac(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    row_labels = ["auto (no N0 sub)", "auto (per-sim N0 sub)", "cross"]

    for (ci, d) in enumerate(datasets)
        c0 = n_per * (ci - 1)

        B_sets = [
            # Row 0: full auto — no N0 subtraction; N0 fluctuations drive off-diagonal
            (bands_per_sim(d.ℓ_kk, d.Cl_auto_qe_full_sims, corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_auto_gi_full_sims, corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_auto_mj_sims,      corr_edges)),
            # Row 1: N0 subtraction — QE RDN0 (per-sim); GI: fgmc (S4) or linrd (UL); MAP not shown
            (bands_per_sim(d.ℓ_kk, d.Cl_auto_qe_rdn0_sims, corr_edges),
             bands_per_sim(d.ℓ_kk, (occursin("UL", d.label) && d.has_gi_linrd ?
                                     d.Cl_auto_gi_linrd_sims : d.Cl_auto_gi_fgmc_sims), corr_edges),
             fill(NaN, length(corr_edges)-1, 0)),
            # Row 2: cross (N0-free by construction)
            (bands_per_sim(d.ℓ_kk, d.Cl_cross_qe_sims, corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_cross_gi_sims,  corr_edges),
             bands_per_sim(d.ℓ_kk, d.Cl_cross_mj_sims,  corr_edges)),
        ]

        for (row, (B_qe, B_gi, B_mj)) in enumerate(B_sets)
            r = row - 1
            rl = row_labels[row]
            for (ck, B, ttl) in [(c0,   B_qe, "QE – $rl\n$(d.label)"),
                                  (c0+1, B_gi, "GI – $rl\n$(d.label)"),
                                  (c0+2, B_mj, "MAP – $rl\n$(d.label)")]
                ax = getac(r, ck)
                if (!d.has_map && ck == c0+2) || size(B, 2) < 3
                    ax.set_visible(false); continue
                end
                im = fill_corr_ax(ax, B, corr_Lc_e; title=ttl)
                im !== nothing && fig.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
            end
        end
    end
    fig.suptitle(L"Bandpower correlation $\rho_{LL'}$ ($\Delta L=500$)  —  row 1: no N0 sub, row 2: N0 sub (QE: RDN0, GI: fg-MC N0 [S4] / LinRD [UL], MAP: none), row 3: cross", fontsize=10)
    fig.savefig("$OUT_DIR/fig_covariance_correlation.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved fig_covariance_correlation.png")
end

# figN0: mean GI N0 estimates per method (fgmc, mc, linrd, rdn0) vs QE RDN0
let
    for d in datasets
        isfile(d.phi_maps_file) || continue
        has_n0 = false

        n0_fgmc_acc = nothing; ell_fgmc = nothing
        n0_linrd_acc = nothing; ell_linrd = nothing
        n0_linrd_ab_acc = nothing; ell_linrd_ab = nothing
        n0_rdn0_acc = nothing; n0_rdn0v2_acc = nothing
        n0_qe_acc   = nothing
        n_fgmc = 0; n_linrd = 0; n_linrd_ab = 0; n_rdn0 = 0; n_rdn0v2 = 0; n_qe = 0

        jldopen(d.phi_maps_file, "r") do f
            for s in sort([parse(Int, m.captures[1])
                           for k in keys(f)
                           for m in [match(r"^sim_(\d+)$", k)]
                           if m !== nothing])
                if haskey(f, "sim_$s/N0_gi_fgmc")
                    v = Float64.(read(f, "sim_$s/N0_gi_fgmc"))
                    if ell_fgmc === nothing && haskey(f, "sim_$s/N0_gi_fgmc_ell")
                        ell_fgmc = Float64.(read(f, "sim_$s/N0_gi_fgmc_ell"))
                    end
                    n0_fgmc_acc = n0_fgmc_acc === nothing ? copy(v) : n0_fgmc_acc .+ v
                    n_fgmc += 1; has_n0 = true
                end
                if haskey(f, "sim_$s/N0_gi_linrd")
                    v = Float64.(read(f, "sim_$s/N0_gi_linrd"))
                    if ell_linrd === nothing && haskey(f, "sim_$s/N0_gi_linrd_ell")
                        ell_linrd = Float64.(read(f, "sim_$s/N0_gi_linrd_ell"))
                    end
                    n0_linrd_acc = n0_linrd_acc === nothing ? copy(v) : n0_linrd_acc .+ v
                    n_linrd += 1
                end
                if haskey(f, "sim_$s/N0_gi_linrd_ab")
                    v = Float64.(read(f, "sim_$s/N0_gi_linrd_ab"))
                    if ell_linrd_ab === nothing && haskey(f, "sim_$s/N0_gi_linrd_ab_ell")
                        ell_linrd_ab = Float64.(read(f, "sim_$s/N0_gi_linrd_ab_ell"))
                    end
                    n0_linrd_ab_acc = n0_linrd_ab_acc === nothing ? copy(v) : n0_linrd_ab_acc .+ v
                    n_linrd_ab += 1
                end
                if haskey(f, "sim_$s/N0_gi_rdn0")
                    v = Float64.(read(f, "sim_$s/N0_gi_rdn0"))
                    n0_rdn0_acc = n0_rdn0_acc === nothing ? copy(v) : n0_rdn0_acc .+ v
                    n_rdn0 += 1
                end
                if haskey(f, "sim_$s/N0_gi_rdn0_v2")
                    v = Float64.(read(f, "sim_$s/N0_gi_rdn0_v2"))
                    n0_rdn0v2_acc = n0_rdn0v2_acc === nothing ? copy(v) : n0_rdn0v2_acc .+ v
                    n_rdn0v2 += 1
                end
                if haskey(f, "sim_$s/N0_rdn0")
                    v = Float64.(read(f, "sim_$s/N0_rdn0"))
                    n0_qe_acc = n0_qe_acc === nothing ? copy(v) : n0_qe_acc .+ v
                    n_qe += 1
                end
            end
        end
        has_n0 || continue

        fig, ax = PythonPlot.subplots(1, 1; figsize=(8.0, 5.0), constrained_layout=true)

        function plot_n0!(ell, acc, n, label, color; ls="-")
            (n == 0 || ell === nothing || length(ell) != length(acc)) && return
            kf = @. (ell^2 / 2)^2
            ax.semilogy(ell, kf .* acc ./ n; label=label, color=color, lw=2, ls=ls)
        end

        plot_n0!(ell_fgmc,     n0_fgmc_acc,     n_fgmc,     "GI fg-MC N0 ($(n_fgmc) sims)",              CLR["gi_fgmc"])
        plot_n0!(ell_linrd,    n0_linrd_acc,    n_linrd,    "GI lin-RD N0 ($(n_linrd) sims)",             CLR["gi_linrd"]; ls="-.")
        plot_n0!(ell_linrd_ab, n0_linrd_ab_acc, n_linrd_ab, "GI lin-RD analytic baseline ($(n_linrd_ab) sims)", CLR["gi_linrd"]; ls="--")
        plot_n0!(ell_fgmc,     n0_rdn0_acc,     n_rdn0,     "GI RDN0 ($(n_rdn0) sims)",                  CLR["gi_rdn0"]; ls="--")
        plot_n0!(ell_fgmc,     n0_rdn0v2_acc,   n_rdn0v2,   "GI RDN0-v2/σ_fid ($(n_rdn0v2) sims)",      CLR["gi_rdn0"]; ls=":")
        if n_qe > 0 && length(d.ℓ_kk) == length(n0_qe_acc)
            kfac_qe = @. (d.ℓ_kk^2 / 2)^2
            ax.semilogy(d.ℓ_kk,
                kfac_qe .* n0_qe_acc ./ n_qe;
                label="QE RDN0 ($(n_qe) sims)", color=CLR["qe"], lw=2)
        end

        ax.set_xlabel(L"$L$", fontsize=12)
        ax.set_ylabel(L"$(L^2/2)^2\,N_L^{(0)}$", fontsize=11)
        ax.set_title("GI N0 estimates — $(d.label)", fontsize=10)
        ax.set_xlim(d.xlim...)
        ax.legend(frameon=false, fontsize=9)
        fig.savefig("$OUT_DIR/figN0_$(replace(d.label, r"[^A-Za-z0-9]" => "_")).png"; dpi=200)
        PythonPlot.plotclose("all")
        println("Saved figN0 for $(d.label)")
    end
end

# figA: quarter-consistency — σ for each quarter vs full sample, auto and cross
let
    ncols = length(datasets)
    fig, axs = PythonPlot.subplots(2, ncols;
        figsize=(5.5*ncols, 8.0), sharex="col", constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    quarter_alphas = [0.85, 0.65, 0.45, 0.30]   # progressively lighter quarters

    row_vals = [Float64[], Float64[]]   # collect plotted values per row for auto y-scaling

    for (ci, d) in enumerate(datasets)
        Lc = d.Lc; c = ci - 1; ℓv = d.ℓ_kk

        auto_list  = [("qe", d.Cl_auto_qe_sims),  ("gi", d.Cl_auto_gi_sims)]
        cross_list = [("qe", d.Cl_cross_qe_sims), ("gi", d.Cl_cross_gi_sims)]
        d.has_map && push!(auto_list,  ("mj", d.Cl_auto_mj_sims))
        d.has_map && push!(cross_list, ("mj", d.Cl_cross_mj_sims))
        for (row, (sims_list, ylabel)) in enumerate([
                (auto_list,  L"\sigma[C_L^{\hat\kappa\hat\kappa}]"),
                (cross_list, L"\sigma[C_L^{\kappa\hat\kappa}]")])
            ax = getax(row - 1, c)

            for (key, sims) in sims_list
                length(sims) < 8 && continue
                n = length(sims)
                q = n ÷ 4   # quarter size

                # Full-sample σ (thick dotted — the "mean" reference)
                A_full = reduce(hcat, sims)
                _, _, σ_full = coarsen(ℓv, A_full; edges=proc_edges)
                σ_full .*= d.σ_scale
                msk = @. !isnan(Lc) & isfinite(σ_full) & (σ_full > 1e-20)
                any(msk) && ax.semilogy(Lc[msk], σ_full[msk];
                    color=CLR[key], lw=2.5, ls=":", label="$(LBL[key]) (all $(n))")
                append!(row_vals[row], σ_full[msk])

                # 4 quarters (thin solid, progressively lighter)
                for qi in 1:4
                    i_lo = (qi-1)*q + 1
                    i_hi = qi == 4 ? n : qi*q   # last quarter gets remainder
                    sub = sims[i_lo:i_hi]
                    length(sub) < 2 && continue
                    A_q = reduce(hcat, sub)
                    _, _, σ_q = coarsen(ℓv, A_q; edges=proc_edges)
                    σ_q .*= d.σ_scale
                    msk_q = @. !isnan(Lc) & isfinite(σ_q) & (σ_q > 1e-20)
                    lbl = qi == 1 ? "$(LBL[key]) Q$(qi)–Q4 ($(length(sub)) each)" : ""
                    any(msk_q) && ax.semilogy(Lc[msk_q], σ_q[msk_q];
                        color=CLR[key], lw=1.2, ls="-",
                        alpha=quarter_alphas[qi], label=lbl)
                    append!(row_vals[row], σ_q[msk_q])
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

    # Auto y-scaling: share y per row, set log ticks from data range
    for (row, vals) in enumerate(row_vals)
        isempty(vals) && continue
        ax0 = getax(row - 1, 0)
        set_log_ticks(ax0, minimum(vals), maximum(vals))
        for ci in 2:ncols; getax(row - 1, ci - 1).sharey(ax0); end
    end

    fig.suptitle("Quarter consistency  (dotted = full sample, solid = 4 quarters)",
        fontsize=10)
    fig.savefig("$OUT_DIR/figA_quarter_consistency.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved figA_quarter_consistency.png")
end

# figC: MAP convergence — total logpdf + likelihood/prior split + CG iteration count
# Uses WL_map file for logpdf histories; diag_map file (if present) for ll/prior/CG.
let
    map_entries = [
        ("S4-like (1 µK-arcmin)",       "results/WL_map_12000.jld2",        "results/diag_map_12000.jld2"),
        ("UL (0.1 µK-arcmin, orig)",    "results/WL_map_12000_ul.jld2",     "results/diag_map_12000_ul.jld2"),
        ("UL (Hess+deproj)",            "results/WL_map_12000_ul_hess.jld2","results/diag_map_12000_ul_hess.jld2"),
    ]
    entries = [(lbl, wf, df) for (lbl, wf, df) in map_entries if isfile(wf)]
    isempty(entries) && (println("No MAP WL files found — skipping convergence plot"); @goto skip_conv)

    has_diag = [isfile(df) for (_, _, df) in entries]
    any_diag = any(has_diag)

    # 2 rows: row 0 = Δlogpdf curves; row 1 = CG iterations (if diag available)
    nrows = any_diag ? 2 : 1
    fig, axs = PythonPlot.subplots(nrows, length(entries);
        figsize=(5.5*length(entries), 4.0*nrows), constrained_layout=true,
        squeeze=false)
    getax(row, col) = axs[row, col]

    for (ci, (lbl, wl_file, diag_file)) in enumerate(entries)
        col = ci - 1
        d_map = JLD2.load(wl_file)
        if !haskey(d_map, "logpdf_histories")
            getax(0, col).set_title("$lbl\n(no history saved)")
            continue
        end

        # ── Row 0: logpdf / loglike / logprior convergence ─────────────────────
        ax = getax(0, col)

        # Total logpdf from WL file — filter -Inf outliers before averaging
        hists_all = d_map["logpdf_histories"]
        hists = filter(h -> all(isfinite, h), hists_all)
        n_bad = length(hists_all) - length(hists)
        isempty(hists) && (ax.set_title("$lbl\n(all histories non-finite)"); continue)
        nsteps = minimum(length.(hists))
        shifted_lp = [Float64.(h[1:nsteps]) .- h[1] for h in hists]
        mean_lp    = [mean(s[t] for s in shifted_lp) for t in 1:nsteps]
        steps = 1:nsteps

        for s in shifted_lp
            ax.plot(steps, s; color="#9467BD", lw=0.4, alpha=0.12)
        end
        ax.plot(steps, mean_lp; color="#9467BD", lw=2.5,
            label="logpdf ($(length(hists)) sims$(n_bad>0 ? ", $n_bad excl." : ""))")

        # Likelihood and prior from diag file
        if has_diag[ci] && isfile(diag_file)
            ll_hists = Vector{Float64}[]; lp_hists = Vector{Float64}[]
            jldopen(diag_file, "r") do fd
                for k in keys(fd)
                    startswith(k, "sim_") || continue
                    haskey(fd, "$k/loglike") || continue
                    ll = Float64.(fd["$k/loglike"]); lp = Float64.(fd["$k/logprior"])
                    all(isfinite, ll) && push!(ll_hists, ll[1:min(end,nsteps)])
                    all(isfinite, lp) && push!(lp_hists, lp[1:min(end,nsteps)])
                end
            end
            if !isempty(ll_hists)
                ns = minimum(length.(ll_hists))
                mean_ll = [mean(h[t] for h in ll_hists) for t in 1:ns]
                mean_lpr = [mean(h[t] for h in lp_hists) for t in 1:ns]
                # shift to zero at step 1
                ax.plot(1:ns, mean_ll  .- mean_ll[1];  color="#D62728", lw=2.0, ls="--", label="loglike Δ")
                ax.plot(1:ns, mean_lpr .- mean_lpr[1]; color="#1F77B4", lw=2.0, ls=":",  label="logprior Δ")
            end
        end

        ax.axhline(0; color="k", ls=":", lw=0.8)
        ax.set_ylabel(L"\Delta\log(\cdot)", fontsize=10)
        ax.legend(frameon=false, fontsize=7.5, loc="lower right")
        last5_rate = nsteps >= 6 ? (mean_lp[end] - mean_lp[end-4]) / 5 : NaN
        avg_rate   = mean_lp[end] / (nsteps - 1)
        conv_str   = (!isnan(last5_rate) && abs(avg_rate) > 0) ?
            (abs(last5_rate)/abs(avg_rate) < 0.05 ?
                @sprintf("✓ converged (last5 Δ=%.1f)", last5_rate*5) :
                @sprintf("⚠ still improving (last5 Δ=%.1f, %.1f%%/step)", last5_rate*5, 100*abs(last5_rate)/abs(avg_rate))) : ""
        ax.set_title("$lbl  $conv_str", fontsize=9)
        nrows == 1 && ax.set_xlabel("MAP step")

        # ── Row 1: f-step CG iteration count per MAP step ──────────────────────
        any_diag || continue
        ax2 = getax(1, col)
        if has_diag[ci] && isfile(diag_file)
            ncg_mat = Vector{Float64}[]
            jldopen(diag_file, "r") do fd
                for k in keys(fd)
                    startswith(k, "sim_") || continue
                    haskey(fd, "$k/nCG") || continue
                    push!(ncg_mat, Float64.(fd["$k/nCG"][1:min(end,nsteps)]))
                end
            end
            if !isempty(ncg_mat)
                ns = minimum(length.(ncg_mat))
                mean_cg  = [mean(h[t] for h in ncg_mat) for t in 1:ns]
                low_cg   = [quantile([h[t] for h in ncg_mat], 0.25) for t in 1:ns]
                high_cg  = [quantile([h[t] for h in ncg_mat], 0.75) for t in 1:ns]
                ax2.fill_between(1:ns, low_cg, high_cg; color="#2CA02C", alpha=0.25)
                ax2.plot(1:ns, mean_cg; color="#2CA02C", lw=2.0, label="mean CG iters (IQR shaded)")
                ax2.legend(frameon=false, fontsize=7.5)
            end
        else
            ax2.set_visible(false)
        end
        ax2.set_xlabel("MAP step"); ax2.set_ylabel("CG iterations (f-step)", fontsize=9)
    end

    fig.savefig("$OUT_DIR/figC_map_convergence.png"; dpi=200)
    PythonPlot.plotclose("all")
    println("Saved figC_map_convergence.png")
    @label skip_conv
end

# SNR table
let
    _sf(x) = isnan(x) ? "        -" : @sprintf("%9.1f", x)
    _sp(x) = x <= 0   ? "        -" : @sprintf("%9.0f", x)   # paper values (integers)
    W = 32   # dataset column width
    col = "  $(rpad("Dataset", W))  Auto-QE  Auto-GI  Auto-MAP  Cross-QE  Cross-GI  Cross-MAP"
    sep = "="^length(col)
    thn = "  " * "-"^(length(col)-2)

    lo_hi = isempty(datasets) ? "?" : let d=first(datasets); "$(Int(d.snr_L_range[1]))-$(Int(d.snr_L_range[2]))"; end
    hdr = "SNR Table  (f_sky=0.4, from sims, L=$lo_hi)"
    lines = [hdr, sep, col]

    for (ci, d) in enumerate(datasets)
        tag = rpad(d.label, W)
        push!(lines, "  $tag $(_sf(d.snr.a_qe)) $(_sf(d.snr.a_gi_fgmc)) $(_sf(d.snr.a_mj))  $(_sf(d.snr.x_qe)) $(_sf(d.snr.x_gi)) $(_sf(d.snr.x_mj))")
    end

    push!(lines, thn)
    push!(lines, "  $(rpad("Paper S4 (Hadzhiyska+2019)", W)) $(_sp(100))  $(_sp(360))  $(_sp(0))   $(_sp(550))  $(_sp(1440))  $(_sp(0))")
    push!(lines, "  $(rpad("Paper UL (Hadzhiyska+2019)", W)) $(_sp(205)) $(_sp(1515))  $(_sp(0))   $(_sp(710))  $(_sp(4100))  $(_sp(0))")
    push!(lines, sep)

    for l in lines; println(l); end
    open("$OUT_DIR/snr_table.txt", "w") do io
        for l in lines; println(io, l); end
    end
    println("Saved snr_table.txt")
end

println("\nAll figures saved to $OUT_DIR")
