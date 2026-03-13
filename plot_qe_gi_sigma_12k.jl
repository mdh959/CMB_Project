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

const Cℓ          = camb(r=0.05, ℓmax=21000)
const θpix        = 0.7438046267475303
const Nside       = 512
const pol         = :I

const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (Nside * θpix_rad)^2 / (4π)
const f_sky_paper = 0.4
const σ_scale     = sqrt(f_sky_patch / f_sky_paper)
const minW        = 1e-8
const ΔL          = 500
const Δℓ_spec     = 30
const edges       = collect(4500 : Int(ΔL) : 11000 + Int(ΔL))

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

function process_noise_level(WL_file, phi_maps_file, label; Lmax=12000, beamFWHM=1.0, μKarcminT=1.0)
    println("\n=== Processing $label ===")

    # Load W_L
    d_wl       = JLD2.load(WL_file)
    ℓ_template = Float64.(d_wl["ℓ_template"])
    W_gi       = Float64.(d_wl["W_gi"])
    nsims_wl   = d_wl["nsims_completed"]
    has_wf     = haskey(d_wl, "W_qe_wf")
    W_qe_wf    = has_wf ? Float64.(d_wl["W_qe_wf"]) : nothing
    println("  W_L: $nsims_wl sims, $(length(ℓ_template)) ℓ-bins  (QE WF: $has_wf)")

    # Load empirical W_qe from file. For UL noise the QE response departs significantly
    # from 1 (can be negative at low L in signal-dominated regime), so we must not
    # hardcode W_L = 1 — that causes the UL QE to be completely mis-debiased.
    # The debias function already zeros modes where W <= minW (handles negative W).
    has_qe_raw  = haskey(d_wl, "W_qe")
    W_qe_raw    = has_qe_raw ? Float64.(d_wl["W_qe"]) : ones(length(ℓ_template))
    WL_qe_Cℓs    = Cℓs(ℓ_template, W_qe_raw)
    WL_gi_Cℓs    = Cℓs(ℓ_template, W_gi)
    WL_qe_wf_Cℓs = has_wf ? Cℓs(ℓ_template, W_qe_wf) : nothing

    # Load phi maps index
    phi_sims = Int[]
    jldopen(phi_maps_file, "r") do f
        for key in keys(f)
            m = match(r"^sim_(\d+)$", key)
            m !== nothing && push!(phi_sims, parse(Int, m.captures[1]))
        end
    end
    sort!(phi_sims)
    nsims = length(phi_sims)
    println("  phi maps: $nsims sims from $phi_maps_file")

    # Get projection metadata from a single load_sim call (reused for all sims)
    Cℓn_meta = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
    meta_seed = jldopen(phi_maps_file, "r") do f; read(f, "sim_$(phi_sims[1])/seed"); end
    (; ϕ) = load_sim(; seed=meta_seed, Cℓ=Cℓ, Cℓn=Cℓn_meta, θpix=θpix,
                       T=Float64, Nside=Nside, beamFWHM=beamFWHM, pol=pol,
                       bandpass_mask=LowPass(Lmax),
                       pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))
    ϕ_ref = Map(ϕ)

    # Per-sim spectra
    Cl_auto_qe_sims    = Vector{Vector{Float64}}()
    Cl_auto_qe_wf_sims = Vector{Vector{Float64}}()
    Cl_auto_gi_sims    = Vector{Vector{Float64}}()
    Cl_cross_qe_sims   = Vector{Vector{Float64}}()
    Cl_cross_qe_wf_sims = Vector{Vector{Float64}}()
    Cl_cross_gi_sims   = Vector{Vector{Float64}}()
    Cl_true_sims       = Vector{Vector{Float64}}()
    ℓ_kk               = nothing

    jldopen(phi_maps_file, "r") do f
        for (i, s) in enumerate(phi_sims)
            ϕ_true_arr   = read(f, "sim_$s/ϕ_true")
            ϕ_qe_arr     = read(f, "sim_$s/ϕ_qe")
            ϕ_gi_arr     = read(f, "sim_$s/ϕ_gi")
            has_qe_wf_s  = haskey(f, "sim_$s/ϕ_qe_wf")
            ϕ_qe_wf_arr  = has_qe_wf_s ? read(f, "sim_$s/ϕ_qe_wf") : nothing

            # Wrap as CMBLensing fields using shared projection metadata
            ϕ_true_field   = typeof(ϕ_ref)(Float64.(ϕ_true_arr), ϕ_ref.metadata)
            ϕ_qe_field     = typeof(ϕ_ref)(Float64.(ϕ_qe_arr),   ϕ_ref.metadata)
            ϕ_gi_field     = typeof(ϕ_ref)(Float64.(ϕ_gi_arr),   ϕ_ref.metadata)
            ϕ_qe_wf_field  = has_qe_wf_s ? typeof(ϕ_ref)(Float64.(ϕ_qe_wf_arr), ϕ_ref.metadata) : nothing

            # Debias in Fourier space: ϕ_deb(k) = ϕ_raw(k) / W_L(|k|)
            ϕ_qe_deb    = debias_phi_with_WL(ϕ_qe_field, WL_qe_Cℓs; minW=minW)
            ϕ_gi_deb    = debias_phi_with_WL(ϕ_gi_field, WL_gi_Cℓs; minW=minW)
            ϕ_qe_wf_deb = (has_wf && ϕ_qe_wf_field !== nothing) ?
                           debias_phi_with_WL(ϕ_qe_wf_field, WL_qe_wf_Cℓs; minW=minW) : nothing

            cl_tt         = get_Cℓ(ϕ_true_field;               Δℓ=Δℓ_spec)
            cl_auto_qe    = get_Cℓ(ϕ_qe_deb;                   Δℓ=Δℓ_spec)
            cl_auto_gi    = get_Cℓ(ϕ_gi_deb;                   Δℓ=Δℓ_spec)
            cl_xqe        = get_Cℓ(ϕ_true_field, ϕ_qe_deb;     Δℓ=Δℓ_spec)
            cl_xgi        = get_Cℓ(ϕ_true_field, ϕ_gi_deb;     Δℓ=Δℓ_spec)

            if ℓ_kk === nothing
                ℓ_kk = Float64.(collect(cl_tt.ℓ))
            end
            kfac = @. (ℓ_kk^2 / 2)^2

            push!(Cl_true_sims,     kfac .* Float64.(cl_tt.Cℓ))
            push!(Cl_auto_qe_sims,  kfac .* Float64.(cl_auto_qe.Cℓ))
            push!(Cl_auto_gi_sims,  kfac .* Float64.(cl_auto_gi.Cℓ))
            push!(Cl_cross_qe_sims, kfac .* Float64.(cl_xqe.Cℓ))
            push!(Cl_cross_gi_sims, kfac .* Float64.(cl_xgi.Cℓ))

            if ϕ_qe_wf_deb !== nothing
                cl_auto_qe_wf  = get_Cℓ(ϕ_qe_wf_deb;                Δℓ=Δℓ_spec)
                cl_xqe_wf      = get_Cℓ(ϕ_true_field, ϕ_qe_wf_deb;  Δℓ=Δℓ_spec)
                push!(Cl_auto_qe_wf_sims,  kfac .* Float64.(cl_auto_qe_wf.Cℓ))
                push!(Cl_cross_qe_wf_sims, kfac .* Float64.(cl_xqe_wf.Cℓ))
            end

            print("\r  Debiased sim $i/$nsims")
            flush(stdout)
        end
    end
    println()

    Aqe  = reduce(hcat, Cl_auto_qe_sims)
    Agi  = reduce(hcat, Cl_auto_gi_sims)
    Xqe  = reduce(hcat, Cl_cross_qe_sims)
    Xgi  = reduce(hcat, Cl_cross_gi_sims)
    Tmat = reduce(hcat, Cl_true_sims)
    has_qe_wf_data = !isempty(Cl_auto_qe_wf_sims)
    Aqe_wf = has_qe_wf_data ? reduce(hcat, Cl_auto_qe_wf_sims) : nothing
    Xqe_wf = has_qe_wf_data ? reduce(hcat, Cl_cross_qe_wf_sims) : nothing

    Lc,  C̄_aqe,  σ_aqe  = coarsen(ℓ_kk, Aqe;  edges=edges); σ_aqe  .*= σ_scale
    _,   C̄_agi,  σ_agi  = coarsen(ℓ_kk, Agi;  edges=edges); σ_agi  .*= σ_scale
    _,   C̄_xqe,  σ_xqe  = coarsen(ℓ_kk, Xqe;  edges=edges); σ_xqe  .*= σ_scale
    _,   C̄_xgi,  σ_xgi  = coarsen(ℓ_kk, Xgi;  edges=edges); σ_xgi  .*= σ_scale
    _,   C̄_true, _      = coarsen(ℓ_kk, Tmat; edges=edges)

    C̄_aqe_wf = σ_aqe_wf = σ_xqe_wf = C̄_xqe_wf = σ_th_aqe_wf = σ_th_xqe_wf = nothing
    if has_qe_wf_data
        _, C̄_aqe_wf, σ_aqe_wf = coarsen(ℓ_kk, Aqe_wf; edges=edges); σ_aqe_wf .*= σ_scale
        _, C̄_xqe_wf, σ_xqe_wf = coarsen(ℓ_kk, Xqe_wf; edges=edges); σ_xqe_wf .*= σ_scale
    end

    Neff         = @. (2Lc + 1) * ΔL * f_sky_paper
    σ_th_aqe     = @. sqrt(2 / Neff) * abs(C̄_aqe)
    σ_th_agi     = @. sqrt(2 / Neff) * abs(C̄_agi)
    σ_th_xqe     = @. sqrt(max(abs(C̄_true)*abs(C̄_aqe) + C̄_xqe^2, 0.0) / Neff)
    σ_th_xgi     = @. sqrt(max(abs(C̄_true)*abs(C̄_agi) + C̄_xgi^2, 0.0) / Neff)
    if has_qe_wf_data
        σ_th_aqe_wf = @. sqrt(2 / Neff) * abs(C̄_aqe_wf)
        σ_th_xqe_wf = @. sqrt(max(abs(C̄_true)*abs(C̄_aqe_wf) + C̄_xqe_wf^2, 0.0) / Neff)
    end

    println("\n  ΔL=$(Int(ΔL)), empirical σ (rescaled to f_sky=0.4):")
    println("  L       σ_auto_QE   σ_auto_GI   σ_cross_QE  σ_cross_GI  Knox_a_QE  Knox_a_GI")
    for b in eachindex(Lc)
        isnan(Lc[b]) && continue
        @printf "  L=%-5d %9.2e %9.2e  %9.2e %9.2e  %9.2e %9.2e\n" round(Int,Lc[b]) σ_aqe[b] σ_agi[b] σ_xqe[b] σ_xgi[b] σ_th_aqe[b] σ_th_agi[b]
    end

    return (Lc=Lc,
            C̄_xqe=C̄_xqe,    σ_xqe=σ_xqe,    σ_th_xqe=σ_th_xqe,
            C̄_aqe=C̄_aqe,    σ_aqe=σ_aqe,    σ_th_aqe=σ_th_aqe,
            C̄_xgi=C̄_xgi,    σ_xgi=σ_xgi,    σ_th_xgi=σ_th_xgi,
            C̄_agi=C̄_agi,    σ_agi=σ_agi,    σ_th_agi=σ_th_agi,
            C̄_xqe_wf=C̄_xqe_wf, σ_xqe_wf=σ_xqe_wf, σ_th_xqe_wf=σ_th_xqe_wf,
            C̄_aqe_wf=C̄_aqe_wf, σ_aqe_wf=σ_aqe_wf, σ_th_aqe_wf=σ_th_aqe_wf,
            C̄_true=C̄_true, nsims=nsims, has_qe_wf=has_qe_wf_data,
            W_qe=W_qe_raw, W_gi=W_gi, W_qe_wf=(has_wf ? W_qe_wf : nothing),
            ℓ_wl=ℓ_template)
end

s4 = process_noise_level(
    "results/WL_qe_gi_12000.jld2",
    "results/phi_maps_qe_gi_12000.jld2",
    "S4-like (1µK-arcmin, Lmax=12000)";
    Lmax=12000, beamFWHM=1.0, μKarcminT=1.0)

has_ul = isfile("results/WL_qe_gi_12000_ul.jld2") &&
         isfile("results/phi_maps_qe_gi_12000_ul.jld2")
if has_ul
    ul = process_noise_level(
        "results/WL_qe_gi_12000_ul.jld2",
        "results/phi_maps_qe_gi_12000_ul.jld2",
        "Ultra-low (0.1µK-arcmin, Lmax=20000)";
        Lmax=20000, beamFWHM=0.3, μKarcminT=0.1)
end

# ── Plot ──────────────────────────────────────────────────────────────────────
PythonPlot.rc("font",        family="serif", size=11)
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)
PythonPlot.rc("xtick.minor", width=0.5, size=2.5, visible=true)
PythonPlot.rc("ytick.minor", width=0.5, size=2.5, visible=true)

colours = Dict("qe" => "#D62728", "gi" => "#1F77B4", "qe_wf" => "#2CA02C")
labels  = Dict("qe" => "QE (raw)", "gi" => "GI",    "qe_wf" => "QE (WF)")

ncols = has_ul ? 2 : 1
fig, axs_arr = PythonPlot.subplots(4, ncols;
    figsize=(6.5*ncols, 14.0), sharex=true, constrained_layout=true)

getax(axs, row, col) = ncols == 1 ? axs[row] : axs[row, col]

function fill_sigma_panel(ax, data, is_auto; title_str="", show_legend=false)
    Lc = data.Lc
    entries = is_auto ?
        [("qe", data.σ_aqe, data.σ_th_aqe), ("gi", data.σ_agi, data.σ_th_agi)] :
        [("qe", data.σ_xqe, data.σ_th_xqe), ("gi", data.σ_xgi, data.σ_th_xgi)]
    for (key, σ_sim, σ_th) in entries
        msk = @. !isnan(Lc) & isfinite(σ_sim) & (σ_sim > 0)
        !any(msk) && continue
        ax.semilogy(Lc[msk], σ_sim[msk]; color=colours[key], linewidth=2.0, label=labels[key])
        msk_th = @. !isnan(Lc) & isfinite(σ_th) & (σ_th > 0)
        any(msk_th) && ax.semilogy(Lc[msk_th], σ_th[msk_th];
            color=colours[key], linestyle="--", linewidth=1.2, alpha=0.7)
    end
    # QE WF line (only if data is available)
    if data.has_qe_wf
        σ_wf = is_auto ? data.σ_aqe_wf : data.σ_xqe_wf
        σ_th_wf = is_auto ? data.σ_th_aqe_wf : data.σ_th_xqe_wf
        if σ_wf !== nothing
            msk = @. !isnan(Lc) & isfinite(σ_wf) & (σ_wf > 0)
            any(msk) && ax.semilogy(Lc[msk], σ_wf[msk]; color=colours["qe_wf"], linewidth=2.0, label=labels["qe_wf"])
            if σ_th_wf !== nothing
                msk_th = @. !isnan(Lc) & isfinite(σ_th_wf) & (σ_th_wf > 0)
                any(msk_th) && ax.semilogy(Lc[msk_th], σ_th_wf[msk_th];
                    color=colours["qe_wf"], linestyle="--", linewidth=1.2, alpha=0.7)
            end
        end
    end
    ax.set_xlim(5000, 12000)
    ax.set_ylim(1e-14, 1e-9)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    isempty(title_str) || ax.set_title(title_str, fontsize=9)
    show_legend && ax.legend(loc="upper left", frameon=false, fontsize=9,
        title="solid=empirical, dashed=Knox", title_fontsize=7)
end

function fill_mean_cross_panel(ax, data; show_legend=false)
    Lc = data.Lc
    msk_t = @. !isnan(Lc) & isfinite(data.C̄_true) & (data.C̄_true > 0)
    any(msk_t) && ax.semilogy(Lc[msk_t], data.C̄_true[msk_t];
        color="black", linestyle="--", linewidth=1.5, label=L"C_L^{\kappa\kappa}\,(true)")
    entries = [("qe", data.C̄_xqe), ("gi", data.C̄_xgi)]
    if data.has_qe_wf && data.C̄_xqe_wf !== nothing
        push!(entries, ("qe_wf", data.C̄_xqe_wf))
    end
    for (key, C̄_x) in entries
        msk = @. !isnan(Lc) & isfinite(C̄_x) & (abs(C̄_x) > 0)
        !any(msk) && continue
        ax.semilogy(Lc[msk], abs.(C̄_x[msk]); color=colours[key], linewidth=2.0, label=labels[key])
    end
    ax.set_xlim(5000, 12000)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    show_legend && ax.legend(loc="upper right", frameon=false, fontsize=8)
end

function fill_WL_panel(ax, data; title_str="", show_legend=false)
    ℓ = data.ℓ_wl
    # Plot W_L for each estimator vs ℓ, restricted to the plotted L range
    for (key, W) in [("qe", data.W_qe), ("gi", data.W_gi)]
        W === nothing && continue
        msk = @. !isnan(ℓ) & isfinite(W) & (ℓ >= 5000) & (ℓ <= 12000)
        !any(msk) && continue
        ax.plot(ℓ[msk], W[msk]; color=colours[key], linewidth=2.0, label=labels[key])
    end
    if data.has_qe_wf && data.W_qe_wf !== nothing
        W = data.W_qe_wf
        msk = @. !isnan(ℓ) & isfinite(W) & (ℓ >= 5000) & (ℓ <= 12000)
        any(msk) && ax.plot(ℓ[msk], W[msk]; color=colours["qe_wf"], linewidth=2.0, label=labels["qe_wf"])
    end
    ax.axhline(1.0; color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axhline(0.0; color="k",    linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlim(5000, 12000)
    ax.set_ylim(-0.2, 1.5)
    ax.spines["top"].set_visible(false)
    ax.spines["right"].set_visible(false)
    isempty(title_str) || ax.set_title(title_str, fontsize=9)
    show_legend && ax.legend(loc="upper right", frameon=false, fontsize=9)
end

# Row 0: σ[C_auto]
ax = getax(axs_arr, 0, 0)
fill_sigma_panel(ax, s4, true; title_str="1 µK-arcmin, Lmax=12000, $(s4.nsims) sims", show_legend=true)
ax.set_ylabel(L"\sigma[C_L^{\hat{\kappa}\hat{\kappa}}]", fontsize=12)
has_ul && fill_sigma_panel(getax(axs_arr, 0, 1), ul, true;
    title_str="0.1 µK-arcmin, Lmax=20000, $(ul.nsims) sims", show_legend=true)

# Row 1: σ[C_cross]
ax = getax(axs_arr, 1, 0)
fill_sigma_panel(ax, s4, false)
ax.set_ylabel(L"\sigma[C_L^{\kappa\hat{\kappa}}]", fontsize=12)
has_ul && fill_sigma_panel(getax(axs_arr, 1, 1), ul, false)

# Row 2: mean debiased cross (diagnostic: should overlay C_true if W_L correct)
ax = getax(axs_arr, 2, 0)
fill_mean_cross_panel(ax, s4; show_legend=true)
ax.set_ylabel(L"\bar{C}_L^{\kappa\hat{\kappa}}", fontsize=12)
if has_ul
    ax2 = getax(axs_arr, 2, 1)
    fill_mean_cross_panel(ax2, ul; show_legend=true)
end

# Row 3: empirical W_L transfer function (W=1 → unbiased; W<1 → estimator undershoots)
ax = getax(axs_arr, 3, 0)
fill_WL_panel(ax, s4; show_legend=true)
ax.set_ylabel(L"W_L = \langle C_L^{\phi_{\rm true},\hat\phi}\rangle / C_L^{\phi\phi}", fontsize=10)
ax.set_xlabel(L"L", fontsize=12)
if has_ul
    ax2 = getax(axs_arr, 3, 1)
    fill_WL_panel(ax2, ul; show_legend=false)
    ax2.set_xlabel(L"L", fontsize=12)
end

savepath = "results/qe_gi_sigma_12000.png"
fig.savefig(savepath; dpi=200)
println("\nSaved $savepath")
PythonPlot.plotclose("all")
