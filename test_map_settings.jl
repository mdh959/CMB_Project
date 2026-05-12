#!/usr/bin/env julia
#
# test_map_settings.jl
#
# Diagnostic: is MAP_joint underconverged, prior-suppressed, fundamentally
# different from GI, or converging to the same solution?
#
# 50-sim S4 (1 µK·arcmin, 1' beam, Lmax=12000) comparison.  Fixed settings; no scan.
# Also: 5-sim near-noiseless check (0.01 µK·arcmin, zero start) to test MAP→GI convergence.
#
# Conditions:
#   QE — quadratic estimate (Wiener-filtered, unlensed weights)
#   GI — gradient inversion (Boryana et al.)
#   A  — MAP_joint, QE warm start,              prior_deprojection_factor = 0
#   B  — MAP_joint, QE warm start,              prior_deprojection_factor = 0.5
#   C  — MAP_joint, smooth QE+GI hybrid start,  prior_deprojection_factor = 0
#
# Output: results/test_map_settings_deproj_sigma.jld2
#         results/map_deproj_sigma/*.pdf
#
# Run: julia test_map_settings.jl

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
import CMBLensing: m_rfft, m_irfft   # internal FFT helpers, not in public export list
using Statistics: mean, std
using Printf
using JLD2
using PyPlot

# ── GI estimator (Boryana et al. method) ──────────────────────────────────────
# Compute spatial gradients of a pixel-space field via FFT differentiation.
function _grad_fft(field)
    m = Map(field); proj = m.proj; Ny, _ = size(m.arr)
    F = m_rfft(m.arr, (1,2)); NyF, NxF = size(F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,  NxF)
    dTdx = m_irfft((im .* ℓx2D) .* F, Ny, (1,2))
    dTdy = m_irfft((im .* ℓy2D) .* F, Ny, (1,2))
    return dTdx, dTdy
end

# Fixed-gradient GI reconstruction (Hadzhiyska et al. 2019).
# φ̂(L) = −i [Lx FFT(gx T_hp) + Ly FFT(gy T_hp)] / (Lx² σ_xx + 2LxLy σ_xy + Ly² σ_yy)
# where gx,gy are gradients of LowPass(2000)*T and T_hp = HighPass(4000)*T.
function gi_estimate(ds; denom_floor_frac=1e-8)
    Lhp  = 4000; Lmax_gi = 20000; Lgrad = 2000

    m = Map(ds.d); proj = m.proj
    dTdx, dTdy = _grad_fft(Map(LowPass(Lgrad) * ds.d))

    T_lp = Map(LowPass(Lhp) * ds.d).arr
    T_hp = Map(ds.d).arr .- T_lp

    A_F = m_rfft(dTdx .* T_hp, (1,2))
    B_F = m_rfft(dTdy .* T_hp, (1,2))

    σ_xx = mean(dTdx .^ 2); σ_xy = mean(dTdx .* dTdy); σ_yy = mean(dTdy .^ 2)
    denom_thr = denom_floor_frac * max(σ_xx, σ_yy)

    NyF, NxF = size(A_F)
    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,  NxF)
    L2   = @. ℓx2D^2 + ℓy2D^2

    numer = @. -im * (ℓx2D * A_F + ℓy2D * B_F)
    # E[(L·∇T)²] denominator in Fourier space
    denom2D = @. ℓx2D^2 * σ_xx + 2*ℓx2D*ℓy2D * σ_xy + ℓy2D^2 * σ_yy

    # && is not broadcastable; use & with a separate Boolean mask
    L_mask = @. (L2 < Lmax_gi^2) & (abs(denom2D) > denom_thr)
    φ_F    = @. ifelse(L_mask, numer / denom2D, complex(0.0))
    return FlatFourier(φ_F, proj)
end

# ── Cf̃ preconditioner (matches production run) ───────────────────────────────
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

# ── Parameters ────────────────────────────────────────────────────────────────
# ── S4 parameters (main run) ──────────────────────────────────────────────────
const nsims     = 50
const seed0     = 1000      # seeds seed0+1 … seed0+nsims
const Lmax      = 12000
const beamFWHM  = 1.0
const μKarcminT = 1.0
const Δℓ        = 30
const STEPS     = 40
const αmax      = 0.3
const SNR_Lo    = 4000.0
const SNR_Hi    = 12000.0

# ── Near-noiseless UL check (zero start, separate block at end) ───────────────
const nsims_ul     = 5
const seed0_ul     = 9000   # seeds 9001–9005 (distinct from S4 run)
const μKarcminT_ul = 0.01
const beamFWHM_ul  = 0.3
const STEPS_ul     = 50
const αmax_ul      = 0.05

const ALL_KEYS = (:qe, :gi, :A, :B, :C)
const MAP_KEYS = (:A, :B, :C)

const FIG_DIR = "results/map_deproj_sigma"
mkpath(FIG_DIR)

const Cℓ_fid = camb(r=0.05, ℓmax=35000)
const load_kw = (
    Cℓ   = Cℓ_fid,
    Cℓn  = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax),
    θpix = 0.7438046267475303, T = Float64, Nside = 512,
    beamFWHM = beamFWHM, pol = :I,
    bandpass_mask = LowPass(Lmax),
    pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

println("MAP diagnostic — S4 $(μKarcminT) µK·arcmin, beam=$(beamFWHM)', Lmax=$(Lmax), αmax=$(αmax), $(nsims) sims, $(STEPS) steps")
println("=" ^ 70)

# ── Storage ───────────────────────────────────────────────────────────────────
cross_sims    = Dict(k => Vector{Float64}[] for k in ALL_KEYS)
auto_sims     = Dict(k => Vector{Float64}[] for k in ALL_KEYS)
true_sims     = Vector{Float64}[]
ell_axis      = nothing
# MAP-GI cross per MAP condition (for ρ(MAP_deb, GI_deb))
cross_map_gi  = Dict(k => Vector{Float64}[] for k in MAP_KEYS)
# MAP convergence histories
map_histories = Dict(k => Any[] for k in MAP_KEYS)
# ‖ϕ_MAP − ϕ_start‖_rms per sim per MAP condition
norm_diff     = Dict(k => Float64[] for k in MAP_KEYS)

# ── Smooth hybrid warm start ──────────────────────────────────────────────────
# Cosine taper QE→GI over L ∈ [3000, 5000]; GI used up to Lmax=12000.
# LowPass(4000; Δℓ=2000) gives w_QE; complement gives w_GI.
function make_hybrid_start(ϕqe, ϕgi)
    lp = LowPass(4000; Δℓ=2000)
    ϕ_raw = lp * (ϕqe - ϕgi) + ϕgi     # = w_QE*ϕqe + (1-w_QE)*ϕgi
    ϕ_map = Map(ϕ_raw)
    αsafe = min(get_max_lensing_step(zero(ϕ_map), ϕ_map) * 0.9, αmax)
    return αsafe * ϕ_raw
end

# ── Main simulation loop ──────────────────────────────────────────────────────
println("\nRunning sims...")
for s in 1:nsims
    seed = seed0 + s
    @printf "  Sim %2d/%d (seed=%d)  " s nsims seed; flush(stdout)

    r      = load_sim(; seed=seed, load_kw...)
    ds     = r.ds; ϕ_true = r.ϕ

    cl_true = Float64.(get_Cℓ(ϕ_true; Δℓ=Δℓ).Cℓ)
    push!(true_sims, cl_true)
    if ell_axis === nothing
        ell_axis = Float64.(collect(get_Cℓ(ϕ_true; Δℓ=Δℓ).ℓ))
    end

    # QE
    print("QE "); flush(stdout)
    ϕqe = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true).ϕqe
    push!(cross_sims[:qe], Float64.(get_Cℓ(ϕ_true, ϕqe; Δℓ=Δℓ).Cℓ))
    push!(auto_sims[:qe],  Float64.(get_Cℓ(ϕqe;          Δℓ=Δℓ).Cℓ))

    # GI
    print("GI "); flush(stdout)
    ϕgi = gi_estimate(ds)
    push!(cross_sims[:gi], Float64.(get_Cℓ(ϕ_true, ϕgi; Δℓ=Δℓ).Cℓ))
    push!(auto_sims[:gi],  Float64.(get_Cℓ(ϕgi;          Δℓ=Δℓ).Cℓ))

    ϕ_hybrid = make_hybrid_start(ϕqe, ϕgi)

    # MAP conditions
    for (key, ϕ_start, deproj) in [
            (:A, ϕqe,      0.0),
            (:B, ϕqe,      0.5),
            (:C, ϕ_hybrid, 0.0),
        ]
        @printf "MAP-%s " key; flush(stdout)
        try
            res = MAP_joint(ds, FieldTuple(ϕ=ϕ_start);
                nsteps    = STEPS, αmax = αmax,
                prior_deprojection_factor = deproj,
                conjgrad_kwargs = (tol=1e-3, nsteps=500),
                progress  = false,
                history_keys = (:total_logpdf, :α, :ΔΩ°_norm, :argmaxf_logpdf_history))
            ϕ_mj = res.ϕ
            push!(cross_sims[key],  Float64.(get_Cℓ(ϕ_true, ϕ_mj; Δℓ=Δℓ).Cℓ))
            push!(auto_sims[key],   Float64.(get_Cℓ(ϕ_mj;          Δℓ=Δℓ).Cℓ))
            # ρ(MAP_deb, GI_deb) = ρ(raw) since W_L factors cancel in correlation
            push!(cross_map_gi[key],Float64.(get_Cℓ(ϕgi, ϕ_mj; Δℓ=Δℓ).Cℓ))
            push!(norm_diff[key],   sqrt(mean(abs2.(Map(ϕ_mj - ϕ_start).arr))))
            push!(map_histories[key], res.history)
        catch err
            @printf "FAILED(%s) " string(err)[1:min(40,length(string(err)))]
        end
    end
    println("done")
end

println("\nSims done. Computing statistics...")

# ── Statistics ────────────────────────────────────────────────────────────────
snr_idx     = findall(x -> SNR_Lo ≤ x ≤ SNR_Hi, ell_axis)
C_true_mean = mean(reduce(hcat, true_sims), dims=2)[:]

struct CondStats
    W_L       :: Vector{Float64}   # transfer function ⟨C(ϕ̂,ϕ_true)⟩ / C_true
    ρ_L       :: Vector{Float64}   # correlation with truth
    N0        :: Vector{Float64}   # MC noise bias ⟨C_auto⟩ - W_L² C_true
    σ_auto    :: Vector{Float64}   # std of debiased auto / C_true
    σ_cross   :: Vector{Float64}   # std of raw cross / C_true
    nsim_k    :: Int
end

function compute_stats(cross_s, auto_s, C_true_mean)
    n = length(cross_s); n == 0 && return nothing
    cross_mat = reduce(hcat, cross_s)   # ℓ × n
    auto_mat  = reduce(hcat, auto_s)

    W_L = mean(cross_mat, dims=2)[:] ./ C_true_mean
    ρ_L = mean(cross_mat, dims=2)[:] ./ sqrt.(mean(auto_mat, dims=2)[:] .* C_true_mean)
    N0  = mean(auto_mat,  dims=2)[:] .- W_L.^2 .* C_true_mean

    # Auto: field-level debiasing  ϕ_deb = ϕ/W_L  →  C_auto_deb = (C_auto − N0)/W_L²
    W2     = max.(W_L, 1e-6).^2
    σ_auto = std(reduce(hcat, [(auto_mat[:,s] .- N0) ./ W2 for s in 1:n]), dims=2)[:] ./ C_true_mean

    # Cross: raw scatter (σ_cross = std[C(ϕ_true, ϕ̂)] / C_true)
    σ_cross = std(cross_mat, dims=2)[:] ./ C_true_mean

    return CondStats(W_L, ρ_L, N0, σ_auto, σ_cross, n)
end

stats = Dict{Symbol, Union{Nothing, CondStats}}()
for k in ALL_KEYS
    stats[k] = compute_stats(cross_sims[k], auto_sims[k], C_true_mean)
end

# ρ(MAP_deb, GI_deb) per L-bin per MAP condition
function compute_rho_mg(k)
    isempty(cross_map_gi[k]) && return fill(NaN, length(ell_axis))
    cm = reduce(hcat, cross_map_gi[k])
    am = reduce(hcat, auto_sims[k])
    ag = reduce(hcat, auto_sims[:gi])
    nc = min(size(cm,2), size(am,2), size(ag,2))
    mean(cm[:,1:nc], dims=2)[:] ./
        sqrt.(mean(am[:,1:nc], dims=2)[:] .* mean(ag[:,1:nc], dims=2)[:])
end

# ── Labels and colours ────────────────────────────────────────────────────────
const labels = Dict(
    :qe => "QE",
    :gi => "GI",
    :A  => "MAP-A (deproj=0)",
    :B  => "MAP-B (deproj=0.5)",
    :C  => "MAP-C (hybrid start)",
)
const colors = Dict(:qe => "C0", :gi => "C1", :A => "C2", :B => "C3", :C => "C4")
const lss    = Dict(:qe => "-",  :gi => "--", :A => "-",  :B => "--", :C => ":")

# ── Results tables ────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("Results — SNR band L=$(round(Int,SNR_Lo))–$(round(Int,SNR_Hi)), n=$nsims sims")
println("=" ^ 70)

bands = [(4000,6000),(6000,9000),(9000,12000)]

println("\nW_L and ρ_L:")
@printf "  %-24s  %6s  %6s  %6s  %6s  %6s\n" "Condition" "W(all)" "ρ(all)" "L4-6k" "L6-9k" "L9-12k"
println("  " * "-"^60)
for k in ALL_KEYS
    s = stats[k]; s === nothing && continue
    w_all = mean(s.W_L[snr_idx]); ρ_all = mean(s.ρ_L[snr_idx])
    w_bands = [begin
        idx = findall(x -> lo ≤ x ≤ hi, ell_axis)
        isempty(idx) ? NaN : mean(s.W_L[idx])
    end for (lo,hi) in bands]
    @printf "  %-24s  %6.3f  %6.3f  %6.3f  %6.3f  %6.3f\n" labels[k] w_all ρ_all w_bands[1] w_bands[2] w_bands[3]
end

println("\nDebiased σ_auto and raw σ_cross:")
@printf "  %-24s  %10s  %10s\n" "Condition" "σ_auto" "σ_cross"
println("  " * "-"^48)
for k in ALL_KEYS
    s = stats[k]; s === nothing && continue
    @printf "  %-24s  %10.4f  %10.4f\n" labels[k] mean(abs.(s.σ_auto[snr_idx])) mean(abs.(s.σ_cross[snr_idx]))
end

println("\nMAP–GI cross-correlation ρ(MAP_deb, GI_deb) in SNR band:")
@printf "  %-24s  %10s  %12s\n" "Condition" "ρ(MAP,GI)" "‖ϕ_MAP−ϕ₀‖_rms"
println("  " * "-"^50)
for k in MAP_KEYS
    isempty(cross_map_gi[k]) && continue
    ρ_mg = mean(compute_rho_mg(k)[snr_idx])
    nd   = isempty(norm_diff[k]) ? NaN : mean(norm_diff[k])
    @printf "  %-24s  %10.4f  %12.3e\n" labels[k] ρ_mg nd
end

# ── Convergence diagnostics ───────────────────────────────────────────────────
function convergence_report(key, hists)
    isempty(hists) && return
    nstep = length(hists[1]); nh = length(hists)
    lp_ps = [mean([h[i].total_logpdf                     for h in hists]) for i in 1:nstep]
    α_ps  = [mean([h[i].α                                for h in hists]) for i in 1:nstep]
    dn_ps = [mean([h[i].ΔΩ°_norm                        for h in hists]) for i in 1:nstep]
    cg_it = [mean([length(h[i].argmaxf_logpdf_history)  for h in hists]) for i in 1:nstep]
    cg_rs = [mean([h[i].argmaxf_logpdf_history[end].res for h in hists]) for i in 1:nstep]

    println("\n  MAP-$key ($nh sims, $nstep steps):")
    println("  Step |  Δlogpdf  |    α     |  ΔΩ°_norm  | CG iters |  CG res")
    println("  " * "-"^66)
    lp0 = lp_ps[1]
    for i in filter(j -> j ≤ nstep, [1,2,3,5,10,15,20,30,40])
        @printf "  %4d | %+9.2f | %8.4f | %10.3e | %8.1f | %10.3e\n" i (lp_ps[i]-lp0) α_ps[i] dn_ps[i] cg_it[i] cg_rs[i]
    end
    δ5 = lp_ps[end] - lp_ps[max(1,end-4)]
    @printf "  Last 5-step Δlogpdf = %+.2f  (%s)\n" δ5 (δ5 > 0.5 ? "STILL IMPROVING" : "levelled off")
    @printf "  Mean CG residual    = %.2e  (%s)\n" mean(cg_rs) (mean(cg_rs) > 1e-3 ? "ABOVE tol — CG not converging" : "below tol")
end

println("\n" * "=" ^ 70)
println("Convergence Diagnostics")
println("=" ^ 70)
for k in MAP_KEYS
    convergence_report(k, map_histories[k])
end

# ── Figures ───────────────────────────────────────────────────────────────────

# Fig 1: W_L and ρ_L vs L
fig, (ax1, ax2) = subplots(1, 2, figsize=(11,4))
for k in ALL_KEYS
    s = stats[k]; s === nothing && continue
    ax1.plot(ell_axis, s.W_L, color=colors[k], ls=lss[k], lw=1.5, label=labels[k])
    ax2.plot(ell_axis, s.ρ_L, color=colors[k], ls=lss[k], lw=1.5, label=labels[k])
end
ax1.axhline(1.0, color="k", lw=0.6, ls="--")
for ax in [ax1, ax2]
    ax.axvspan(SNR_Lo, SNR_Hi, alpha=0.07, color="k")
    ax.set_xlabel(L"$L$", fontsize=12); ax.set_xlim(0, Lmax); ax.legend(fontsize=8)
end
ax1.set_ylabel(L"$W_L$", fontsize=12); ax1.set_title("Transfer function")
ax2.set_ylabel(L"$\rho_L$", fontsize=12); ax2.set_title("Correlation with truth")
fig.suptitle("$(nsims) sims, $(μKarcminT) µK·arcmin", fontsize=10)
tight_layout(); savefig(joinpath(FIG_DIR, "W_rho_vs_L.pdf")); close(fig)

# Fig 2: σ_auto and σ_cross vs L
fig, (ax1, ax2) = subplots(1, 2, figsize=(11,4))
for k in ALL_KEYS
    s = stats[k]; s === nothing && continue
    ax1.plot(ell_axis, abs.(s.σ_auto),  color=colors[k], ls=lss[k], lw=1.5, label=labels[k])
    ax2.plot(ell_axis, abs.(s.σ_cross), color=colors[k], ls=lss[k], lw=1.5, label=labels[k])
end
for ax in [ax1, ax2]
    ax.axvspan(SNR_Lo, SNR_Hi, alpha=0.07, color="k")
    ax.set_xlabel(L"$L$", fontsize=12); ax.set_xlim(0, Lmax); ax.legend(fontsize=8)
end
ax1.set_ylabel(L"$\sigma_{\rm auto}\,/\,C_L^{\phi\phi,\rm fid}$", fontsize=11)
ax1.set_title("Debiased auto scatter  [std of (C_auto−N0)/W_L²]")
ax2.set_ylabel(L"$\sigma_{\rm cross}\,/\,C_L^{\phi\phi,\rm fid}$", fontsize=11)
ax2.set_title("Raw cross scatter  [std of C(ϕ_true,ϕ̂)]")
fig.suptitle("$(nsims) sims, $(μKarcminT) µK·arcmin", fontsize=10)
tight_layout(); savefig(joinpath(FIG_DIR, "sigma_vs_L.pdf")); close(fig)

# Fig 3: ρ(MAP_deb, GI_deb) vs L
if !isempty(cross_map_gi[:A])
    fig, ax = subplots(figsize=(7,4))
    for k in MAP_KEYS
        isempty(cross_map_gi[k]) && continue
        ax.plot(ell_axis, compute_rho_mg(k), color=colors[k], ls=lss[k], lw=1.5, label=labels[k])
    end
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.axvspan(SNR_Lo, SNR_Hi, alpha=0.07, color="k")
    ax.set_xlabel(L"$L$", fontsize=12)
    ax.set_ylabel(L"$\rho(\hat\phi_{\rm MAP,deb},\,\hat\phi_{\rm GI,deb})$", fontsize=11)
    ax.set_title("Cross-correlation between debiased MAP and GI")
    ax.set_xlim(0, Lmax); ax.legend(fontsize=8)
    tight_layout(); savefig(joinpath(FIG_DIR, "rho_map_gi_deb.pdf")); close(fig)
end

# Fig 4: MAP / GI ratios (W_L and ρ_L)
s_gi = stats[:gi]
if s_gi !== nothing && s_gi.nsim_k > 0
    fig, (ax1, ax2) = subplots(1, 2, figsize=(11,4))
    for k in MAP_KEYS
        s = stats[k]; s === nothing && continue
        ax1.plot(ell_axis, s.W_L ./ max.(s_gi.W_L, 1e-8),
                 color=colors[k], ls=lss[k], lw=1.5, label=labels[k])
        ax2.plot(ell_axis, s.ρ_L ./ max.(s_gi.ρ_L, 1e-8),
                 color=colors[k], ls=lss[k], lw=1.5, label=labels[k])
    end
    for ax in [ax1, ax2]
        ax.axhline(1.0, color="k", lw=0.6, ls="--")
        ax.axvspan(SNR_Lo, SNR_Hi, alpha=0.07, color="k")
        ax.set_xlabel(L"$L$", fontsize=12); ax.set_xlim(0, Lmax); ax.legend(fontsize=8)
    end
    ax1.set_ylabel(L"$W_L^{\rm MAP}/W_L^{\rm GI}$", fontsize=12); ax1.set_title("W_L ratio MAP/GI")
    ax2.set_ylabel(L"$\rho_L^{\rm MAP}/\rho_L^{\rm GI}$", fontsize=12); ax2.set_title("ρ_L ratio MAP/GI")
    tight_layout(); savefig(joinpath(FIG_DIR, "map_gi_ratios.pdf")); close(fig)
end

# Fig 5: convergence history
fig, axes = subplots(2, 2, figsize=(10,7))
for k in MAP_KEYS
    hists = map_histories[k]; isempty(hists) && continue
    nstep = length(hists[1]); steps = 1:nstep
    lp  = [mean([h[i].total_logpdf                    for h in hists]) for i in steps]
    α_  = [mean([h[i].α                               for h in hists]) for i in steps]
    dn  = [mean([h[i].ΔΩ°_norm                       for h in hists]) for i in steps]
    cgi = [mean([length(h[i].argmaxf_logpdf_history)  for h in hists]) for i in steps]
    axes[1,1].plot(steps, lp .- lp[1], color=colors[k], lw=1.5, label=labels[k])
    axes[1,2].plot(steps, α_,           color=colors[k], lw=1.5, label=labels[k])
    axes[2,1].plot(steps, dn,           color=colors[k], lw=1.5, label=labels[k])
    axes[2,2].plot(steps, cgi,          color=colors[k], lw=1.5, label=labels[k])
end
axes[1,1].set_ylabel("Δlogpdf");         axes[1,1].set_title("Total logpdf (relative to step 1)")
axes[1,2].set_ylabel(L"$\alpha$");       axes[1,2].set_title("Line-search step size")
axes[2,1].set_ylabel(L"$\Delta\Omega^\circ_{\rm norm}$"); axes[2,1].set_title("Gradient norm")
axes[2,2].set_ylabel("CG iterations");   axes[2,2].set_title("Inner CG iterations per step")
for ax in axes[:]; ax.set_xlabel("MAP step"); ax.legend(fontsize=8); end
fig.suptitle("MAP convergence diagnostics — mean over $(nsims) sims", fontsize=10)
tight_layout(); savefig(joinpath(FIG_DIR, "convergence.pdf")); close(fig)

println("\nFigures saved to $FIG_DIR/")

# ── Save ──────────────────────────────────────────────────────────────────────
out = "results/test_map_settings_deproj_sigma.jld2"
jldsave(out;
    ell_axis, snr_idx, C_true_mean,
    cross_sims, auto_sims, true_sims,
    cross_map_gi, norm_diff,
    map_histories_A = map_histories[:A],
    map_histories_B = map_histories[:B],
    map_histories_C = map_histories[:C],
    stats_qe = stats[:qe], stats_gi = stats[:gi],
    stats_A  = stats[:A],  stats_B  = stats[:B], stats_C = stats[:C],
)
println("Saved → $out")

# ── Conclusions ───────────────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("Diagnostic Conclusions")
println("=" ^ 70)

for k in MAP_KEYS
    hists = map_histories[k]; s = stats[k]
    (isempty(hists) || s === nothing) && continue
    nstep = length(hists[1])
    lp_trace = [mean([h[i].total_logpdf for h in hists]) for i in 1:nstep]
    δ5 = lp_trace[end] - lp_trace[max(1,end-4)]
    mean_α = mean(mean([h[i].α for h in hists]) for i in 1:nstep)
    W_snr  = mean(s.W_L[snr_idx])
    ρ_mg   = isempty(cross_map_gi[k]) ? NaN : mean(compute_rho_mg(k)[snr_idx])

    converged      = δ5 < 0.5
    prior_suppress = W_snr < 0.7
    optim_fail     = mean_α < 0.05
    like_gi        = !isnan(ρ_mg) && ρ_mg > 0.9

    println("\nMAP-$(k) ($(labels[k])):")
    @printf "  Converged?          %s  (last-5-step Δlogpdf = %+.2f)\n"  (converged      ? "YES" : "NO ")  δ5
    @printf "  Prior suppressed?   %s  (mean W_L in SNR band = %.3f)\n"  (prior_suppress ? "YES" : "NO ")  W_snr
    @printf "  Optimisation fail?  %s  (mean line-search α   = %.4f)\n"  (optim_fail     ? "YES" : "NO ")  mean_α
    @printf "  Converges to GI?    %s  (ρ(MAP_deb,GI_deb)   = %.3f)\n"  (like_gi        ? "YES" : "NO ")  ρ_mg
    @printf "  Warm-start effect?  %-3s (‖ϕ_MAP−ϕ_start‖_rms = %.2e)\n" "—" (isempty(norm_diff[k]) ? NaN : mean(norm_diff[k]))
end

# ── Near-noiseless UL convergence check ──────────────────────────────────────
# Zero warm start, 0.01 µK·arcmin, 50 steps, αmax=0.05.
# Goal: if MAP converges to GI here, the optimisation is sound; failure points
# to prior suppression or shell-crossing in the richer S4 landscape.
println("\n" * "=" ^ 70)
println("Near-noiseless UL check ($(μKarcminT_ul) µK·arcmin, $(nsims_ul) sims, zero start)")
println("=" ^ 70)

const load_kw_ul = (
    Cℓ   = Cℓ_fid,
    Cℓn  = noiseCℓs(μKarcminT=μKarcminT_ul, ℓknee=0, ℓmax=Lmax),
    θpix = 0.7438046267475303, T = Float64, Nside = 512,
    beamFWHM = beamFWHM_ul, pol = :I,
    bandpass_mask = LowPass(Lmax),
    pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

ul_ρ_map_gi  = Float64[]   # ρ(MAP_deb, GI_deb) per sim in SNR band
ul_ρ_map_true = Float64[]  # ρ(MAP, ϕ_true) per sim in SNR band
ul_ρ_gi_true  = Float64[]  # ρ(GI, ϕ_true) per sim in SNR band
ul_ell_axis   = nothing

for s in 1:nsims_ul
    seed = seed0_ul + s
    @printf "  UL sim %d/%d (seed=%d) " s nsims_ul seed; flush(stdout)

    r      = load_sim(; seed=seed, load_kw_ul...)
    ds     = r.ds; ϕ_true = r.ϕ
    ϕ_zero = zero(r.ϕ)

    if ul_ell_axis === nothing
        ul_ell_axis = Float64.(collect(get_Cℓ(ϕ_true; Δℓ=Δℓ).ℓ))
    end
    ul_snr_idx = findall(x -> SNR_Lo ≤ x ≤ SNR_Hi, ul_ell_axis)

    print("GI "); flush(stdout)
    ϕgi_ul = gi_estimate(ds)

    print("MAP "); flush(stdout)
    try
        res = MAP_joint(ds, FieldTuple(ϕ=ϕ_zero);
            nsteps    = STEPS_ul, αmax = αmax_ul,
            prior_deprojection_factor = 0.0,
            conjgrad_kwargs = (tol=1e-3, nsteps=500),
            progress  = false,
            history_keys = (:total_logpdf,))
        ϕ_mj = res.ϕ

        cl_x_mt = Float64.(get_Cℓ(ϕ_true, ϕ_mj;   Δℓ=Δℓ).Cℓ)
        cl_a_m  = Float64.(get_Cℓ(ϕ_mj;             Δℓ=Δℓ).Cℓ)
        cl_x_gt = Float64.(get_Cℓ(ϕ_true, ϕgi_ul;  Δℓ=Δℓ).Cℓ)
        cl_a_g  = Float64.(get_Cℓ(ϕgi_ul;            Δℓ=Δℓ).Cℓ)
        cl_a_t  = Float64.(get_Cℓ(ϕ_true;            Δℓ=Δℓ).Cℓ)
        cl_x_mg = Float64.(get_Cℓ(ϕgi_ul, ϕ_mj;    Δℓ=Δℓ).Cℓ)

        ρ_mt = mean(cl_x_mt[ul_snr_idx] ./ sqrt.(cl_a_m[ul_snr_idx] .* cl_a_t[ul_snr_idx]))
        ρ_gt = mean(cl_x_gt[ul_snr_idx] ./ sqrt.(cl_a_g[ul_snr_idx] .* cl_a_t[ul_snr_idx]))
        ρ_mg = mean(cl_x_mg[ul_snr_idx] ./ sqrt.(cl_a_m[ul_snr_idx] .* cl_a_g[ul_snr_idx]))

        push!(ul_ρ_map_true, ρ_mt)
        push!(ul_ρ_gi_true,  ρ_gt)
        push!(ul_ρ_map_gi,   ρ_mg)
        @printf "ρ(MAP,true)=%.3f  ρ(GI,true)=%.3f  ρ(MAP,GI)=%.3f\n" ρ_mt ρ_gt ρ_mg
    catch err
        @printf "FAILED(%s)\n" string(err)[1:min(60,length(string(err)))]
    end
end

if !isempty(ul_ρ_map_gi)
    println("\n  UL check summary (mean over $(length(ul_ρ_map_gi)) sims, SNR band):")
    @printf "  ρ(MAP, ϕ_true) = %.4f\n"  mean(ul_ρ_map_true)
    @printf "  ρ(GI,  ϕ_true) = %.4f\n"  mean(ul_ρ_gi_true)
    @printf "  ρ(MAP, GI)     = %.4f  (%s)\n" mean(ul_ρ_map_gi) (mean(ul_ρ_map_gi) > 0.95 ? "MAP ≈ GI ✓" : "MAP ≠ GI — further investigation needed")
end

println("\nDone.")
