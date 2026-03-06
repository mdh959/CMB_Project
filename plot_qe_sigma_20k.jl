#!/usr/bin/env julia
"""
plot_qe_sigma_20k.jl

Load per-sim raw κ spectra and W_L from run_qe_wl20k.jl.
Apply approximate spectral debiasing (divide cross by W_L, auto by W_L²).
Plot σ[C_L^{κ̂κ̂}] and σ[C_L^{κκ̂_deb}] vs Knox formula.
Same 3-panel structure and coarsening as plot_sigma_Ckk.jl.

Usage:  julia plot_qe_sigma_20k.jl
Input:  results/WL_qe_20000.jld2, results/spectra_qe_20000.jld2
Output: results/qe_sigma_20000.png, results/qe_sigma_20000.jld2
"""

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing       # for Cℓs interpolation
using Statistics: mean, std
using PythonPlot
using Printf
using JLD2

# ── Load ─────────────────────────────────────────────────────────────────────
const WL_file      = "results/WL_rawqe_20000.jld2"
const spectra_file = "results/spectra_rawqe_20000.jld2"

println("Loading W_L from $WL_file ...")
ℓ_template, W_qe_running = JLD2.load(WL_file, "ℓ_template", "W_qe_running")

println("Loading spectra from $spectra_file ...")
Cl_cross_raw_sims, Cl_auto_raw_sims, Cl_true_sims, ℓ_kk =
    JLD2.load(spectra_file,
        "Cl_cross_raw_sims", "Cl_auto_raw_sims", "Cl_true_sims", "ℓ_kk")

nsims = length(Cl_cross_raw_sims)
println("  $nsims sims, $(length(ℓ_kk)) ℓ bins")

# ── Debias raw spectra ────────────────────────────────────────────────────────
# Approximate spectral debiasing (valid when W_L varies slowly within Δℓ=30 bins):
#   Cl_cross_raw(s) = W_L(ℓ) × Cl_cross_deb(s)  →  Cl_cross_deb(s) = Cl_cross_raw(s) / W_L
#   Cl_auto_raw(s)  = W_L²(ℓ) × Cl_auto_deb(s)  →  Cl_auto_deb(s)  = Cl_auto_raw(s) / W_L²
#
# Bins where W_L is unreliably small (< minW) are excluded from plots.
# minW = 1e-3 is conservative: bins with W_L < 0.001 are removed.
const minW = 1e-3

WL_interp  = Cℓs(ℓ_template, W_qe_running)
W_L_at_kk  = Float64[WL_interp(ℓ) for ℓ in ℓ_kk]
W_L_floor  = @. max(abs(W_L_at_kk), minW)   # floor for division; NaN guard

# Mark bins where W_L < minW as NaN (unreliable — W_L has not converged here)
mask_ok = abs.(W_L_at_kk) .>= minW

Xmat = reduce(hcat, Cl_cross_raw_sims)   # (nℓ × nsims), raw cross κ spectra
Amat = reduce(hcat, Cl_auto_raw_sims)    # (nℓ × nsims), raw auto κ spectra
Tmat = reduce(hcat, Cl_true_sims)        # (nℓ × nsims), true κ auto

# Debias (element-wise along ℓ dimension)
Xmat_deb = Xmat ./ W_L_floor                 # Cl_cross_deb(s)
Amat_deb = Amat ./ W_L_floor.^2              # Cl_auto_deb(s)

# Zero out unreliable bins so they don't enter means/stds
Xmat_deb[.!mask_ok, :] .= NaN
Amat_deb[.!mask_ok, :] .= NaN

n_ok = sum(mask_ok)
println("W_L diagnostic:")
println("  Bins with |W_L| >= minW=$minW: $n_ok / $(length(ℓ_kk))")
for ℓ_check in [1000, 2000, 3000, 5000, 7000, 9000, 11000]
    idx = argmin(abs.(ℓ_kk .- ℓ_check))
    @printf "  W_L(ℓ≈%5d) = %8.4f  (ok=%s)\n" round(Int,ℓ_kk[idx]) W_L_at_kk[idx] mask_ok[idx]
end

# ── Geometry ─────────────────────────────────────────────────────────────────
const θpix        = 0.7438046267475303
const θpix_rad    = θpix * π / (180 * 60)
const f_sky_patch = (512 * θpix_rad)^2 / (4π)
const f_sky_paper = 0.4
const σ_scale     = sqrt(f_sky_patch / f_sky_paper)
println("σ_scale = $(round(σ_scale; sigdigits=4))  (rescale patch → f_sky=0.4)")

# ── Coarsen (identical to plot_sigma_Ckk.jl) ─────────────────────────────────
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

const ΔL   = 500.0   # same as plot_sigma_Ckk.jl
const edges = collect(2500 : Int(ΔL) : 10000 + Int(ΔL))

Lc,   C̄_cross, σ_cross = coarsen(ℓ_kk, Xmat_deb; edges=edges); σ_cross .*= σ_scale
Lc_a, C̄_auto,  σ_auto  = coarsen(ℓ_kk, Amat_deb; edges=edges); σ_auto  .*= σ_scale
Lc_t, C̄_true,  _       = coarsen(ℓ_kk, Tmat;     edges=edges)

# ── Knox formulas (identical to plot_sigma_Ckk.jl) ───────────────────────────
# C^κκ ≈ mean true κ auto-spectrum per band (best empirical estimate)
Cs = C̄_true
σ_th_auto  = @. sqrt(2 / ((2Lc_a+1)*ΔL*f_sky_paper)) * abs(C̄_auto)
σ_th_cross = @. sqrt(max(abs(Cs)*abs(C̄_auto) + C̄_cross^2, 0.0) / ((2Lc+1)*ΔL*f_sky_paper))

# ── Diagnostics ──────────────────────────────────────────────────────────────
println("\n  ΔL=$(Int(ΔL)), empirical σ rescaled to f_sky=0.4:")
println("  L      C̄_cross   C̄_auto    C̄_true    σ_cross(emp)  σ_cross(Knox)  ratio")
for b in eachindex(Lc)
    isnan(Lc[b]) && continue
    r = isfinite(σ_th_cross[b]) && σ_th_cross[b] > 0 ? σ_cross[b]/σ_th_cross[b] : NaN
    @printf "  L=%-5d %9.2e %9.2e %9.2e  %9.2e    %9.2e   %.3f\n" round(Int,Lc[b]) C̄_cross[b] C̄_auto[b] C̄_true[b] σ_cross[b] σ_th_cross[b] r
end

# ── Save ─────────────────────────────────────────────────────────────────────
jldsave("results/rawqe_sigma_20000.jld2";
    ℓ_kk, Lc, C̄_cross, C̄_auto, C̄_true, σ_cross, σ_auto, σ_th_cross, σ_th_auto,
    W_L_at_kk, f_sky_patch, f_sky_paper, σ_scale, nsims, ΔL, mask_ok)
println("\nSaved results/rawqe_sigma_20000.jld2")

# ── Plot — 3 panels matching plot_sigma_Ckk.jl ───────────────────────────────
PythonPlot.rc("font",        family="serif", size=11)
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)
PythonPlot.rc("xtick.minor", width=0.5, size=2.5, visible=true)
PythonPlot.rc("ytick.minor", width=0.5, size=2.5, visible=true)

col = "#D62728"
fig, axs = PythonPlot.subplots(3, 1; figsize=(6.5, 10.5), sharex=true,
    constrained_layout=true,
    gridspec_kw=Dict("height_ratios"=>[2, 2, 2], "hspace"=>0.0))
ax_auto  = axs[0]
ax_cross = axs[1]
ax_mean  = axs[2]

# Panel 1: σ[C_L^{κ̂_deb κ̂_deb}]
let msk = @. !isnan(Lc_a) & isfinite(σ_auto) & (σ_auto > 0)
    any(msk) && ax_auto.semilogy(Lc_a[msk], σ_auto[msk];
        color=col, linewidth=2.0, label="QE (raw, LowPass(20000))")
end
let msk = @. !isnan(Lc_a) & isfinite(σ_th_auto) & (σ_th_auto > 0)
    any(msk) && ax_auto.semilogy(Lc_a[msk], σ_th_auto[msk];
        color=col, linestyle="--", linewidth=1.5, alpha=0.8, label="Knox")
end
ax_auto.set_ylabel(L"\sigma[C_L^{\hat{\kappa}_\mathrm{deb}\hat{\kappa}_\mathrm{deb}}]", fontsize=12)
ax_auto.legend(frameon=false, fontsize=9, title="solid=empirical, dashed=Knox", title_fontsize=7)
ax_auto.set_title("QE (raw, debiased) — LowPass(20000), 1µK-arcmin, 1' beam, $nsims sims", fontsize=9)
ax_auto.set_xlim(5000, 11000); ax_auto.set_ylim(1e-14, 1e-9)
ax_auto.spines["top"].set_visible(false); ax_auto.spines["right"].set_visible(false)

# Panel 2: σ[C_L^{κ κ̂_deb}]
let msk = @. !isnan(Lc) & isfinite(σ_cross) & (σ_cross > 0)
    any(msk) && ax_cross.semilogy(Lc[msk], σ_cross[msk];
        color=col, linewidth=2.0, label="QE (raw, LowPass(20000))")
end
let msk = @. !isnan(Lc) & isfinite(σ_th_cross) & (σ_th_cross > 0)
    any(msk) && ax_cross.semilogy(Lc[msk], σ_th_cross[msk];
        color=col, linestyle="--", linewidth=1.5, alpha=0.8, label="Knox")
end
ax_cross.set_ylabel(L"\sigma[C_L^{\kappa\hat{\kappa}_\mathrm{deb}}]", fontsize=12)
ax_cross.legend(frameon=false, fontsize=9, title="solid=empirical, dashed=Knox", title_fontsize=7)
ax_cross.set_xlim(3000, 11000); ax_cross.set_ylim(1e-14, 1e-9)
ax_cross.spines["top"].set_visible(false); ax_cross.spines["right"].set_visible(false)

# Panel 3: mean C(κ_true, κ̂_deb) vs mean C^κκ (should overlap if debiasing is correct)
let msk = @. !isnan(Lc) & isfinite(C̄_cross) & (C̄_cross > 0)
    any(msk) && ax_mean.semilogy(Lc[msk], C̄_cross[msk];
        color=col, linewidth=2.0,
        label=L"\langle C_L^{\kappa\hat{\kappa}_\mathrm{deb}}\rangle")
end
let msk = @. !isnan(Lc_t) & isfinite(C̄_true) & (C̄_true > 0)
    any(msk) && ax_mean.semilogy(Lc_t[msk], C̄_true[msk];
        color="k", linestyle="--", linewidth=1.5,
        label=L"C_L^{\kappa\kappa}\ (\mathrm{true})")
end
ax_mean.set_xlabel(L"L", fontsize=12)
ax_mean.set_ylabel(L"\langle C_L^{\kappa\hat{\kappa}_\mathrm{deb}}\rangle", fontsize=12)
ax_mean.legend(frameon=false, fontsize=9)
ax_mean.set_xlim(5000, 11000)
ax_mean.spines["top"].set_visible(false); ax_mean.spines["right"].set_visible(false)

savepath = "results/rawqe_sigma_20000.png"
fig.savefig(savepath; dpi=200)
println("Saved $savepath")
PythonPlot.plotclose("all")
