#!/usr/bin/env julia
# diag_gi_wl.jl
#
# Compares three GI variants on ~10 sims to measure W_L, σ_cross, σ_auto:
#   1. gi_estimate        Lhp=4000 (production)
#   2. gi_estimate_corrected  Lhp=4000 (per-pixel Wiener — theoretically optimal)
#   3. gi_estimate        Lhp=3000 (lower cut — tests whether extra modes help)
#
# Outputs: results/diagnose/diag_gi_wl.jld2 and diag_gi_wl.png

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using LinearAlgebra
using Statistics
using JLD2
using Printf
using PythonPlot

include("utils.jl")
using .Utils

# ── Config ────────────────────────────────────────────────────────────────────

const Cℓ_fid   = camb(r=0.05, ℓmax=35000)
const θpix     = 0.7438046267475303
const Nside    = 512
const pol      = :I
const seed0    = 1000
const nsims    = 10
const Lmax     = 12000
const Δℓ       = 500          # coarser bin — only 10 sims so σ estimates are noisy anyway
const μKarcminT = 1.0
const beamFWHM  = 1.0

const VARIANTS = [
    ("gi_4000",    (ds) -> gi_estimate(ds; Lhp=4000, Lmax=Lmax),           "GI Lhp=4000 (prod)"),
    ("gi_corr",    (ds) -> gi_estimate_corrected(ds; Lhp=4000, Lmax=Lmax), "GI corrected Lhp=4000"),
    ("gi_3000",    (ds) -> gi_estimate(ds; Lhp=3000, Lmax=Lmax),           "GI Lhp=3000"),
]

mkpath("results/diagnose")

# ── Run ───────────────────────────────────────────────────────────────────────

Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
load_kwargs = (
    Cℓ=Cℓ_fid, Cℓn=Cℓn, θpix=θpix, T=Float64, Nside=Nside,
    beamFWHM=beamFWHM, pol=pol, bandpass_mask=LowPass(Lmax),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

# Per-variant accumulators: R_L = C(φ_true, φ̂) / C(φ_true, φ_true) per sim
R_sims  = Dict(k => Vector{Vector{Float64}}() for (k,_,_) in VARIANTS)
Cx_sims = Dict(k => Vector{Vector{Float64}}() for (k,_,_) in VARIANTS)
Ca_sims = Dict(k => Vector{Vector{Float64}}() for (k,_,_) in VARIANTS)
ℓ_bins  = nothing

for s in 1:nsims
    seed = seed0 + s
    @printf "\nSim %d/%d (seed=%d)\n" s nsims seed
    (; ϕ, ds) = load_sim(; seed=seed, load_kwargs...)
    cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)
    global ℓ_bins
    ℓ_bins === nothing && (ℓ_bins = Float64.(collect(cl_tt.ℓ)))

    for (key, estimator, label) in VARIANTS
        print("  $label ... "); flush(stdout)
        t0 = time()
        ϕ̂ = try
            estimator(ds)
        catch err
            @warn "failed: $err"
            nothing
        end
        ϕ̂ === nothing && continue
        @printf "%.1fs\n" time()-t0

        ctt  = Float64.(cl_tt.Cℓ)
        ctx  = Float64.(get_Cℓ(ϕ, ϕ̂; Δℓ=Δℓ).Cℓ)
        caa  = Float64.(get_Cℓ(ϕ̂;    Δℓ=Δℓ).Cℓ)

        push!(R_sims[key],  @. ifelse(abs(ctt) > 0, ctx / ctt, 0.0))
        push!(Cx_sims[key], ctx)
        push!(Ca_sims[key], caa)
    end
end

# ── Compute W_L, σ_cross, σ_auto/W² ─────────────────────────────────────────

function stats(vv)
    isempty(vv) && return fill(NaN, length(ℓ_bins)), fill(NaN, length(ℓ_bins))
    mat = reduce(hcat, vv)   # Nbins × Nsims
    mean(mat; dims=2)[:], std(mat; dims=2)[:]
end

W_mean  = Dict{String,Vector{Float64}}()
W_std   = Dict{String,Vector{Float64}}()
σ_cross = Dict{String,Vector{Float64}}()
σ_auto  = Dict{String,Vector{Float64}}()

for (key, _, _) in VARIANTS
    W_mean[key],  W_std[key]  = stats(R_sims[key])
    _, σx          = stats(Cx_sims[key])
    W            = max.(W_mean[key], 1e-3)
    # σ_auto in units of W²: std of C_auto/W²
    Ca_deb_sims = [ca ./ W.^2 for ca in Ca_sims[key]]
    _, σa        = stats(Ca_deb_sims)
    σ_cross[key] = σx
    σ_auto[key]  = σa
end

jldsave("results/diagnose/diag_gi_wl.jld2";
    W_mean, W_std, σ_cross, σ_auto, ℓ_bins, VARIANTS=[(k,l) for (k,_,l) in VARIANTS],
    nsims, μKarcminT)
println("\nSaved results/diagnose/diag_gi_wl.jld2")

# ── Plot ──────────────────────────────────────────────────────────────────────

PythonPlot.rc("font", family="serif", size=10)
fig, axs = subplots(1, 3; figsize=(14, 4.5), constrained_layout=true)

CLR = ["C0", "C1", "C2"]
msk = ℓ_bins .>= 3000

for (i, (key, _, label)) in enumerate(VARIANTS)
    c = CLR[i]
    ℓ = ℓ_bins[msk]
    W = W_mean[key][msk]
    Ws = W_std[key][msk]
    axs[0].plot(ℓ, W; color=c, lw=2, label=label)
    axs[0].fill_between(ℓ, W .- Ws, W .+ Ws; color=c, alpha=0.2)
    axs[1].semilogy(ℓ, σ_cross[key][msk]; color=c, lw=2, label=label)
    axs[2].semilogy(ℓ, max.(σ_auto[key][msk], 1e-20); color=c, lw=2, label=label)
end

axs[0].axhline(1.0; color="k", ls=":", lw=0.8)
axs[0].set_title(L"Transfer function $W_L$")
axs[0].set_xlabel(L"L"); axs[0].set_ylabel(L"W_L")
axs[0].set_xlim(3000, Lmax); axs[0].legend(frameon=false, fontsize=8)

axs[1].set_title(L"$\sigma_{\rm cross}$ (MC, $n=%$nsims$ sims)")
axs[1].set_xlabel(L"L"); axs[1].set_ylabel(L"\sigma[C^{\kappa\hat\kappa}]")
axs[1].set_xlim(3000, Lmax); axs[1].legend(frameon=false, fontsize=8)

axs[2].set_title(L"$\sigma_{\rm auto}/W^2$ (debiased)")
axs[2].set_xlabel(L"L"); axs[2].set_ylabel(L"\sigma[C^{\hat\kappa\hat\kappa}/W^2]")
axs[2].set_xlim(3000, Lmax); axs[2].legend(frameon=false, fontsize=8)

fig.suptitle("GI variant comparison — S4 noise ($(μKarcminT) µK-arcmin), $(nsims) sims", fontsize=10)
fig.savefig("results/diagnose/diag_gi_wl.png"; dpi=160)
PythonPlot.plotclose("all")
println("Saved results/diagnose/diag_gi_wl.png\nDone.")
