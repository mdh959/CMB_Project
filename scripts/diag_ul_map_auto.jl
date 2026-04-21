#!/usr/bin/env julia
# scripts/diag_ul_map_auto.jl
#
# Diagnoses why UL MAP auto SNR is anomalously high.
#
# Checks:
#   1. W_mj (MAP transfer function) vs ℓ — is it ≈ 1 or anomalous?
#   2. Per-sim auto / cross / true C_kk for a sample of sims
#   3. Ratio C_auto_mj / C_true per band — bias indicator
#   4. std(C_auto_mj) vs std(C_cross_mj) vs std(C_true) — variance comparison
#
# Run from project root:
#   julia scripts/diag_ul_map_auto.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using CMBLensing, JLD2, Statistics, Printf, PythonPlot

include("../utils.jl")
using .Utils

const Cℓ    = camb(r=0.05, ℓmax=35000)
const θpix  = 0.7438046267475303
const Nside = 512
const seed0 = 1000
const Δℓ    = 100   # coarse bins for cleaner plot
const N_diag = 20   # sims to sample

load_kw_ul = (
    Cℓ=Cℓ, Cℓn=noiseCℓs(μKarcminT=0.1, ℓknee=0, ℓmax=12000),
    θpix=θpix, T=Float64, Nside=Nside, beamFWHM=0.3, pol=:I,
    bandpass_mask=LowPass(12000),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

wl_qe_file  = "results/WL_qe_gi_12000_ul.jld2"
wl_map_file = "results/WL_map_12000_ul.jld2"
phi_map_file = "results/phi_maps_map_12000_ul.jld2"

isfile(wl_map_file)  || error("Missing: $wl_map_file")
isfile(phi_map_file) || error("Missing: $phi_map_file")

# ── 1. Load W_mj and W_qe_raw ─────────────────────────────────────────────────
d_qe  = JLD2.load(wl_qe_file)
d_map = JLD2.load(wl_map_file)

ℓ_qe  = Float64.(d_qe["ℓ_template"])
W_qe  = Float64.(d_qe["W_qe_raw"])
W_gi  = Float64.(d_qe["W_gi_b"])
ℓ_map = Float64.(d_map["ℓ_template"])
W_mj  = Float64.(d_map["W_mj"])
n_map = get(d_map, "nsims_map_done", 0)

println("="^65)
println("UL MAP diagnostic  ($n_map MAP sims)")
println("="^65)
println()
println("  ℓ        W_QE_raw   W_GI_b    W_MAP_mj   W_MAP/W_QE")
for ℓ_c in [4000, 6000, 8000, 10000, 12000]
    iq = argmin(abs.(ℓ_qe  .- ℓ_c))
    im = argmin(abs.(ℓ_map .- ℓ_c))
    ratio = W_qe[iq] > 1e-6 ? W_mj[im] / W_qe[iq] : NaN
    @printf "  %-6d   %8.4f   %8.4f   %8.4f   %6.3f\n" ℓ_c W_qe[iq] W_gi[iq] W_mj[im] ratio
end
println()
println("  W_MAP/W_QE > 1 → MAP is OVERSHOOTING the warm start")
println("  W_MAP/W_QE ≈ 1 → MAP barely moved from warm start")
println("  W_MAP/W_QE < 1 → MAP partially reconstructs (normal for high L at S4)")

# ── 2. Metadata / wrap helper ─────────────────────────────────────────────────
(; ϕ) = load_sim(; seed=seed0+1, load_kw_ul...)
ϕ_ref = Map(ϕ)
wrap(arr) = typeof(ϕ_ref)(Float64.(arr), ϕ_ref.metadata)

WL_mj_cℓs = Cℓs(ℓ_map, W_mj)

# ── 3. Per-sim spectra ────────────────────────────────────────────────────────
map_sims = Int[]
jldopen(phi_map_file, "r") do f
    for k in keys(f)
        m = match(r"^sim_(\d+)$", k)
        m !== nothing && haskey(f, "sim_$(m.captures[1])/ϕ_mj") &&
            push!(map_sims, parse(Int, m.captures[1]))
    end
end
sort!(map_sims)
println("  Total MAP sims with ϕ_mj: $(length(map_sims))")

# Sample N_diag sims evenly
step = max(1, length(map_sims) ÷ N_diag)
sample_sims = map_sims[1:step:end][1:min(N_diag, end)]

ℓv = nothing
Cl_auto_arr   = Vector{Vector{Float64}}()
Cl_cross_arr  = Vector{Vector{Float64}}()
Cl_true_arr   = Vector{Vector{Float64}}()

println("\n  Loading $N_diag sample sims...")
jldopen(phi_map_file, "r") do f
    for s in sample_sims
        ϕt   = wrap(read(f, "sim_$s/ϕ_true"))
        ϕm_r = wrap(read(f, "sim_$s/ϕ_mj"))
        ϕm_d = debias_phi_with_WL(ϕm_r, WL_mj_cℓs; minW=1e-8)

        cl_tt = get_Cℓ(ϕt;      Δℓ=Δℓ)
        cl_aa = get_Cℓ(ϕm_d;    Δℓ=Δℓ)
        cl_xa = get_Cℓ(ϕt, ϕm_d; Δℓ=Δℓ)

        if ℓv === nothing
            ℓv = Float64.(collect(cl_tt.ℓ))
        end
        kfac = @. (ℓv^2 / 2)^2

        push!(Cl_true_arr,  kfac .* Float64.(cl_tt.Cℓ))
        push!(Cl_auto_arr,  kfac .* Float64.(cl_aa.Cℓ))
        push!(Cl_cross_arr, kfac .* Float64.(cl_xa.Cℓ))
        print("."); flush(stdout)
    end
end
println()

# ── 4. Band summary statistics ────────────────────────────────────────────────
bands = [(4000,6000), (6000,8000), (8000,10000), (10000,12000)]
println()
println("  Band      mean(auto/true)   mean(cross/true)   std(auto)/std(true)   std(cross)/std(true)")
for (lo, hi) in bands
    idx = findall(x -> lo <= x < hi, ℓv)
    isempty(idx) && continue
    auto_bands  = [mean(s[idx]) for s in Cl_auto_arr]
    cross_bands = [mean(s[idx]) for s in Cl_cross_arr]
    true_bands  = [mean(s[idx]) for s in Cl_true_arr]
    μt = mean(true_bands); μt == 0 && continue
    @printf "  %d–%d   %8.4f            %8.4f             %8.4f                 %8.4f\n" lo hi mean(auto_bands)/μt mean(cross_bands)/μt std(auto_bands)/std(true_bands) std(cross_bands)/std(true_bands)
end
println()
println("  auto/true > 1 → MAP auto is biased HIGH (overshooting)")
println("  std(auto)/std(true) < 1 → MAP auto variance too small (locked to warm start)")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
PythonPlot.rc("font", family="serif", size=10)
fig, axs = PythonPlot.subplots(1, 3; figsize=(15, 4.5), constrained_layout=true)

msk = @. isfinite(ℓv) & (ℓv >= 3500) & (ℓv <= 12500)
ℓp  = ℓv[msk]

# Panel 1: W_mj vs ℓ
ax = axs[0]
msk_q = @. isfinite(ℓ_qe) & (ℓ_qe >= 3500) & (ℓ_qe <= 12500)
msk_m = @. isfinite(ℓ_map) & (ℓ_map >= 3500) & (ℓ_map <= 12500)
ax.plot(ℓ_qe[msk_q], W_qe[msk_q]; color="#D62728", lw=1.5, label="QE raw")
ax.plot(ℓ_qe[msk_q], W_gi[msk_q]; color="#1F77B4", lw=1.5, label="GI")
ax.plot(ℓ_map[msk_m], W_mj[msk_m]; color="#9467BD", lw=2,   label="MAP joint")
ax.axhline(1; color="k", ls="--", lw=0.8)
ax.set_xlabel(L"L"); ax.set_ylabel(L"W_L")
ax.set_title("Transfer function W_L  (UL, 0.1 µK)", fontsize=10)
ax.legend(fontsize=8); ax.set_xlim(3500, 12500)

# Panel 2: mean spectra (auto, cross, true)
ax = axs[1]
Amat = reduce(hcat, Cl_auto_arr);  μA = mean(Amat; dims=2)[:]
Xmat = reduce(hcat, Cl_cross_arr); μX = mean(Xmat; dims=2)[:]
Tmat = reduce(hcat, Cl_true_arr);  μT = mean(Tmat; dims=2)[:]
pos = @. msk & (μT > 0) & (μA > 0) & (μX > 0)
ax.semilogy(ℓv[pos], μT[pos]; color="k",       ls="--", lw=1.5, label="true C_kk")
ax.semilogy(ℓv[pos], μA[pos]; color="#9467BD", lw=2,           label="MAP auto (debiased)")
ax.semilogy(ℓv[pos], μX[pos]; color="#2CA02C", lw=2,           label="MAP cross (debiased)")
ax.set_xlabel(L"L"); ax.set_ylabel(L"L^4 C_L^{\kappa\kappa} / 4")
ax.set_title("Mean spectra  ($N_diag sample sims)", fontsize=10)
ax.legend(fontsize=8); ax.set_xlim(3500, 12500)

# Panel 3: std of auto, cross, true
ax = axs[2]
σA = std(Amat; dims=2)[:]; σX = std(Xmat; dims=2)[:]; σT = std(Tmat; dims=2)[:]
pos2 = @. msk & (σT > 0) & (σA > 0) & (σX > 0)
ax.semilogy(ℓv[pos2], σA[pos2]; color="#9467BD", lw=2, label="std(MAP auto)")
ax.semilogy(ℓv[pos2], σX[pos2]; color="#2CA02C", lw=2, label="std(MAP cross)")
ax.semilogy(ℓv[pos2], σT[pos2]; color="k", ls="--", lw=1.5, label="std(true C_kk)")
ax.set_xlabel(L"L"); ax.set_ylabel("std across sims")
ax.set_title("Variance across sims", fontsize=10)
ax.legend(fontsize=8); ax.set_xlim(3500, 12500)

outpath = "results/diag_ul_map_auto.png"
fig.savefig(outpath; dpi=150)
PythonPlot.plotclose("all")
println("Saved: $outpath")