#!/usr/bin/env julia
# ρ_L diagnostic on a genuinely small patch.
#
# Each sim draws a fresh N_CROP×N_CROP T map, runs QE + MAP_joint + MAP_marg
# on that small dataset, and computes
#
#   ρ_L = C_L^{ϕ_true, ϕ̂} / sqrt(C_L^{ϕϕ} × C_L^{ϕ̂ϕ̂})
#
# ρ is formed from mean spectra (not mean of per-sim ρ) to avoid the
# mean-of-ratios bias. Per-sim ρ is computed separately for SEM error bars.
#
# Testing MAP's advantage in the sparse-data regime: with very few T modes
# available, MAP's full posterior (with prior C_ℓ^{ϕϕ}) should outperform
# QE's perturbative estimator significantly.

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using Statistics: mean, std
using PythonPlot
using Printf

# MAP_marg internally calls set_distributed_dataset; define a serial fallback if absent
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset()  = _DIST_DS[]
    end
end

# ── Parameters ────────────────────────────────────────────────────────────────
const Cℓ            = camb(r=0.05, ℓmax=21000)
const Cℓn           = noiseCℓs(μKarcminT=1.0, ℓknee=0)
const θpix          = 0.7438046267475303   # arcmin/pixel
const pol           = :I
const beamFWHM      = 1.0
const bandpass_mask = LowPass(8000)
const N_CROP        = 32    # pixels per side; try 32, 64, 128
const Δℓ            = 500   # bin width
const nsims         = 100
const seed0         = 1000

const θpix_rad  = θpix * π / 10800
const L_fund    = round(Int, 2π / (N_CROP * θpix_rad))
const L_nyq     = round(Int, π / θpix_rad)
const patch_deg = round(N_CROP * θpix / 60; digits=2)

println("Patch:   $(N_CROP)×$(N_CROP) pixels = $(patch_deg)° × $(patch_deg)°")
println("L_fund ≈ $L_fund   L_nyq ≈ $L_nyq   Δℓ=$Δℓ")

# ── Spectrum accumulators ──────────────────────────────────────────────────────
# Average spectra across sims first, then form ρ (avoids mean-of-ratios bias).
cltt_sum  = nothing                    # ϕ_true auto (shared)
clrr_q = nothing; cltr_q = nothing    # QE
clrr_j = nothing; cltr_j = nothing    # MAP joint
clrr_m = nothing; cltr_m = nothing    # MAP marg
ℓ_vec  = nothing

# Per-sim ρ for SEM estimate
ρ_per_sim_q = Vector{Vector{Float64}}()
ρ_per_sim_j = Vector{Vector{Float64}}()
ρ_per_sim_m = Vector{Vector{Float64}}()
nsims_done  = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
for s in 1:nsims
    seed = seed0 + s
    @printf "→ Sim %d/%d  (seed=%d)\n" s nsims seed

    global ℓ_vec, cltt_sum, clrr_q, cltr_q, clrr_j, cltr_j, clrr_m, cltr_m, nsims_done
    local ds, ϕ, ϕqe, ϕ_marg, ϕ_joint

    try
        (; ϕ, ds) = load_sim(
            seed=seed, Cℓ=Cℓ, Cℓn=Cℓn,
            θpix=θpix, T=Float64, Nside=N_CROP,
            beamFWHM=beamFWHM, pol=pol,
            bandpass_mask=bandpass_mask,
            pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )
    catch err; println("  load_sim failed: $err — skip"); continue; end

    try
        ϕqe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true).ϕqe
    catch err; println("  QE failed: $err — skip"); continue; end

    try
        ϕ_marg, _ = MAP_marg(
            ds; ϕstart=ϕqe, nsteps=15, progress=false,
            conjgrad_kwargs=(tol=1e-3, nsteps=100), pmap=map,
        )
    catch err; println("  MAP_marg failed: $err — skip"); continue; end

    try
        ϕ_joint = MAP_joint(
            ds, FieldTuple(ϕ=ϕqe); nsteps=15, progress=false,
            conjgrad_kwargs=(tol=1e-3, nsteps=100),
        ).ϕ
    catch err; println("  MAP_joint failed: $err — skip"); continue; end

    # ── spectra ───────────────────────────────────────────────────────────────
    cl_tt_obj = get_Cℓ(ϕ; Δℓ=Δℓ)
    tt = Float64.(cl_tt_obj.Cℓ)
    if ℓ_vec === nothing
        ℓ_vec = Float64.(collect(cl_tt_obj.ℓ))
    end
    qq = Float64.(get_Cℓ(ϕqe;         Δℓ=Δℓ).Cℓ)
    jj = Float64.(get_Cℓ(ϕ_joint;     Δℓ=Δℓ).Cℓ)
    mm = Float64.(get_Cℓ(ϕ_marg;      Δℓ=Δℓ).Cℓ)
    tq = Float64.(get_Cℓ(ϕ, ϕqe;     Δℓ=Δℓ).Cℓ)
    tj = Float64.(get_Cℓ(ϕ, ϕ_joint; Δℓ=Δℓ).Cℓ)
    tm = Float64.(get_Cℓ(ϕ, ϕ_marg;  Δℓ=Δℓ).Cℓ)

    # accumulate
    cltt_sum = cltt_sum === nothing ? tt : cltt_sum .+ tt
    clrr_q   = clrr_q   === nothing ? qq : clrr_q   .+ qq
    cltr_q   = cltr_q   === nothing ? tq : cltr_q   .+ tq
    clrr_j   = clrr_j   === nothing ? jj : clrr_j   .+ jj
    cltr_j   = cltr_j   === nothing ? tj : cltr_j   .+ tj
    clrr_m   = clrr_m   === nothing ? mm : clrr_m   .+ mm
    cltr_m   = cltr_m   === nothing ? tm : cltr_m   .+ tm

    # per-sim ρ for error bars
    ρq = clamp.(tq ./ max.(sqrt.(max.(tt .* qq, 0.0)), 1e-30), -1.0, 1.0)
    ρj = clamp.(tj ./ max.(sqrt.(max.(tt .* jj, 0.0)), 1e-30), -1.0, 1.0)
    ρm = clamp.(tm ./ max.(sqrt.(max.(tt .* mm, 0.0)), 1e-30), -1.0, 1.0)
    push!(ρ_per_sim_q, ρq); push!(ρ_per_sim_j, ρj); push!(ρ_per_sim_m, ρm)

    nsims_done += 1
    for (nm, r) in [("qe",ρq), ("joint",ρj), ("marg",ρm)]
        g = filter(isfinite, r); isempty(g) || @printf "  %-6s  ρ̄=%.3f\n" nm mean(g)
    end
end

println("\n$nsims_done / $nsims sims completed.")
nsims_done == 0 && error("No sims succeeded.")

# ── ρ from mean spectra (unbiased estimator) ──────────────────────────────────
n = Float64(nsims_done)
function rho_from_mean(tt_sum, rr_sum, tr_sum)
    μtt = tt_sum ./ n;  μrr = rr_sum ./ n;  μtr = tr_sum ./ n
    clamp.(μtr ./ max.(sqrt.(max.(μtt .* μrr, 0.0)), 1e-30), -1.0, 1.0)
end
ρ_q = rho_from_mean(cltt_sum, clrr_q, cltr_q)
ρ_j = rho_from_mean(cltt_sum, clrr_j, cltr_j)
ρ_m = rho_from_mean(cltt_sum, clrr_m, cltr_m)

# SEM from per-sim ρ spread
sem_q = std(reduce(hcat, ρ_per_sim_q); dims=2)[:] ./ sqrt(nsims_done)
sem_j = std(reduce(hcat, ρ_per_sim_j); dims=2)[:] ./ sqrt(nsims_done)
sem_m = std(reduce(hcat, ρ_per_sim_m); dims=2)[:] ./ sqrt(nsims_done)

# mask above 85% Nyquist (beyond this pixelization dominates)
valid = @. isfinite(ℓ_vec) & (ℓ_vec < 0.85 * L_nyq)

# ── Plot ──────────────────────────────────────────────────────────────────────
PythonPlot.rc("font",  family="serif", size=11)
PythonPlot.rc("axes",  linewidth=0.8)
PythonPlot.rc("xtick", direction="in", top=true)
PythonPlot.rc("ytick", direction="in", right=true)

fig, ax = PythonPlot.subplots(1, 1; figsize=(6, 4), constrained_layout=true)

for (ρ, sem, col, lab) in [
        (ρ_q, sem_q, "#D62728", "QE (WF)"),
        (ρ_j, sem_j, "#1F77B4", "MAP joint"),
        (ρ_m, sem_m, "#2CA02C", "MAP marg"),
    ]
    msk = valid .& isfinite.(ρ)
    ax.plot(ℓ_vec[msk], ρ[msk]; color=col, label=lab, linewidth=2.0)
    ax.fill_between(ℓ_vec[msk], (ρ.-sem)[msk], (ρ.+sem)[msk]; color=col, alpha=0.15)
end

ax.axhline(1; color="k", linewidth=0.7, linestyle=":")
ax.axhline(0; color="k", linewidth=0.5, linestyle="--", alpha=0.4)
ax.set_xlabel(L"L")
ax.set_ylabel(L"\langle \rho_L \rangle")
ax.set_ylim(-0.1, 1.15)
ax.set_title("$nsims_done sims · $(N_CROP)×$(N_CROP) patch ($(patch_deg)°) · Δℓ=$Δℓ", fontsize=9)
ax.legend(frameon=false, fontsize=9)
ax.spines["top"].set_visible(false)
ax.spines["right"].set_visible(false)

outfile = "results/rho_L_patch$(N_CROP)px.png"
fig.savefig(outfile; dpi=150)
println("Saved $outfile")
PythonPlot.plotclose("all")
