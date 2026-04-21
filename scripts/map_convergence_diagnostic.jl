#!/usr/bin/env julia
# scripts/map_convergence_diagnostic.jl
#
# Checks whether MAP_joint has converged at MAPJ_STEPS=20.
# Does NOT rerun all 800 sims — reruns N_test sims with N_steps_ext steps.
#
# Output:
#   PART 1 — W_L table: QE_WF vs GI_B vs MAP (20-step, from file)
#   PART 2 — logpdf convergence plot (all test sims overlaid)
#   PART 3 — ΔW_L = W(N_steps_ext) − W(20 steps) at L=5000-11000
#             If ΔW is large, 20 steps is insufficient.
#
# Run from project root: julia scripts/map_convergence_diagnostic.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using CMBLensing, JLD2, Statistics, Printf, PythonPlot

include("../utils.jl")
using .Utils

# ── Settings — must match S4 block in run_qe_gi_wl12k.jl ────────────────────
const Cℓ        = camb(r=0.05, ℓmax=35000)
const Cℓn_s4    = noiseCℓs(μKarcminT=1.0, ℓknee=0, ℓmax=12000)
const θpix      = 0.7438046267475303
const Nside     = 512
const seed0     = 1000
const N_test    = 5     # sims to rerun
const N_steps_ext = 40  # extended step count (current is 20)

load_kw = (
    Cℓ=Cℓ, Cℓn=Cℓn_s4, θpix=θpix, T=Float64, Nside=Nside,
    beamFWHM=1.0, pol=:I, bandpass_mask=LowPass(12000),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

safe_div(a, b) = @. ifelse(abs(b) > 0.0, a / b, 0.0)

# ── PART 1: W_L table ────────────────────────────────────────────────────────
println("="^65)
println("PART 1: W_L at key scales  (S4: 1µK-arcmin, Lmax=12000)")
println("="^65)

d_qe  = JLD2.load("results/WL_qe_gi_12000.jld2")
d_map = JLD2.load("results/WL_map_12000.jld2")

ℓ_qe  = Float64.(d_qe["ℓ_template"])
W_qe  = Float64.(d_qe["W_qe_wf"])
W_gi  = Float64.(d_qe["W_gi_b"])
ℓ_map = Float64.(d_map["ℓ_template"])
W_mj  = Float64.(d_map["W_mj"])
n_map = get(d_map, "nsims_map_done", 0)

println("  MAP W_mj from $n_map sims (20 steps per sim)")
println("  GI W_gi_b from $(d_qe["nsims_completed"]) sims")
println()
println("  ℓ        W_QE_WF    W_GI_B     W_MAP_20   W_MAP/W_QE")
for ℓ_c in [1000, 3000, 5000, 7000, 9000, 11000]
    iq = argmin(abs.(ℓ_qe  .- ℓ_c))
    im = argmin(abs.(ℓ_map .- ℓ_c))
    ratio = W_qe[iq] > 1e-6 ? W_mj[im] / W_qe[iq] : NaN
    @printf "  %-6d   %8.4f   %8.4f   %8.4f   %6.3f\n" ℓ_c W_qe[iq] W_gi[iq] W_mj[im] ratio
end
println()
println("  Interpretation:")
println("  W_MAP/W_QE ≈ 1.0 → MAP ≈ warm-start (not converged beyond QE)")
println("  W_MAP/W_QE > 1.0 → MAP has improved on QE (converging)")
println("  W_GI >> W_MAP at L>5000 → GI outperforms MAP at those scales")

# ── PART 2 & 3: rerun N_test sims with N_steps_ext ──────────────────────────
println("\n" * "="^65)
println("PART 2+3: Rerunning $N_test sims with $N_steps_ext steps")
println("="^65)

map_phi_file = "results/phi_maps_map_12000.jld2"

# Pick first N_test sims that have a stored ϕ_mj
test_sims = Int[]
jldopen(map_phi_file, "r") do f
    all_s = sort([parse(Int, match(r"^sim_(\d+)$", k).captures[1])
                  for k in keys(f) if occursin(r"^sim_\d+$", k)])
    for s in all_s
        haskey(f, "sim_$s/ϕ_mj") && push!(test_sims, s)
        length(test_sims) >= N_test && break
    end
end
println("Test sims: $test_sims")

# Projection metadata
meta_seed = seed0 + test_sims[1]
ϕ_meta = load_sim(; seed=meta_seed, load_kw...).ϕ
ϕ_ref = Map(ϕ_meta)
wrap(arr) = typeof(ϕ_ref)(Float64.(arr), ϕ_ref.metadata)

fig, axes = subplots(1, 2; figsize=(12, 5), constrained_layout=true)
ax_pdf = axes[0]
ax_wl  = axes[1]

ΔW_rms = Float64[]

for (k, s) in enumerate(test_sims)
    seed = seed0 + s
    print("\n  Sim $s/$N_test (seed=$seed) ... ")
    flush(stdout)

    # Load stored 20-step phi
    ϕ_mj_20 = jldopen(map_phi_file, "r") do f; wrap(read(f, "sim_$s/ϕ_mj")); end

    # Fresh load + warm start
    (; ϕ, ds) = load_sim(; seed=seed, load_kw...)
    ϕqe_ws = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true).ϕqe

    # Run extended MAP (skip sim if CG diverges)
    res = try
        MAP_joint(ds, FieldTuple(ϕ=ϕqe_ws);
                  nsteps=N_steps_ext,
                  conjgrad_kwargs=(tol=1e-3, nsteps=200),
                  progress=false, history_keys=(:logpdf,))
    catch e
        print("  SKIPPED ($(typeof(e)))")
        flush(stdout)
        continue
    end
    ϕ_mj_ext = res.ϕ

    # logpdf trajectory
    lpdf = Float64[h.logpdf for h in res.history]
    ax_pdf.plot(1:length(lpdf), lpdf; alpha=0.75, label="sim $s")

    l20  = lpdf[min(20, end)]
    lfin = lpdf[end]
    Δl   = lfin - l20
    @printf "logpdf: step20=%.3f  step%d=%.3f  Δ=%.3f" l20 N_steps_ext lfin Δl
    Δl < 0.1 && print("  (converged)")
    Δl > 5.0 && print("  *** STILL IMPROVING ***")
    flush(stdout)

    # Transfer function: 20-step vs extended
    cl_tt   = get_Cℓ(ϕ; Δℓ=30)
    ℓv      = Float64.(collect(cl_tt.ℓ))
    ctt     = Float64.(cl_tt.Cℓ)
    R20     = safe_div(Float64.(get_Cℓ(ϕ, ϕ_mj_20;  Δℓ=30).Cℓ), ctt)
    R_ext   = safe_div(Float64.(get_Cℓ(ϕ, ϕ_mj_ext; Δℓ=30).Cℓ), ctt)
    msk     = @. isfinite(ℓv) & (ℓv >= 4000) & (ℓv <= 12000)
    ΔR      = R_ext[msk] .- R20[msk]
    push!(ΔW_rms, sqrt(mean(ΔR .^ 2)))
    ax_wl.plot(ℓv[msk], ΔR; alpha=0.75, label="sim $s")
end

# ── Finalise logpdf plot ─────────────────────────────────────────────────────
ax_pdf.axvline(20; color="red", ls="--", lw=1.8, label="current 20 steps")
ax_pdf.set_xlabel("MAP outer step", fontsize=11)
ax_pdf.set_ylabel("log P(ϕ | d)", fontsize=11)
ax_pdf.set_title("MAP convergence — S4 (1µK-arcmin)", fontsize=12)
ax_pdf.set_xlim(1, N_steps_ext)
ax_pdf.legend(fontsize=9)

# ── Finalise ΔW_L plot ───────────────────────────────────────────────────────
ax_wl.axhline(0; color="k", ls="--", lw=0.8)
ax_wl.set_xlabel(raw"$\ell$", fontsize=11)
ax_wl.set_ylabel(raw"$W_L^{80\,\mathrm{steps}} - W_L^{20\,\mathrm{steps}}$", fontsize=11)
ax_wl.set_title("Transfer function gain from extra steps", fontsize=12)
ax_wl.set_xlim(4000, 12000)
ax_wl.legend(fontsize=9)

savefig("results/map_convergence_diagnostic.png"; dpi=150)
println("\n\nSaved: results/map_convergence_diagnostic.png")

println("\nRMS ΔW_L at L=4000-12000 per sim: ",
        [@sprintf("%.4f", x) for x in ΔW_rms])
println("If all ΔW_rms < 0.01: 20 steps is sufficient.")
println("If any ΔW_rms > 0.05: increase MAPJ_STEPS in run_qe_gi_wl12k.jl and re-run.")
