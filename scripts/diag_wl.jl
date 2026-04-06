#!/usr/bin/env julia
# Diagnostics: check stored W_L and run a single fresh UL sim.
# Run from project root: julia scripts/diag_wl.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using JLD2, Statistics, Printf

# ── Part 1: Stored W_L file ───────────────────────────────────────────────────
println("="^70)
println("PART 1: Stored W_L file")
println("="^70)

wl_file = "results/WL_qe_gi_12000_ul.jld2"
if isfile(wl_file)
    f  = JLD2.load(wl_file)
    el = Float64.(f["ell_template"] isa Nothing ? f["l_template"] : get(f, "ell_template", get(f, "ℓ_template", Float64[])))
    # Try different possible key names
    for k in ["ℓ_template", "l_template", "ell_template"]
        if haskey(f, k); el = Float64.(f[k]); break; end
    end
    ns = get(f, "nsims_completed", 0)
    println("nsims_completed = $ns")
    println("ell_template: $(length(el)) bins, spacing=$(round(el[2]-el[1];sigdigits=4)), range=$(el[1]) to $(el[end])")

    # Check R_L vector consistency
    for key in ["R_qe_wf_sims", "R_gi_o_sims", "R_gi_b_sims"]
        haskey(f, key) || continue
        R    = f[key]
        lens = length.(R)
        bad  = sum(lens .!= lens[1])
        @printf "  %-18s %d sims, vector length %d %s\n" key length(R) lens[1] (bad>0 ? "WARNING: $bad INCONSISTENT" : "OK")
    end

    # Print W_L values
    Wqe  = haskey(f,"W_qe_wf") ? Float64.(f["W_qe_wf"]) : fill(NaN,length(el))
    Wgio = haskey(f,"W_gi_o")  ? Float64.(f["W_gi_o"])  : fill(NaN,length(el))
    Wgib = haskey(f,"W_gi_b")  ? Float64.(f["W_gi_b"])  : fill(NaN,length(el))
    println("\n  ell        W_QE_WF    W_GI_O     W_GI_B")
    msk  = @. isfinite(el) & (el >= 4500) & (el <= 12000)
    idxs = findall(msk)
    step = max(1, length(idxs) ÷ 12)
    for i in idxs[1:step:end]
        @printf "  %-7.0f    %8.4f   %8.4f   %8.4f\n" el[i] Wqe[i] Wgio[i] Wgib[i]
    end

    # Oscillation check: count sign reversals
    for (name, W) in [("W_qe_wf",Wqe), ("W_gi_o",Wgio), ("W_gi_b",Wgib)]
        m2 = @. isfinite(W) & (el >= 5000) & (el <= 11000)
        Ws = W[m2]
        isempty(Ws) && continue
        sc = sum(diff(sign.(Ws .- mean(Ws))) .!= 0)
        nb = length(Ws)
        @printf "  %-12s sign changes: %d / %d bins (%d%%)\n" name sc nb round(Int,100*sc/max(nb-1,1))
    end
else
    println("File not found: $wl_file")
end

# ── Part 2: Single fresh UL sim ───────────────────────────────────────────────
println()
println("="^70)
println("PART 2: Single UL sim (seed=1001) -- per-sim R_L")
println("="^70)

using CMBLensing
include("../utils.jl"); using .Utils

Cℓ_cmb = camb(r=0.05, ℓmax=35000)
Cℓn    = noiseCℓs(μKarcminT=0.1, ℓknee=0, ℓmax=20000)

println("Loading sim (θpix=0.5, Nside=512, beam=0.3', Lmax=20000)...")
(; ϕ, ds) = load_sim(; seed=1001, Cℓ=Cℓ_cmb, Cℓn=Cℓn,
                       θpix=0.5, T=Float64, Nside=512, beamFWHM=0.3, pol=:I,
                       bandpass_mask=LowPass(20000),
                       pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))

println("Running estimators...")
ϕqe_wf = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true).ϕqe
ϕgi_b  = gi_estimate_boryana(ds)
ϕgi_o  = gi_estimate(ds; Lmax=20000)

Δℓ = 150
cl_tt   = get_Cℓ(ϕ;          Δℓ=Δℓ)
cl_tqwf = get_Cℓ(ϕ, ϕqe_wf; Δℓ=Δℓ)
cl_tgib = get_Cℓ(ϕ, ϕgi_b;  Δℓ=Δℓ)
cl_tgio = get_Cℓ(ϕ, ϕgi_o;  Δℓ=Δℓ)

ell = Float64.(collect(cl_tt.ℓ))
println("\n  ell      R_QE_WF   R_GI_B   R_GI_O   (should all be smooth, ~0.5-1)")
msk = @. (ell >= 4500) & (ell <= 12000)
for i in findall(msk)[1:2:end]
    d = cl_tt.Cℓ[i]
    rq  = d > 0 ? cl_tqwf.Cℓ[i] / d : NaN
    rb  = d > 0 ? cl_tgib.Cℓ[i] / d : NaN
    ro  = d > 0 ? cl_tgio.Cℓ[i] / d : NaN
    @printf "  %-6.0f   %8.4f  %8.4f  %8.4f\n" ell[i] rq rb ro
end
println()
println("Expected: R_QE ~ 1 (WF in UL limit), R_GI_B and R_GI_O ~ B(L) ~ 0.5-0.9")
println("If oscillating between negative and positive: FlatFourier/normalisation bug.")
println("If consistently < 0.1: denominator SUM/MEAN mismatch (wrong by factor N_pix).")
