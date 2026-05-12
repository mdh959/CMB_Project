#!/usr/bin/env julia
# diag_map_prior_test.jl
#
# Minimal test:
# Does MAP high-L W_L increase if we weaken the φ prior at high L?
#
# Runs same sims with:
#   1. fiducial prior
#   2. high-L prior × 3
#   3. high-L prior × 10
#   4. high-L prior × 30
#
# Diagnostic:
#   if W_L rises at L>6000 when prior is weakened,
#   then MAP was prior-suppressing high-L modes.

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

const Cℓ_fid  = camb(r=0.05, ℓmax=35000)
const θpix    = 0.7438046267475303
const Nside   = 512
const pol     = :I
const seed0   = 1000

const Lmax    = 12000
const Δℓ      = 500
const L_PROBE = [4000, 6000, 8000, 10000]

const nsims   = 3
const nsteps  = 30

const μKarcminT = 1.0
const beamFWHM  = 1.0
const αmax      = 0.3

const PRIOR_TESTS = [
    ("fid",  1.0),
    ("x3",   3.0),
    ("x10", 10.0),
    ("x30", 30.0),
]

mkpath("results/diagnose")

# ── MAP preconditioner patch ──────────────────────────────────────────────────

@eval CMBLensing begin
    function Hessian_logpdf_preconditioner(::Val{:f}, ds::DataSet)
        @unpack Cf̃, B̂, M̂, Cn̂ = ds
        v = copy(diag(pinv(Cf̃) + B̂' * M̂' * pinv(Cn̂) * M̂ * B̂))
        arr = v.arr
        good = isfinite.(arr) .& (arr .> 0)
        fill!(view(arr, .!good), any(good) ? mean(arr[good]) : 1.0)
        arr .= clamp.(arr, 1e-30, 1e30)
        return Diagonal(v)
    end
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function weaken_phi_prior!(ds, factor; Lcut=4000)
    # Increases Cϕ at high L → weakens prior penalty there.
    # This is diagnostic only, not a final physical estimator.

    Cϕdiag = diag(ds.Cϕ)
    Cϕmap  = copy(Cϕdiag)
    arr    = Cϕmap.arr
    L      = Cϕmap.proj.ℓmag

    mask = L .>= Lcut
    arr[mask] .*= factor

    ds.Cϕ = Diagonal(Cϕmap)
    return ds
end

function safe_start(ϕ_field)
    m = Map(ϕ_field)
    α = min(get_max_lensing_step(zero(m), m) * 0.9, 1.0)
    return α * ϕ_field
end

function probe_WL_rho(ϕ_true, ϕ_est)
    cl_tt = get_Cℓ(ϕ_true;        Δℓ=Δℓ)
    cl_tx = get_Cℓ(ϕ_true, ϕ_est; Δℓ=Δℓ)
    cl_xx = get_Cℓ(ϕ_est;         Δℓ=Δℓ)

    ℓ  = Float64.(collect(cl_tt.ℓ))
    TT = Float64.(cl_tt.Cℓ)
    TX = Float64.(cl_tx.Cℓ)
    XX = Float64.(cl_xx.Cℓ)

    WL = TX ./ max.(TT, 1e-100)
    ρL = TX ./ max.(sqrt.(max.(TT .* XX, 0.0)), 1e-100)

    at(v) = [v[argmin(abs.(ℓ .- Float64(L)))] for L in L_PROBE]
    return at(WL), at(ρL)
end

function run_one(seed, prior_factor)
    Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)

    load_kwargs = (
        Cℓ=Cℓ_fid,
        Cℓn=Cℓn,
        θpix=θpix,
        T=Float64,
        Nside=Nside,
        beamFWHM=beamFWHM,
        pol=pol,
        bandpass_mask=LowPass(Lmax),
        pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
    )

    (; ϕ, ds) = load_sim(; seed=seed, load_kwargs...)

    prior_factor != 1.0 && weaken_phi_prior!(ds, prior_factor; Lcut=4000)

    ϕqe = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=true).ϕqe
    ϕstart = safe_start(ϕqe)

    WL0, ρ0 = probe_WL_rho(Map(ϕ), Map(ϕstart))

    result = MAP_joint(ds, FieldTuple(ϕ=ϕstart);
        nsteps=nsteps,
        αmax=αmax,
        conjgrad_kwargs=(tol=1e-3, nsteps=500),
        progress=false,
        history_keys=(:total_logpdf,),
    )

    WLf, ρf = probe_WL_rho(Map(ϕ), Map(result.ϕ))
    lp = Float64.([h.total_logpdf for h in result.history])

    return (; WL0, ρ0, WLf, ρf, lp)
end

# ── Main run ──────────────────────────────────────────────────────────────────

results = Dict{String, Any}()

for (tag, fac) in PRIOR_TESTS
    println("\n=== Prior test: $tag, factor=$fac ===")

    sims = []
    for s in 1:nsims
        seed = seed0 + s
        print("  sim $s/$nsims seed=$seed ... ")
        flush(stdout)

        r = try
            run_one(seed, fac)
        catch err
            @warn "failed" tag seed err
            nothing
        end

        push!(sims, r)
        println(r === nothing ? "failed" : "done")
    end

    sims = filter(!isnothing, sims)
    results[tag] = sims

    if !isempty(sims)
        WL_mean = mean(reduce(hcat, [r.WLf for r in sims]); dims=2)[:]
        ρ_mean  = mean(reduce(hcat, [r.ρf  for r in sims]); dims=2)[:]

        println("  final mean:")
        for (i, L) in enumerate(L_PROBE)
            @printf "    L=%5d   W_L=%.3f   ρ_L=%.3f\n" L WL_mean[i] ρ_mean[i]
        end
    end
end

jldsave("results/diagnose/diag_map_prior_test.jld2"; results, L_PROBE, PRIOR_TESTS)
println("\nSaved results/diagnose/diag_map_prior_test.jld2")

# ── Plot ──────────────────────────────────────────────────────────────────────

PythonPlot.rc("font", family="serif", size=10)

fig, axs = subplots(1, 3; figsize=(14, 4.2), constrained_layout=true)

colors = Dict("fid"=>"C0", "x3"=>"C1", "x10"=>"C2", "x30"=>"C3")

for (tag, fac) in PRIOR_TESTS
    sims = get(results, tag, [])
    isempty(sims) && continue

    WL0 = mean(reduce(hcat, [r.WL0 for r in sims]); dims=2)[:]
    WLf = mean(reduce(hcat, [r.WLf for r in sims]); dims=2)[:]
    ρf  = mean(reduce(hcat, [r.ρf  for r in sims]); dims=2)[:]

    lp_curves = [r.lp .- r.lp[1] for r in sims]
    nmin = minimum(length.(lp_curves))
    lp_mean = mean(reduce(hcat, [c[1:nmin] for c in lp_curves]); dims=2)[:]

    axs[0].plot(1:nmin, lp_mean; label=tag, color=colors[tag], lw=2)
    axs[1].plot(L_PROBE, WLf; "o-", label=tag, color=colors[tag], lw=2)
    axs[1].plot(L_PROBE, WL0; ":", color=colors[tag], alpha=0.5)
    axs[2].plot(L_PROBE, ρf; "o-", label=tag, color=colors[tag], lw=2)
end

axs[0].set_title("MAP logpdf gain")
axs[0].set_xlabel("step")
axs[0].set_ylabel(L"\Delta \log p")
axs[0].legend(frameon=false)

axs[1].axhline(1.0; color="k", ls=":", lw=0.8)
axs[1].set_title(L"final $W_L$")
axs[1].set_xlabel(L"L")
axs[1].set_ylabel(L"W_L")
axs[1].legend(frameon=false)

axs[2].axhline(1.0; color="k", ls=":", lw=0.8)
axs[2].set_title(L"final $\rho_L$")
axs[2].set_xlabel(L"L")
axs[2].set_ylabel(L"\rho_L")
axs[2].set_ylim(-0.05, 1.05)
axs[2].legend(frameon=false)

fig.suptitle("MAP prior weakening test: fiducial vs boosted high-L Cϕ")
fig.savefig("results/diagnose/diag_map_prior_test.png"; dpi=160)
PythonPlot.plotclose("all")

println("Saved results/diagnose/diag_map_prior_test.png")
println("\nDone.")