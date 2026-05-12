#!/usr/bin/env julia
# gi_direction_scan.jl
#
# Diagnostic:
#   Test whether the MAP posterior prefers to move towards or away from the
#   GI high-L reconstruction.
#
# Candidate field:
#   phi_eps = phi_MAP + eps * HighL(phi_GI - phi_MAP)
#
# If log p decreases for eps > 0, the MAP objective rejects the GI-like high-L
# direction. If log p increases for eps > 0 but MAP optimisation does not move
# there, the issue is likely optimiser/preconditioner related.

using CMBLensing
using Statistics
using Printf
using DelimitedFiles
using JLD2

# ----------------------------
# User settings
# ----------------------------

const EPS_LIST = [-0.5, 0.0, 0.25, 0.5, 1.0]
const LCUT = 4000.0
const LMAX = 12000.0
const Δℓ = 2000

# Set true if you want to evaluate the MAP posterior objective at each proposal.
# This requires argmaxf_logpdf and logpdf to work with your CMBLensing.jl setup.
const DO_PROFILE_LOGP = true

# CG settings used when profiling over f at fixed phi.
const PROFILE_CG_KWARGS = (tol = 1e-3, nsteps = 300)

# Output
const OUTDIR = "results/diagnose"
const OUTFILE = joinpath(OUTDIR, "gi_direction_scan.jld2")
const OUTCSV  = joinpath(OUTDIR, "gi_direction_scan_summary.csv")


# ----------------------------
# Helper functions
# ----------------------------

function highL_mask(ff; Lcut=LCUT, Lmax=LMAX)
    proj = ff.proj
    arr = ff.arr
    NyF, NxF = size(arr)

    ℓx2D = repeat(proj.ℓx[1:NxF]', NyF, 1)
    ℓy2D = repeat(proj.ℓy[1:NyF],  1,   NxF)
    L2 = @. ℓx2D^2 + ℓy2D^2

    return @. (L2 >= Lcut^2) & (L2 <= Lmax^2)
end

function highL_part(ff; Lcut=LCUT, Lmax=LMAX)
    mask = highL_mask(ff; Lcut=Lcut, Lmax=Lmax)
    return FlatFourier(ifelse.(mask, ff.arr, zero(eltype(ff.arr))), ff.proj)
end

function propose_towards_gi(phi_map, phi_gi; eps, Lcut=LCUT, Lmax=LMAX)
    @assert size(phi_map.arr) == size(phi_gi.arr)
    @assert phi_map.proj == phi_gi.proj

    mask = highL_mask(phi_map; Lcut=Lcut, Lmax=Lmax)
    delta_arr = phi_gi.arr .- phi_map.arr
    prop_arr = copy(phi_map.arr)
    prop_arr[mask] .= phi_map.arr[mask] .+ eps .* delta_arr[mask]

    return FlatFourier(prop_arr, phi_map.proj)
end

function safe_get_Cℓ(a, b; Δℓ=Δℓ)
    spec = get_Cℓ(a, b; Δℓ=Δℓ)
    return Float64.(spec.ℓ), Float64.(spec.Cℓ)
end

function safe_get_auto(a; Δℓ=Δℓ)
    spec = get_Cℓ(a; Δℓ=Δℓ)
    return Float64.(spec.ℓ), Float64.(spec.Cℓ)
end

function response_and_rho(phi_true, phi_hat; Δℓ=Δℓ)
    ℓ, C_true_hat = safe_get_Cℓ(phi_true, phi_hat; Δℓ=Δℓ)
    _, C_true     = safe_get_auto(phi_true; Δℓ=Δℓ)
    _, C_hat      = safe_get_auto(phi_hat; Δℓ=Δℓ)

    W = C_true_hat ./ max.(C_true, 1e-30)
    ρ = C_true_hat ./ sqrt.(max.(C_true .* C_hat, 1e-30))

    return ℓ, W, ρ, C_true_hat, C_true, C_hat
end

function band_average(ℓ, y; Llo=LCUT, Lhi=LMAX)
    m = (ℓ .>= Llo) .& (ℓ .<= Lhi) .& isfinite.(y)
    return any(m) ? mean(y[m]) : NaN
end

function profile_logposterior(ds, phi)
    # Profile over the unlensed CMB f at fixed phi, then evaluate log posterior.
    # This is the relevant scalar objective for asking whether a proposed phi
    # direction is preferred by the MAP posterior.
    try
        fhat = argmaxf_logpdf(
            ds,
            (; ϕ = phi);
            conjgrad_kwargs = PROFILE_CG_KWARGS,
            preconditioner = :diag
        )
        lp = logpdf(ds, (; f = fhat, ϕ = phi))
        return Float64(lp)
    catch err
        @warn "Could not evaluate profiled log posterior. Returning NaN." exception=(err, catch_backtrace())
        return NaN
    end
end


# ----------------------------
# Main scan for one simulation
# ----------------------------

function run_gi_direction_scan_one(ds, phi_true, phi_map, phi_gi;
                                   eps_list=EPS_LIST,
                                   Lcut=LCUT,
                                   Lmax=LMAX,
                                   Δℓ=Δℓ,
                                   do_logp=DO_PROFILE_LOGP)

    results = Dict{Float64, Any}()

    println("\nGI-direction scan")
    println("  proposal: phi_eps = phi_MAP + eps * HighL(phi_GI - phi_MAP)")
    println("  Lcut = $Lcut, Lmax = $Lmax")
    println("  eps values = $(eps_list)")
    println()

    for eps in eps_list
        phi_eps = propose_towards_gi(phi_map, phi_gi; eps=eps, Lcut=Lcut, Lmax=Lmax)

        ℓ, W, ρ, C_cross, C_true, C_auto = response_and_rho(phi_true, phi_eps; Δℓ=Δℓ)

        lp = do_logp ? profile_logposterior(ds, phi_eps) : NaN

        W_high = band_average(ℓ, W; Llo=Lcut, Lhi=Lmax)
        ρ_high = band_average(ℓ, ρ; Llo=Lcut, Lhi=Lmax)

        results[eps] = (
            phi = phi_eps,
            ℓ = ℓ,
            W = W,
            ρ = ρ,
            C_cross = C_cross,
            C_true = C_true,
            C_auto = C_auto,
            logp_profile = lp,
            W_high = W_high,
            ρ_high = ρ_high
        )

        @printf("eps=%6.2f   logp=%14.4e   <W>=%8.4f   <rho>=%8.4f\n",
                eps, lp, W_high, ρ_high)
    end

    # Differences relative to eps = 0
    if haskey(results, 0.0)
        lp0 = results[0.0].logp_profile
        W0  = results[0.0].W_high
        ρ0  = results[0.0].ρ_high

        println("\nRelative to eps = 0 MAP field:")
        for eps in eps_list
            r = results[eps]
            @printf("eps=%6.2f   Δlogp=%14.4e   Δ<W>=%9.4f   Δ<rho>=%9.4f\n",
                    eps, r.logp_profile - lp0, r.W_high - W0, r.ρ_high - ρ0)
        end
    end

    return results
end


# ----------------------------
# Save utilities
# ----------------------------

function save_scan_results(results; outfile=OUTFILE, outcsv=OUTCSV)
    mkpath(dirname(outfile))

    eps_sorted = sort(collect(keys(results)))

    @save outfile results eps_sorted LCUT LMAX Δℓ EPS_LIST

    rows = Any[
        ["eps", "logp_profile", "W_high", "rho_high"]
    ]

    for eps in eps_sorted
        r = results[eps]
        push!(rows, [eps, r.logp_profile, r.W_high, r.ρ_high])
    end

    writedlm(outcsv, rows, ",")

    println("\nSaved:")
    println("  $outfile")
    println("  $outcsv")
end


# ----------------------------
# Example usage
# ----------------------------
#
# In your existing MAP/GI pipeline, after you have loaded or computed:
#
#   ds
#   phi_true
#   phi_map
#   phi_gi
#
# run:
#
#   include("gi_direction_scan.jl")
#   results = run_gi_direction_scan_one(ds, phi_true, phi_map, phi_gi)
#   save_scan_results(results)
#
# Interpretation:
#
#   eps = 0       : original MAP field
#   eps = 1       : replace MAP high-L modes by GI high-L modes
#   eps > 0       : move MAP towards GI at high L
#   eps < 0       : move MAP away from GI at high L
#
# If Δlogp(eps>0) < 0, the posterior rejects the GI high-L direction.
# If Δlogp(eps>0) > 0 but MAP optimisation did not find this direction,
# the issue is likely optimisation or preconditioning.
# If W increases but rho decreases, the GI direction is adding response
# amplitude without improving truth correlation.