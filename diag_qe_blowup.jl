#!/usr/bin/env julia
#
# diag_qe_blowup.jl
#
# Diagnose the QE stripe/ring outlier mechanism in UL sims.
#
# Hypothesis:
#   Near the bandpass taper edge, the QE inverse-variance denominator
#
#       ΣTtot(L) = B̂(L)^2 C_f(L) + C_n(L)
#
#   can become very small. Then
#
#       u_L = ΣTtot(L)^(-1) [B̂ d]_L
#
#   becomes huge for one/few Fourier modes, producing a stripe/ring artefact in
#   the QE reconstruction. Since κ_L ∝ L^2 φ_L, high-L φ artefacts are further
#   amplified in κ.
#
# Outputs:
#   results/diag_qe_blowup/summary.txt
#   results/diag_qe_blowup/sigma_tot_max_u.png
#   results/diag_qe_blowup/sigma_tot_zoom.png
#   results/diag_qe_blowup/u_map_simXXX.png
#   results/diag_qe_blowup/qe_phi_fourier_max.png
#
# Run:
#   julia diag_qe_blowup.jl

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
import CMBLensing: m_rfft
using Statistics: mean, std, median, quantile
using Printf
using PythonPlot

# ─────────────────────────────────────────────────────────────────────────────
# Parameters matching the UL QE runs
# ─────────────────────────────────────────────────────────────────────────────

const μKarcminT = 0.1
const beamFWHM  = 0.3
const Lmax      = 12000
const θpix      = 0.7438046267475303
const Nside     = 512
const seed0     = 1000
const Δℓ        = 30

const Cℓ_fid = camb(r=0.05, ℓmax=35000)

const load_kw = (
    Cℓ   = Cℓ_fid,
    Cℓn  = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax),
    θpix = θpix,
    T = Float64,
    Nside = Nside,
    beamFWHM = beamFWHM,
    pol = :I,
    bandpass_mask = LowPass(Lmax),
    pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

# From your UL scan:
# below-threshold exclude set:
#   Set([121, 183, 242, 327, 442, 611, 661, 705, 894, 951, 967])
const OUTLIER_IDXS = [121, 967, 661, 242, 705, 611, 327, 183, 894, 442, 951]

# Normal comparison sims. Avoid overlap with outliers.
const NORMAL_IDXS = [i for i in 1:30 if !(i in OUTLIER_IDXS)][1:20]

const OUTDIR = "results/diag_qe_blowup"
mkpath(OUTDIR)

println("QE blow-up diagnostic — UL params ($(μKarcminT) µK·arcmin, $(beamFWHM)' beam, Lmax=$(Lmax))")
println("Outlier sim indices: $OUTLIER_IDXS")
println("Outlier seeds:       $(seed0 .+ OUTLIER_IDXS)")
println("Normal sim indices:  $NORMAL_IDXS")
println("Normal seeds:        $(seed0 .+ NORMAL_IDXS)")
println("=" ^ 90)

# ─────────────────────────────────────────────────────────────────────────────
# Helper structs
# ─────────────────────────────────────────────────────────────────────────────

struct SimDiag
    sim_idx       :: Int
    seed          :: Int

    max_u         :: Float64
    max_v         :: Float64
    max_u_map     :: Float64

    worst_j       :: Int
    worst_i       :: Int
    worst_L       :: Float64
    worst_lx      :: Float64
    worst_ly      :: Float64
    worst_ΣT      :: Float64
    worst_TF      :: Float64
    worst_Cf̃     :: Float64
    worst_Cn      :: Float64
    worst_d_F     :: Float64
    worst_Bd      :: Float64
    whitened      :: Float64

    min_ΣT        :: Float64
    med_ΣT        :: Float64

    # Final QE φ diagnostics
    max_phi       :: Float64
    phi_j         :: Int
    phi_i         :: Int
    phi_L         :: Float64
    phi_lx        :: Float64
    phi_ly        :: Float64

    # Distance between worst u mode and worst QE φ mode
    ΔL_u_phi      :: Float64
end

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

function get_operator_diags(ds)
    m    = Map(ds.d)
    proj = m.proj
    Ny, Nx = size(m.arr)
    NyF = Ny ÷ 2 + 1

    f_ones  = FlatFourier(ones(ComplexF64, NyF, Nx), proj)

    # Transfer function = (M̂*B̂), matching the QE denominator (quadratic_estimate.jl:97)
    TF_diag  = real.((ds.M̂ * ds.B̂ * f_ones).arr)[1:NyF, 1:Nx]
    Cf̃_diag = real.((ds.Cf̃ * f_ones).arr)[1:NyF, 1:Nx]   # lensed Cf̃, as used in QE
    Cn_diag  = real.((ds.Cn̂ * f_ones).arr)[1:NyF, 1:Nx]

    ΣT_diag = @. TF_diag^2 * Cf̃_diag + Cn_diag

    ℓx_arr = proj.ℓx[1:Nx]
    ℓy_arr = proj.ℓy[1:NyF]

    lx2D = repeat(ℓx_arr', NyF, 1)
    ly2D = repeat(ℓy_arr,  1,   Nx)
    L2D  = @. sqrt(lx2D^2 + ly2D^2)

    return (; m, proj, Ny, Nx, NyF, TF_diag, Cf̃_diag, Cn_diag, ΣT_diag, lx2D, ly2D, L2D)
end

function safe_argmax_abs(A)
    vals = abs.(A)
    idx = argmax(vals)
    return Tuple(idx), vals[idx]
end

function analyse_sim(sim_idx::Int)
    seed = seed0 + sim_idx
    r    = load_sim(; seed=seed, load_kw...)
    ds   = r.ds

    gd = get_operator_diags(ds)
    (; m, proj, TF_diag, Cf̃_diag, Cn_diag, ΣT_diag, lx2D, ly2D, L2D) = gd

    # Data in Fourier space
    d_F  = m_rfft(m.arr, (1, 2))
    Bd_F = @. TF_diag * d_F

    # Inverse-variance leg
    # Deliberately no floor: this diagnostic wants to expose blow-ups.
    u_F = @. ifelse(ΣT_diag > 0, Bd_F / ΣT_diag, complex(0.0))
    v_F = @. Cf̃_diag * u_F

    (j, i), max_u = safe_argmax_abs(u_F)

    u_field = FlatFourier(u_F, proj)
    u_map   = Map(u_field).arr

    ΣT_val = ΣT_diag[j, i]
    TF_val = TF_diag[j, i]
    Cf̃_val = Cf̃_diag[j, i]
    Cn_val = Cn_diag[j, i]
    d_val  = abs(d_F[j, i])
    Bd_val = abs(Bd_F[j, i])
    whitened = ΣT_val > 0 ? Bd_val / sqrt(ΣT_val) : Inf

    pos_ΣT = ΣT_diag[ΣT_diag .> 0]
    low_L_mask = L2D .< (Lmax / 2)
    low_L_ΣT = ΣT_diag[low_L_mask .& (ΣT_diag .> 0)]
    med_ΣT = isempty(low_L_ΣT) ? NaN : median(low_L_ΣT)

    # Direct final QE φ diagnostic — wiener_filtered=false to see raw blow-up
    ϕqe = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=false).ϕqe
    ϕF = ϕqe.arr

    (jϕ, iϕ), max_phi = safe_argmax_abs(ϕF)

    ΔL_u_phi = sqrt((lx2D[j, i] - lx2D[jϕ, iϕ])^2 + (ly2D[j, i] - ly2D[jϕ, iϕ])^2)

    return SimDiag(
        sim_idx,
        seed,

        max_u,
        maximum(abs.(v_F)),
        maximum(abs.(u_map)),

        j,
        i,
        L2D[j, i],
        lx2D[j, i],
        ly2D[j, i],
        ΣT_val,
        TF_val,
        Cf̃_val,
        Cn_val,
        d_val,
        Bd_val,
        whitened,

        isempty(pos_ΣT) ? NaN : minimum(pos_ΣT),
        med_ΣT,

        max_phi,
        jϕ,
        iϕ,
        L2D[jϕ, iϕ],
        lx2D[jϕ, iϕ],
        ly2D[jϕ, iϕ],

        ΔL_u_phi
    )
end

function get_sigma_tot_vs_L(sim_idx::Int)
    seed = seed0 + sim_idx
    r    = load_sim(; seed=seed, load_kw...)
    ds   = r.ds
    gd   = get_operator_diags(ds)
    return gd.L2D[:], gd.ΣT_diag[:]
end

function get_u_map_and_fourier(sim_idx::Int)
    seed = seed0 + sim_idx
    r    = load_sim(; seed=seed, load_kw...)
    ds   = r.ds
    gd   = get_operator_diags(ds)
    (; m, proj, TF_diag, ΣT_diag, L2D) = gd

    d_F  = m_rfft(m.arr, (1, 2))
    Bd_F = @. TF_diag * d_F
    u_F  = @. ifelse(ΣT_diag > 0, Bd_F / ΣT_diag, complex(0.0))
    u_map = Map(FlatFourier(u_F, proj)).arr

    return (; u_F, u_map, L2D, proj)
end

function group_stats(diags::Vector{SimDiag})
    isempty(diags) && return nothing

    return (
        n = length(diags),
        max_u = mean(d.max_u for d in diags),
        max_phi = mean(d.max_phi for d in diags),
        worst_L = mean(d.worst_L for d in diags),
        phi_L = mean(d.phi_L for d in diags),
        worst_ΣT = mean(d.worst_ΣT for d in diags),
        min_ΣT = mean(d.min_ΣT for d in diags),
        whitened = mean(d.whitened for d in diags),
        frac_near_u = mean(d.worst_L > 0.9 * Lmax for d in diags),
        frac_near_phi = mean(d.phi_L > 0.9 * Lmax for d in diags),
        ΔL_u_phi = mean(d.ΔL_u_phi for d in diags),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Run diagnostics
# ─────────────────────────────────────────────────────────────────────────────

println("\nAnalysing outlier sims...")
outlier_diags = SimDiag[]

for sidx in OUTLIER_IDXS
    @printf("  sim %5d (seed %5d)... ", sidx, seed0 + sidx)
    flush(stdout)

    try
        d = analyse_sim(sidx)
        push!(outlier_diags, d)
        @printf("max|u|=%9.3e  max|φQE|=%9.3e  L_u=%6.0f  L_φ=%6.0f  ΣT_u=%9.3e  white=%9.3e\n",
                d.max_u, d.max_phi, d.worst_L, d.phi_L, d.worst_ΣT, d.whitened)
    catch err
        msg = string(err)
        @printf("FAILED: %s\n", msg[1:min(120, length(msg))])
    end
end

println("\nAnalysing normal sims...")
normal_diags = SimDiag[]

for sidx in NORMAL_IDXS
    @printf("  sim %5d (seed %5d)... ", sidx, seed0 + sidx)
    flush(stdout)

    try
        d = analyse_sim(sidx)
        push!(normal_diags, d)
        @printf("max|u|=%9.3e  max|φQE|=%9.3e  L_u=%6.0f  L_φ=%6.0f  ΣT_u=%9.3e  white=%9.3e\n",
                d.max_u, d.max_phi, d.worst_L, d.phi_L, d.worst_ΣT, d.whitened)
    catch err
        msg = string(err)
        @printf("FAILED: %s\n", msg[1:min(120, length(msg))])
    end
end

go = group_stats(outlier_diags)
gn = group_stats(normal_diags)

# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 100)
println("OUTLIER DETAILED DIAGNOSTICS — worst inverse-filter mode argmax |u_F|")
println("=" ^ 100)

@printf("\n  %-6s %-6s %10s %10s %10s %5s %5s %8s %8s %8s %12s\n",
        "sim", "seed", "max|u|", "max|v|", "max|φ|", "j", "i", "L_u", "lx", "ly", "ΣT_u")
println("  " * "-"^100)

for d in outlier_diags
    @printf("  %-6d %-6d %10.3e %10.3e %10.3e %5d %5d %8.0f %8.0f %8.0f %12.3e\n",
            d.sim_idx, d.seed, d.max_u, d.max_v, d.max_phi,
            d.worst_j, d.worst_i, d.worst_L, d.worst_lx, d.worst_ly, d.worst_ΣT)
end

println("\n" * "=" ^ 100)
println("OUTLIER vs NORMAL GROUP COMPARISON")
println("=" ^ 100)

@printf("\n  %-14s %10s %10s %9s %9s %12s %12s %12s %10s %10s\n",
        "group(n)", "max|u|", "max|φ|", "L_u", "L_φ", "ΣT_u", "minΣT", "whitened", "u>0.9L", "φ>0.9L")
println("  " * "-"^110)

if go !== nothing
    @printf("  %-14s %10.3e %10.3e %9.0f %9.0f %12.3e %12.3e %12.3e %9.0f%% %9.0f%%\n",
            "outliers($(go.n))", go.max_u, go.max_phi, go.worst_L, go.phi_L,
            go.worst_ΣT, go.min_ΣT, go.whitened, 100 * go.frac_near_u, 100 * go.frac_near_phi)
end

if gn !== nothing
    @printf("  %-14s %10.3e %10.3e %9.0f %9.0f %12.3e %12.3e %12.3e %9.0f%% %9.0f%%\n",
            "normal($(gn.n))", gn.max_u, gn.max_phi, gn.worst_L, gn.phi_L,
            gn.worst_ΣT, gn.min_ΣT, gn.whitened, 100 * gn.frac_near_u, 100 * gn.frac_near_phi)
end

# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis logic
# ─────────────────────────────────────────────────────────────────────────────

diagnosis_lines = String[]

push!(diagnosis_lines, "QE blow-up diagnostic — UL params")
push!(diagnosis_lines, "μKarcminT = $μKarcminT")
push!(diagnosis_lines, "beamFWHM  = $beamFWHM")
push!(diagnosis_lines, "Lmax      = $Lmax")
push!(diagnosis_lines, "θpix      = $θpix")
push!(diagnosis_lines, "Nside     = $Nside")
push!(diagnosis_lines, "")

if go !== nothing && gn !== nothing
    ratio_u      = go.max_u / gn.max_u
    ratio_phi    = go.max_phi / gn.max_phi
    ratio_ΣT     = gn.worst_ΣT / max(go.worst_ΣT, 1e-300)
    frac_near_u  = go.frac_near_u
    frac_near_phi = go.frac_near_phi

    push!(diagnosis_lines, @sprintf("Outlier mean max|u_F| is %.2fx larger than normal.", ratio_u))
    push!(diagnosis_lines, @sprintf("Outlier mean max|φ_QE,F| is %.2fx larger than normal.", ratio_phi))
    push!(diagnosis_lines, @sprintf("Outlier mean worst ΣTtot is %.2fx smaller than normal.", ratio_ΣT))
    push!(diagnosis_lines, @sprintf("Outlier worst-u mode has L > 0.9 Lmax in %.1f%% of cases.", 100 * frac_near_u))
    push!(diagnosis_lines, @sprintf("Outlier worst-φ mode has L > 0.9 Lmax in %.1f%% of cases.", 100 * frac_near_phi))
    push!(diagnosis_lines, @sprintf("Mean Fourier-space distance between worst u and worst φ mode is ΔL = %.1f.", go.ΔL_u_phi))
    push!(diagnosis_lines, "")

    if ratio_u > 5 && ratio_phi > 5 && frac_near_u > 0.5
        push!(diagnosis_lines, "CONFIRMED: QE blow-up is consistent with an inverse-filter instability near the taper edge.")
        push!(diagnosis_lines, "Interpretation: ΣTtot becomes very small near Lmax, making u = ΣTtot^{-1}(B d) huge.")
        push!(diagnosis_lines, "This contaminates the QE reconstruction and appears as a stripe/ring artefact in map space.")
    elseif ratio_u > 5 && ratio_phi > 5
        push!(diagnosis_lines, "LIKELY: QE blow-up is driven by inverse-filter amplification, but not always exactly at Lmax.")
        push!(diagnosis_lines, "Check for zero/small ΣTtot modes away from the taper edge.")
    elseif ratio_u > 5 && ratio_phi <= 5
        push!(diagnosis_lines, "PARTIAL: the inverse-filter leg u blows up, but the final QE φ maximum is not proportionally larger.")
        push!(diagnosis_lines, "The final QE estimator may be suppressing or redistributing the bad mode.")
    else
        push!(diagnosis_lines, "NOT CONFIRMED: outliers do not show a large enough inverse-filter blow-up relative to normal sims.")
        push!(diagnosis_lines, "Check for rare high-amplitude CMB realisations, φ-to-κ conversion amplification, or later QE normalisation issues.")
    end
else
    push!(diagnosis_lines, "Insufficient successful diagnostics to compare outlier and normal groups.")
end

println("\n" * "=" ^ 100)
println("DIAGNOSIS")
println("=" ^ 100)
for line in diagnosis_lines
    println("  " * line)
end

println("\n" * "=" ^ 100)
println("RECOMMENDED FIXES")
println("=" ^ 100)
println("""
  1. Add a ΣTtot floor in the QE inverse filter:
       ε_floor = 1e-4 * median(ΣTtot[L < Lmax/2])
       ΣTtot_safe = max.(ΣTtot, ε_floor)

  2. Lower the QE reconstruction bandpass:
       qe_bandpass = LowPass(10000; Δℓ=2000)

  3. Use a wider taper:
       bandpass_mask = LowPass(Lmax; Δℓ=4000)

  4. As a quick diagnostic only, post-filter QE:
       ϕ_qe_safe = LowPass(10000; Δℓ=1000) * ϕ_qe

  5. Re-run the worst sims after each fix and verify:
       max|u_F| drops,
       max|φ_QE,F| drops,
       stripe/ring disappears,
       QE–truth correlation recovers.
""")

# ─────────────────────────────────────────────────────────────────────────────
# Write text summary
# ─────────────────────────────────────────────────────────────────────────────

txtfile = joinpath(OUTDIR, "summary.txt")

open(txtfile, "w") do io
    println(io, "QE blow-up diagnostic — UL params")
    println(io, "μKarcminT = $μKarcminT")
    println(io, "beamFWHM  = $beamFWHM")
    println(io, "Lmax      = $Lmax")
    println(io, "θpix      = $θpix")
    println(io, "Nside     = $Nside")
    println(io, "seed0     = $seed0")
    println(io, "")
    println(io, "Outlier sim indices: $OUTLIER_IDXS")
    println(io, "Outlier seeds:       $(seed0 .+ OUTLIER_IDXS)")
    println(io, "Normal sim indices:  $NORMAL_IDXS")
    println(io, "Normal seeds:        $(seed0 .+ NORMAL_IDXS)")
    println(io, "="^120)

    println(io, "\nOUTLIER DETAILED DIAGNOSTICS")
    @printf(io, "%-8s %-8s %-12s %-12s %-12s %-8s %-8s %-10s %-10s %-10s %-12s %-12s %-12s %-12s %-12s\n",
            "sim", "seed", "max|u|", "max|v|", "max|φ|", "u_j", "u_i",
            "L_u", "Lφ", "ΔL", "ΣT_u", "TF_u", "Cf̃_u", "Cn_u", "whitened")

    for d in outlier_diags
        @printf(io, "%-8d %-8d %-12.4e %-12.4e %-12.4e %-8d %-8d %-10.1f %-10.1f %-10.1f %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n",
                d.sim_idx, d.seed, d.max_u, d.max_v, d.max_phi,
                d.worst_j, d.worst_i, d.worst_L, d.phi_L, d.ΔL_u_phi,
                d.worst_ΣT, d.worst_TF, d.worst_Cf̃, d.worst_Cn, d.whitened)
    end

    println(io, "\nNORMAL DETAILED DIAGNOSTICS")
    @printf(io, "%-8s %-8s %-12s %-12s %-12s %-8s %-8s %-10s %-10s %-10s %-12s %-12s %-12s %-12s %-12s\n",
            "sim", "seed", "max|u|", "max|v|", "max|φ|", "u_j", "u_i",
            "L_u", "Lφ", "ΔL", "ΣT_u", "TF_u", "Cf̃_u", "Cn_u", "whitened")

    for d in normal_diags
        @printf(io, "%-8d %-8d %-12.4e %-12.4e %-12.4e %-8d %-8d %-10.1f %-10.1f %-10.1f %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e\n",
                d.sim_idx, d.seed, d.max_u, d.max_v, d.max_phi,
                d.worst_j, d.worst_i, d.worst_L, d.phi_L, d.ΔL_u_phi,
                d.worst_ΣT, d.worst_TF, d.worst_Cf̃, d.worst_Cn, d.whitened)
    end

    println(io, "\nGROUP COMPARISON")
    @printf(io, "%-14s %-12s %-12s %-10s %-10s %-12s %-12s %-12s %-12s %-12s %-12s\n",
            "group", "max|u|", "max|φ|", "L_u", "Lφ",
            "ΣT_u", "minΣT", "whitened", "u>0.9L", "φ>0.9L", "ΔL_uφ")

    if go !== nothing
        @printf(io, "%-14s %-12.4e %-12.4e %-10.1f %-10.1f %-12.4e %-12.4e %-12.4e %-12.4f %-12.4f %-12.1f\n",
                "outliers($(go.n))", go.max_u, go.max_phi, go.worst_L, go.phi_L,
                go.worst_ΣT, go.min_ΣT, go.whitened, go.frac_near_u, go.frac_near_phi, go.ΔL_u_phi)
    end

    if gn !== nothing
        @printf(io, "%-14s %-12.4e %-12.4e %-10.1f %-10.1f %-12.4e %-12.4e %-12.4e %-12.4f %-12.4f %-12.1f\n",
                "normal($(gn.n))", gn.max_u, gn.max_phi, gn.worst_L, gn.phi_L,
                gn.worst_ΣT, gn.min_ΣT, gn.whitened, gn.frac_near_u, gn.frac_near_phi, gn.ΔL_u_phi)
    end

    println(io, "\nDIAGNOSIS")
    for line in diagnosis_lines
        println(io, line)
    end

    println(io, "\nRECOMMENDED FIXES")
    println(io, "1. Add a ΣTtot floor: ε_floor = 1e-4 * median(ΣTtot[L < Lmax/2]).")
    println(io, "2. Lower QE Lmax to 10000 with a taper.")
    println(io, "3. Use a wider LowPass taper, e.g. LowPass(Lmax; Δℓ=4000).")
    println(io, "4. Post-filter QE as a diagnostic only: LowPass(10000; Δℓ=1000) * ϕqe.")
    println(io, "5. Re-run worst sims and check whether max|u_F|, max|φ_QE,F|, and map artefacts disappear.")
end

println("\nText summary saved to $txtfile")

# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

println("\nGenerating figures...")

# Fig 1: ΣTtot vs L and max|u| histogram
fig, axes = subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]

if !isempty(outlier_diags)
    Lv, ΣTv = get_sigma_tot_vs_L(outlier_diags[1].sim_idx)
    ax1.scatter(Lv, ΣTv, s=0.05, alpha=0.15, color="C1",
                label="outlier sim $(outlier_diags[1].sim_idx)")
end

if !isempty(normal_diags)
    Lv, ΣTv = get_sigma_tot_vs_L(normal_diags[1].sim_idx)
    ax1.scatter(Lv, ΣTv, s=0.05, alpha=0.15, color="C0",
                label="normal sim $(normal_diags[1].sim_idx)")
end

ax1.axvline(Lmax, color="k", ls="--", lw=0.8, label="Lmax")
ax1.set_yscale("log")
ax1.set_xlabel(L"$L$")
ax1.set_ylabel(L"$\Sigma_{T,\mathrm{tot}}(L)=\hat{B}^2 C_\ell^f+\hat{C}_n$")
ax1.set_title("QE denominator ΣTtot vs L")
ax1.legend(fontsize=8, markerscale=5)

ax2 = axes[1]

if !isempty(normal_diags)
    ax2.hist([d.max_u for d in normal_diags], bins=10, color="C0", alpha=0.6, label="normal")
end

if !isempty(outlier_diags)
    for d in outlier_diags
        ax2.axvline(d.max_u, color="C1", lw=1.2, alpha=0.75)
    end
    ax2.axvline(NaN, color="C1", lw=1.2, label="outliers")
end

ax2.set_xlabel(L"$\max |u_F|$")
ax2.set_title("Distribution of max|u|")
ax2.legend(fontsize=9)

tight_layout()
figpath = joinpath(OUTDIR, "sigma_tot_max_u.png")
savefig(figpath, dpi=200)
PythonPlot.close("all")
println("  Saved $figpath")

# Fig 2: ΣTtot near Lmax for outliers
fig, ax = subplots(figsize=(8, 4))

colors_out = ["C1", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "tab:brown", "tab:pink", "tab:gray"]

for (ki, d) in enumerate(outlier_diags)
    Lv, ΣTv = get_sigma_tot_vs_L(d.sim_idx)
    mask = Lv .> 0.7 * Lmax
    col = ki <= length(colors_out) ? colors_out[ki] : "gray"

    ax.scatter(Lv[mask], ΣTv[mask], s=0.1, alpha=0.18,
               color=col, label="sim $(d.sim_idx)")
end

ax.axvline(Lmax, color="k", ls="--", lw=0.8, label="Lmax")
ax.set_yscale("log")
ax.set_xlabel(L"$L$")
ax.set_ylabel(L"$\Sigma_{T,\mathrm{tot}}$")
ax.set_title("ΣTtot near Lmax for outlier sims")
ax.legend(fontsize=7, markerscale=5, ncol=2)

tight_layout()
figpath = joinpath(OUTDIR, "sigma_tot_zoom.png")
savefig(figpath, dpi=200)
PythonPlot.close("all")
println("  Saved $figpath")

# Fig 3: u map and u_F for worst outlier
if !isempty(outlier_diags)
    # Pick largest max_u outlier, not just first in list
    iworst = argmax([d.max_u for d in outlier_diags])
    d_worst = outlier_diags[iworst]

    ud = get_u_map_and_fourier(d_worst.sim_idx)
    (; u_F, u_map) = ud

    fig, axes = subplots(1, 2, figsize=(11, 4))
    ax1 = axes[0]
    ax2 = axes[1]

    vsc = 3 * std(u_map)
    im1 = ax1.imshow(u_map, cmap="RdBu_r", vmin=-vsc, vmax=vsc, origin="lower")
    ax1.set_title("u = ΣTtot⁻¹(B̂d) [map], sim $(d_worst.sim_idx)")
    colorbar(im1, ax=ax1)
    ax1.axis("off")

    abs_u_F = abs.(u_F)
    im2 = ax2.imshow(log10.(abs_u_F .+ 1e-30), cmap="inferno", origin="lower")
    ax2.scatter([d_worst.worst_i - 1], [d_worst.worst_j - 1],
                color="cyan", marker="x", s=80, linewidths=1.5,
                label="argmax |u_F|")
    ax2.set_title(L"$\log_{10}|u_F|$; " * "L_worst=$(round(Int, d_worst.worst_L))")
    colorbar(im2, ax=ax2)
    ax2.legend(fontsize=8)
    ax2.axis("off")

    tight_layout()
    figpath = joinpath(OUTDIR, "u_map_sim$(d_worst.sim_idx).png")
    savefig(figpath, dpi=200)
    PythonPlot.close("all")
    println("  Saved $figpath")
end

# Fig 4: final QE φ max comparison
fig, axes = subplots(1, 2, figsize=(12, 4))

ax1 = axes[0]

if !isempty(normal_diags)
    ax1.hist([d.max_phi for d in normal_diags], bins=10, color="C0", alpha=0.6, label="normal")
end

if !isempty(outlier_diags)
    for d in outlier_diags
        ax1.axvline(d.max_phi, color="C1", lw=1.2, alpha=0.75)
    end
    ax1.axvline(NaN, color="C1", lw=1.2, label="outliers")
end

ax1.set_xlabel(L"$\max |\hat{\phi}_{QE,F}|$")
ax1.set_title("Final QE Fourier maximum")
ax1.legend(fontsize=9)

ax2 = axes[1]

if !isempty(normal_diags)
    ax2.scatter([d.max_u for d in normal_diags],
                [d.max_phi for d in normal_diags],
                color="C0", alpha=0.75, label="normal")
end

if !isempty(outlier_diags)
    ax2.scatter([d.max_u for d in outlier_diags],
                [d.max_phi for d in outlier_diags],
                color="C1", alpha=0.9, label="outliers")
    for d in outlier_diags
        ax2.annotate(string(d.sim_idx), (d.max_u, d.max_phi), fontsize=7)
    end
end

ax2.set_xlabel(L"$\max |u_F|$")
ax2.set_ylabel(L"$\max |\hat{\phi}_{QE,F}|$")
ax2.set_title("Does inverse-filter blow-up propagate to QE φ?")
ax2.legend(fontsize=9)

tight_layout()
figpath = joinpath(OUTDIR, "qe_phi_fourier_max.png")
savefig(figpath, dpi=200)
PythonPlot.close("all")
println("  Saved $figpath")

println("\nFigures saved to $OUTDIR/")
println("Done.")