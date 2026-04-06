#!/usr/bin/env julia
# rhoL_gradpatch_nowindow.jl
#
# Diagnostic: Global reconstruction, gradient-selected local evaluation.
#
# 1) Use reconstructions on the full 512×512 patch (already saved in PHI_FILE).
# 2) For each sim, select a small sub-patch (e.g. 24×24 ~ 0.30°)
#    where |∇T| is in a target range and approximately constant across the patch.
# 3) Compute ρ_L on those selected patches, using mean spectra first (avoids mean-of-ratios bias).
#
# Important fixes:
#   - gradient selection uses Utils.grad_fft in the same style as error_mean.jl
#   - full T map is wrapped as a FlatMap-like object before grad_fft
#   - arrays are forced real Float64 where needed
#   - candidate-centre search is made more efficient

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using Statistics: mean, std, quantile
using PythonPlot
using JLD2
using Printf
using LinearAlgebra

include("utils.jl")
using .Utils

# ── Inputs ─────────────────────────────────────────────────────────────
const PHI_FILE     = "results/checkpoints/phi_maps.jld2"

const θpix   = 0.7438046267475303   # arcmin
const Nside  = 512

# Patch size: 16px ≈ 0.198°, 24px ≈ 0.298°, 32px ≈ 0.397°
const N_CROP = 24
const Δℓ     = 2000

# Gradient selection (units must match error_mean.jl / Utils.grad_fft)
# Two bands are compared side by side:
#   Band A (high gradient):  old [12.8, 13.8) → top ~10–15% of field by gradient strength
#   Band B (moderate gradient): old [9, 10)   → just above median, uniform-slope regions
# From diagnostic stats (new units): p50≈37600, p90≈68800 µK/rad
const BANDS = [(60000.0, 65000.0), (42000.0, 47000.0)]
# Bands in µK/rad (current grad_fft output units). Scale factor vs old µK/arcmin ≈ 4700.
# Band A: old [12.8, 13.8) → high gradient (~p85–p90)
# Band B: old [8.9, 10.0) → moderate gradient, just above median (~p52–p60)

# Within-patch uniformity: std(|gradT|) / mean(|gradT|) must be below this.
const GRAD_RMS_FRAC_MAX = 0.25

# Print gradient stats for ALL sims on first run
const N_DEBUG_STATS = 3

# ── Geometry helpers ─────────────────────────────────────────────────────
const θpix_rad = θpix * π / 10800
const L_nyq = π / θpix_rad
valid_L(ℓ) = ℓ .< 11000

patch_deg = round(N_CROP * θpix / 60; digits=3)
println("Patch: $(N_CROP)×$(N_CROP)  (~$(patch_deg)°)  |  two gradient bands: $(BANDS)  rms/mean<$GRAD_RMS_FRAC_MAX")

# ── Crop helper (by centre pixel) ────────────────────────────────────────
function crop_center_xy(arr::AbstractMatrix, cx::Int, cy::Int, N::Int)
    h = N ÷ 2
    x1 = cx - h + 1
    x2 = cx + h
    y1 = cy - h + 1
    y2 = cy + h
    return arr[y1:y2, x1:x2]  # row=y, col=x
end

# ── 2D Hann apodization window ───────────────────────────────────────────
function hann2d(Ny::Int, Nx::Int)
    wy = 0.5 .* (1 .- cos.(2π .* (0:Ny-1) ./ Ny))
    wx = 0.5 .* (1 .- cos.(2π .* (0:Nx-1) ./ Nx))
    return wy .* wx'
end
const WIN = hann2d(N_CROP, N_CROP)

# ── Correlation coefficient helper ───────────────────────────────────────
function rho_from_means(Ctr, Ctt, Crr)
    denom = sqrt.(max.(Ctt .* Crr, 0.0))
    clamp.(Ctr ./ max.(denom, 1e-30), -1.0, 1.0)
end

# ── Wrap cropped arrays as FlatMaps (small patch metadata) ───────────────
function make_wrap_small(; θpix, N_CROP)
    Cℓ  = camb(r=0.05, ℓmax=21000)
    Cℓn = noiseCℓs(μKarcminT=1.0, ℓknee=0)

    sim_ref = load_sim(
        seed=1001, Cℓ=Cℓ, Cℓn=Cℓn,
        θpix=θpix, T=Float64, Nside=N_CROP,
        beamFWHM=1.0, pol=:I, bandpass_mask=LowPass(6000),
        pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0)
    )

    ϕ_ref = Map(sim_ref.ϕ)

    wrap_small(arr) = typeof(ϕ_ref)(Float64.(real.(arr)), ϕ_ref.metadata)
    return wrap_small
end
wrap_small = make_wrap_small(; θpix=θpix, N_CROP=N_CROP)


# ── Find ALL constant-gradient patches via sliding-window scan ─────────────
# Uses 2D prefix sums to compute patch mean and std in O(Nx*Ny) time.
function find_all_valid_patches(G::AbstractMatrix, N::Int, gmin::Real, gmax::Real,
                                rms_frac_max::Real)
    Ny, Nx = size(G)
    h = N ÷ 2
    n = Float64(N * N)

    # 2D prefix sums for G and G^2
    cs1 = cumsum(cumsum(G,       dims=1), dims=2)
    cs2 = cumsum(cumsum(G .^ 2,  dims=1), dims=2)

    function rect_sum(cs, r1, r2, c1, c2)
        s = cs[r2, c2]
        r1 > 1 && (s -= cs[r1-1, c2])
        c1 > 1 && (s -= cs[r2, c1-1])
        r1 > 1 && c1 > 1 && (s += cs[r1-1, c1-1])
        return s
    end

    patches = Tuple{Int,Int,Float64,Float64}[]  # (cx, cy, μ, frac)
    @inbounds for cy in h:(Ny-h), cx in h:(Nx-h)
        r1, r2 = cy - h + 1, cy + h
        c1, c2 = cx - h + 1, cx + h
        s1 = rect_sum(cs1, r1, r2, c1, c2)
        μ = s1 / n
        (μ < gmin || μ >= gmax) && continue
        s2   = rect_sum(cs2, r1, r2, c1, c2)
        frac = sqrt(max(s2 / n - μ^2, 0.0)) / μ
        frac > rms_frac_max && continue
        push!(patches, (cx, cy, μ, frac))
    end
    return patches
end

# ── Per-sim spectral accumulators (for jackknife over sims) ──────────────────
# Each SimResult holds the SUM of spectra over all valid patches in that sim.
struct SimResult
    s         :: Int
    n_patches :: Int   # how many valid patches contributed
    Σtt :: Vector{Float64}
    Σqq :: Vector{Float64}
    Σtq :: Vector{Float64}
    Σjj :: Vector{Float64}
    Σtj :: Vector{Float64}
    Σmm :: Vector{Float64}
    Σtm :: Vector{Float64}
end

function run_analysis(grad_min::Real, grad_max::Real)
    sim_results = SimResult[]
    ℓ_vec = nothing
    nsims_done = 0
    dbg_count  = 0

    jldopen(PHI_FILE, "r") do f
        idxs = sort([parse(Int, m.captures[1])
            for key in keys(f)
            for m in [match(r"^sim_(\d+)$", key)]
            if m !== nothing])

        for s in idxs
            println("Sim $s  [band $(grad_min)-$(grad_max)]")

            ϕ_true_full  = read(f, "sim_$s/ϕ_true")
            ϕ_qe_full    = read(f, "sim_$s/ϕ_qe")
            ϕ_joint_full = read(f, "sim_$s/ϕ_joint")
            ϕ_marg_full  = read(f, "sim_$s/ϕ_marg")

            if !haskey(f, "sim_$s/seed")
                println("  No seed available → skip")
                continue
            end
            seed_s = read(f, "sim_$s/seed")

            # Regenerate unlensed f to compute gradient
            sim_s = load_sim(seed=seed_s,
                             Cℓ=camb(r=0.05, ℓmax=21000),
                             Cℓn=noiseCℓs(μKarcminT=1.0, ℓknee=0),
                             θpix=θpix, T=Float64, Nside=Nside,
                             beamFWHM=1.0, pol=:I,
                             bandpass_mask=LowPass(6000),
                             pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))
            _, _, G_field = Utils.grad_fft(sim_s.f)
            G = Float64.(G_field)

            nsims_done += 1

            if dbg_count < N_DEBUG_STATS
                dbg_count += 1
                @printf("  G stats: min=%.3g p50=%.3g p90=%.3g p99=%.3g max=%.3g\n",
                        minimum(G), quantile(vec(G), 0.50), quantile(vec(G), 0.90),
                        quantile(vec(G), 0.99), maximum(G))
            end

            patches = find_all_valid_patches(G, N_CROP, grad_min, grad_max, GRAD_RMS_FRAC_MAX)
            if isempty(patches)
                println("  No valid patches found → skip")
                continue
            end
            @printf("  Found %d valid patches (|gradT| ∈ [%.1f,%.1f), rms/mean<%.2f)\n",
                    length(patches), grad_min, grad_max, GRAD_RMS_FRAC_MAX)

            Σtt = Σqq = Σtq = Σjj = Σtj = Σmm = Σtm = nothing

            for (cx, cy, gμ, gfrac) in patches
                ϕt = wrap_small(WIN .* crop_center_xy(ϕ_true_full,  cx, cy, N_CROP))
                ϕq = wrap_small(WIN .* crop_center_xy(ϕ_qe_full,    cx, cy, N_CROP))
                ϕj = wrap_small(WIN .* crop_center_xy(ϕ_joint_full, cx, cy, N_CROP))
                ϕm = wrap_small(WIN .* crop_center_xy(ϕ_marg_full,  cx, cy, N_CROP))

                cl = get_Cℓ(ϕt; Δℓ=Δℓ)
                tt = Float64.(cl.Cℓ)
                ℓ  = Float64.(collect(cl.ℓ))
                if ℓ_vec === nothing
                    ℓ_vec = ℓ
                end

                qq = Float64.(get_Cℓ(ϕq;      Δℓ=Δℓ).Cℓ)
                jj = Float64.(get_Cℓ(ϕj;      Δℓ=Δℓ).Cℓ)
                mm = Float64.(get_Cℓ(ϕm;      Δℓ=Δℓ).Cℓ)
                tq = Float64.(get_Cℓ(ϕt, ϕq; Δℓ=Δℓ).Cℓ)
                tj = Float64.(get_Cℓ(ϕt, ϕj; Δℓ=Δℓ).Cℓ)
                tm = Float64.(get_Cℓ(ϕt, ϕm; Δℓ=Δℓ).Cℓ)

                Σtt = Σtt === nothing ? tt : Σtt .+ tt
                Σqq = Σqq === nothing ? qq : Σqq .+ qq
                Σtq = Σtq === nothing ? tq : Σtq .+ tq
                Σjj = Σjj === nothing ? jj : Σjj .+ jj
                Σtj = Σtj === nothing ? tj : Σtj .+ tj
                Σmm = Σmm === nothing ? mm : Σmm .+ mm
                Σtm = Σtm === nothing ? tm : Σtm .+ tm
            end

            push!(sim_results, SimResult(s, length(patches), Σtt, Σqq, Σtq, Σjj, Σtj, Σmm, Σtm))
        end
    end

    n_sims  = length(sim_results)
    n_total = n_sims > 0 ? sum(r.n_patches for r in sim_results) : 0
    println("\nBand [$grad_min, $grad_max): processed $nsims_done sims | valid: $n_sims | patches: $n_total")
    n_sims == 0 && error("No valid patches for band [$grad_min, $grad_max); relax GRAD_RMS_FRAC_MAX.")
    return sim_results, ℓ_vec
end

sim_results_A, ℓ_vec_A = run_analysis(BANDS[1]...)
sim_results_B, ℓ_vec_B = run_analysis(BANDS[2]...)

# ── ρ_L from grand mean spectra + jackknife SEM over sims ────────────────────
function grand_rho(rs, f_cross, f_true, f_recon)
    Σc = sum(getfield(r, f_cross) for r in rs)
    Σt = sum(getfield(r, f_true)  for r in rs)
    Σr = sum(getfield(r, f_recon) for r in rs)
    n   = Float64(sum(r.n_patches for r in rs))
    rho_from_means(Σc ./ n, Σt ./ n, Σr ./ n)
end

function jackknife_sem(rs, f_cross, f_true, f_recon)
    n = length(rs)
    n < 2 && return nothing
    rho_jk = [grand_rho([rs[j] for j in 1:n if j != i], f_cross, f_true, f_recon) for i in 1:n]
    mat = reduce(hcat, rho_jk)
    vec(sqrt((n-1)^2 / n) .* std(mat; dims=2))
end

function compute_rhos(sim_results)
    ρ_q   = grand_rho(sim_results, :Σtq, :Σtt, :Σqq)
    ρ_j   = grand_rho(sim_results, :Σtj, :Σtt, :Σjj)
    ρ_m   = grand_rho(sim_results, :Σtm, :Σtt, :Σmm)
    sem_q = jackknife_sem(sim_results, :Σtq, :Σtt, :Σqq)
    sem_j = jackknife_sem(sim_results, :Σtj, :Σtt, :Σjj)
    sem_m = jackknife_sem(sim_results, :Σtm, :Σtt, :Σmm)
    return ρ_q, ρ_j, ρ_m, sem_q, sem_j, sem_m
end

ρ_q_A, ρ_j_A, ρ_m_A, sem_q_A, sem_j_A, sem_m_A = compute_rhos(sim_results_A)
ρ_q_B, ρ_j_B, ρ_m_B, sem_q_B, sem_j_B, sem_m_B = compute_rhos(sim_results_B)

n_sims_A  = length(sim_results_A)
n_total_A = n_sims_A > 0 ? sum(r.n_patches for r in sim_results_A) : 0
n_sims_B  = length(sim_results_B)
n_total_B = n_sims_B > 0 ? sum(r.n_patches for r in sim_results_B) : 0

# ── Plot: side-by-side for the two gradient bands ─────────────────────────────
fig, (axA, axB) = PythonPlot.subplots(1, 2; figsize=(13.0, 5.5), constrained_layout=true)

function fill_rho_ax(ax, ℓ_vec, ρ_q, ρ_j, ρ_m, sem_q, sem_j, sem_m, band_min, band_max, n_sims, n_total)
    mask = valid_L(ℓ_vec)
    for (ρ, sem, col, lab) in [
        (ρ_q, sem_q, "#D62728", "QE (WF)"),
        (ρ_j, sem_j, "#1F77B4", "MAP joint"),
        (ρ_m, sem_m, "#2CA02C", "MAP marg"),
    ]
        m = mask .& isfinite.(ρ)
        ax.plot(ℓ_vec[m], ρ[m]; color=col, label=lab, linewidth=2)
        sem !== nothing && ax.fill_between(ℓ_vec[m], (ρ .- sem)[m], (ρ .+ sem)[m]; color=col, alpha=0.2)
    end
    ax.axhline(1; color="k", linestyle=":")
    ax.axhline(0; color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel(L"L", fontsize=12)
    ax.set_ylabel(L"\rho_L", fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("|∇T| ∈ [$band_min, $band_max) µK/arcmin  ($n_sims sims, $n_total patches)", fontsize=10, pad=6)
    ax.legend(frameon=false, fontsize=10)
end

fill_rho_ax(axA, ℓ_vec_A, ρ_q_A, ρ_j_A, ρ_m_A, sem_q_A, sem_j_A, sem_m_A,
            BANDS[1][1], BANDS[1][2], n_sims_A, n_total_A)
fill_rho_ax(axB, ℓ_vec_B, ρ_q_B, ρ_j_B, ρ_m_B, sem_q_B, sem_j_B, sem_m_B,
            BANDS[2][1], BANDS[2][2], n_sims_B, n_total_B)

outfile = "results/rho_L_gradpatch_$(N_CROP)px.png"
fig.savefig(outfile, dpi=150)
println("Saved $outfile")
PythonPlot.plotclose("all")
