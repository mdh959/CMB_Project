#!/usr/bin/env julia
#
# test_gi_rdn0_phase1.jl
#
# Phase 1: fixed-denominator GI RDN0 scaling test.
#
# Key question: does N0_RD = <C(phi_DS) + C(phi_SD) - C(phi_SS)> give
# O(eps^2) residual (vs O(eps) for MC N0) under a deliberate covariance
# mismatch eps?  If yes, per-sim GI RDN0 will reduce sigma_auto.
#
# Design:
#   D = sqrt(1+eps) * unlensed_CMB_A + noise_A   [data, power = (1+eps)*C_fid + N]
#   S = unlensed_CMB_B + noise_B                  [sim,  power = C_fid + N]
#   T = sqrt(1+eps) * unlensed_CMB_T + noise_T    [truth, same covariance as D]
#
#   phi_XY = gi_twoleg(X, Y) with FROZEN fiducial denominator sigma_fid
#
#   N0_MC   = <C(phi_SS)>
#   N0_RD   = <C(phi_DS) + C(phi_SD) - C(phi_SS)>
#   N0_true = <C(phi_TT)>
#
#   Theory: R_MC = N0_MC - N0_true ~ -2*eps * N0_fid  [O(eps)]
#           R_RD = N0_RD - N0_true ~ -eps^2 * N0_fid  [O(eps^2)]
#
# Run: julia test_gi_rdn0_phase1.jl

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using CMBLensing: m_rfft, m_irfft
using Statistics: mean, std
using Printf

include("utils.jl")
using .Utils

# ── Parameters ─────────────────────────────────────────────────────────────────
const nsims_test = 50      # sims per epsilon value (pre-loaded once)
const n_fid_est  = 20      # sims for fiducial denominator pre-computation
const seed0_A    = 5000    # data map base seeds:  5001…5050
const seed0_B    = 5100    # sim map base seeds:   5101…5150
const seed0_T    = 5200    # truth map base seeds: 5201…5250
const seed0_fid  = 5300    # fiducial denom seeds: 5301…5320

const Lhp_gi  = 4000
const Lmax    = 12000
const Δℓ_spec = 30

# Epsilon values for scaling test (no 0 — avoids log(0) in slope fit)
const eps_values = [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]

# CMB power spectrum
const Cℓ_fid = camb(r=0.05, ℓmax=33000)

# Use ultra-low noise so T ≈ beam*CMB: scaling T by sqrt(1+eps) ≈ scaling signal only
const load_kw = (
    Cℓ    = Cℓ_fid,
    Cℓn   = noiseCℓs(μKarcminT=0.01, ℓknee=0, ℓmax=Lmax),
    θpix  = 0.7438046267475303, T=Float64, Nside=512, beamFWHM=1.0, pol=:I,
    bandpass_mask=LowPass(Lmax),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

println("GI RDN0 Phase 1: Fixed-Denominator Scaling Test")
println("=" ^ 70)

# ── Step 1: Reference sim for geometry ─────────────────────────────────────────
println("Loading reference sim for geometry...")
ref_result  = load_sim(; seed=seed0_A+1, load_kw...)
T_ref_map   = Map(ref_result.f)
ref_proj    = T_ref_map.proj

# Determine rfft output dimensions empirically (same as grad_fft does internally)
F_test   = m_rfft(T_ref_map.arr, (1, 2))
NyF, NxF = size(F_test)

# Fourier grid for frozen denominator and numerator (rfft layout)
Lx2D_fid = repeat(ref_proj.ℓx[1:NxF]', NyF, 1)
Ly2D_fid = repeat(ref_proj.ℓy[1:NyF],  1,   NxF)
L2_fid   = @. Lx2D_fid^2 + Ly2D_fid^2
L_mask   = @. (L2_fid > Lhp_gi^2) & (L2_fid ≤ Lmax^2)

# ── Step 2: Frozen fiducial denominator ────────────────────────────────────────
# Computed once from C_fid sims; same for all phi_XY calls regardless of X, Y.
println("Computing frozen fiducial denominator from $n_fid_est sims...")
σ_xx_fid, σ_xy_fid, σ_yy_fid = let
    acc_xx = 0.0; acc_xy = 0.0; acc_yy = 0.0
    for k in 1:n_fid_est
        r    = load_sim(; seed=seed0_fid+k, load_kw...)
        T_lp = Map(LowPass(Lhp_gi) * r.f)
        gx, gy, _ = grad_fft(T_lp)
        acc_xx += mean(gx.^2)
        acc_xy += mean(gx .* gy)
        acc_yy += mean(gy.^2)
        print("\r  fid sim $k/$n_fid_est"); flush(stdout)
    end
    println()
    acc_xx/n_fid_est, acc_xy/n_fid_est, acc_yy/n_fid_est
end
denom_thresh = 1e-8 * max(σ_xx_fid, σ_yy_fid)
@printf "  σ_fid: σ_xx=%.3e  σ_xy=%.3e  σ_yy=%.3e\n" σ_xx_fid σ_xy_fid σ_yy_fid

# Precompute the frozen denominator array (same for every gi_twoleg call)
denom_fid = @. Lx2D_fid^2 * σ_xx_fid +
               2 * Lx2D_fid * Ly2D_fid * σ_xy_fid +
               Ly2D_fid^2 * σ_yy_fid

# ── Two-leg GI estimator with frozen denominator ───────────────────────────────
# phi_XY(L) = -i * (Lx * FFT[gx_X * Y_hp] + Ly * FFT[gy_X * Y_hp]) / denom_fid(L)
# where gx_X, gy_X = gradient of LowPass(Lhp)*X  (from the data)
# and denom_fid is computed from fiducial C_fid (frozen, same regardless of X, Y)
function gi_twoleg(X_in, Y_in)
    X_lp     = Map(LowPass(Lhp_gi) * X_in)
    Y_hp     = Map(HighPass(Lhp_gi) * Y_in)
    T_hp_arr = Y_hp.arr
    gx, gy, _ = grad_fft(X_lp)          # gradient of low-passed X leg
    A_F = m_rfft(gx .* T_hp_arr, (1,2)) # FFT[ dX/dx * Y_hp ]
    B_F = m_rfft(gy .* T_hp_arr, (1,2)) # FFT[ dX/dy * Y_hp ]
    numer = @. -im * (Lx2D_fid * A_F + Ly2D_fid * B_F)
    φ_F   = @. ifelse(L_mask & (abs(denom_fid) > denom_thresh),
                      numer / denom_fid, complex(0.0))
    FlatFourier(φ_F, ref_proj)
end

# ── Step 3: Pre-load all base maps (3 × nsims_test, done once) ─────────────────
println("Pre-loading $(3*nsims_test) base maps (A/B/T sets)...")
maps_A = Vector{typeof(T_ref_map)}(undef, nsims_test)
maps_B = Vector{typeof(T_ref_map)}(undef, nsims_test)
maps_T = Vector{typeof(T_ref_map)}(undef, nsims_test)
for s in 1:nsims_test
    maps_A[s] = Map(load_sim(; seed=seed0_A+s, load_kw...).f)
    maps_B[s] = Map(load_sim(; seed=seed0_B+s, load_kw...).f)
    maps_T[s] = Map(load_sim(; seed=seed0_T+s, load_kw...).f)
    print("\r  loaded $s/$(nsims_test)"); flush(stdout)
end
println()

# Get ℓ axis from a test phi field (non-zero bins live where L_mask is true)
cl_ref_phi = get_Cℓ(gi_twoleg(maps_A[1], maps_A[1]); Δℓ=Δℓ_spec)
ℓ_axis     = Float64.(collect(cl_ref_phi.ℓ))
n_ell      = length(ℓ_axis)
ℓ_select   = findall(@. ℓ_axis >= Lhp_gi && ℓ_axis <= 10000)
println("  ℓ axis: $(round(Int,ℓ_axis[1]))..$(round(Int,ℓ_axis[end])) ($(n_ell) bins), " *
        "analysis range: L=$(Lhp_gi)–10000 ($(length(ℓ_select)) bins)")

# ── Step 4: Main loop over epsilon ─────────────────────────────────────────────
println("\nRunning scaling test over eps = $eps_values ...")

N0_mc_means   = Dict{Float64, Vector{Float64}}()
N0_rd_means   = Dict{Float64, Vector{Float64}}()
N0_true_means = Dict{Float64, Vector{Float64}}()

for eps in eps_values
    sc   = sqrt(1 + eps)   # amplitude scale factor for signal
    println("\n── eps = $eps  (scale = $(round(sc; digits=4))) ──")

    acc_mc   = zeros(n_ell)
    acc_rd   = zeros(n_ell)
    acc_true = zeros(n_ell)

    for s in 1:nsims_test
        # D has power ≈ (1+eps)*C_fid (signal scaled, noise negligible at 0.01 µK·arcmin)
        T_D = sc * maps_A[s]
        # S has power ≈ C_fid (unscaled fiducial)
        T_S = maps_B[s]
        # T_true has same covariance as D (independent realization)
        T_T = sc * maps_T[s]

        ϕ_ds = gi_twoleg(T_D, T_S)
        ϕ_sd = gi_twoleg(T_S, T_D)
        ϕ_ss = gi_twoleg(T_S, T_S)
        ϕ_tt = gi_twoleg(T_T, T_T)

        cl_ss = Float64.(get_Cℓ(ϕ_ss; Δℓ=Δℓ_spec).Cℓ)
        cl_ds = Float64.(get_Cℓ(ϕ_ds; Δℓ=Δℓ_spec).Cℓ)
        cl_sd = Float64.(get_Cℓ(ϕ_sd; Δℓ=Δℓ_spec).Cℓ)
        cl_tt = Float64.(get_Cℓ(ϕ_tt; Δℓ=Δℓ_spec).Cℓ)

        # N0_MC   = C(phi_SS)                          [O(eps) residual expected]
        # N0_RD   = C(phi_DS) + C(phi_SD) - C(phi_SS)  [O(eps^2) residual expected]
        # N0_true = C(phi_TT)                          [truth at data covariance]
        acc_mc   .+= cl_ss
        acc_rd   .+= cl_ds .+ cl_sd .- cl_ss
        acc_true .+= cl_tt

        print("\r  sim $s/$nsims_test"); flush(stdout)
    end
    println()

    N0_mc_means[eps]   = acc_mc   ./ nsims_test
    N0_rd_means[eps]   = acc_rd   ./ nsims_test
    N0_true_means[eps] = acc_true ./ nsims_test

    # Quick residual summary for this eps
    R_mc  = N0_mc_means[eps][ℓ_select]   .- N0_true_means[eps][ℓ_select]
    R_rd  = N0_rd_means[eps][ℓ_select]   .- N0_true_means[eps][ℓ_select]
    N0_ref = mean(abs.(N0_true_means[eps][ℓ_select]))
    @printf "  MC residual RMS: %.3e  (%.1f%% of N0_true)\n" sqrt(mean(R_mc.^2)) 100*sqrt(mean(R_mc.^2))/N0_ref
    @printf "  RD residual RMS: %.3e  (%.1f%% of N0_true)\n" sqrt(mean(R_rd.^2)) 100*sqrt(mean(R_rd.^2))/N0_ref
end

# ── Step 5: Scaling analysis ────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("Scaling analysis (L = $(Lhp_gi)–10000, $(nsims_test) sims per eps)")
println("=" ^ 70)
@printf "%-8s  %-14s  %-14s  %-10s\n" "eps" "RMS_MC" "RMS_RD" "RD/MC"
println("-" ^ 50)

rms_mc_vec  = Float64[]
rms_rd_vec  = Float64[]
eps_abs_vec = Float64[]

for eps in eps_values
    R_mc = N0_mc_means[eps][ℓ_select] .- N0_true_means[eps][ℓ_select]
    R_rd = N0_rd_means[eps][ℓ_select] .- N0_true_means[eps][ℓ_select]
    rms_mc = sqrt(mean(R_mc.^2))
    rms_rd = sqrt(mean(R_rd.^2))
    push!(rms_mc_vec, rms_mc); push!(rms_rd_vec, rms_rd)
    push!(eps_abs_vec, abs(eps))
    @printf "%-8.3f  %-14.3e  %-14.3e  %-10.3f\n" eps rms_mc rms_rd rms_rd/rms_mc
end

# Log-log slope fit (RMS vs |eps|, both signs pooled)
# Noise floor from MC noise (50 sims) contaminates small-eps points.
# Estimate floor as the minimum RD RMS across eps values, then fit only
# on points where the floor-subtracted signal exceeds the floor itself.
println()
log_eps = log.(eps_abs_vec)
function loglog_slope(log_x, log_y)
    n = length(log_x)
    (n*sum(log_x.*log_y) - sum(log_x)*sum(log_y)) /
    (n*sum(log_x.^2) - sum(log_x)^2)
end
mc_slope = loglog_slope(log_eps, log.(rms_mc_vec))
rd_slope = loglog_slope(log_eps, log.(rms_rd_vec))

# Floor-subtracted RD slope: only fit points where signal > noise floor
rd_floor   = minimum(rms_rd_vec)
rd_signal  = sqrt.(max.(rms_rd_vec.^2 .- rd_floor^2, 0.0))
above_floor = findall(rd_signal .> rd_floor)
rd_slope_sig = length(above_floor) >= 2 ?
    loglog_slope(log_eps[above_floor], log.(rd_signal[above_floor])) : NaN

@printf "Log-log slope  MC: %.2f (expect ≈1)    RD (all): %.2f    RD (signal-dominated): %.2f (expect ≈2)\n" mc_slope rd_slope rd_slope_sig
@printf "Noise floor estimate (min RD RMS): %.3e  (%d/%d eps values above floor)\n" rd_floor length(above_floor) length(rms_rd_vec)

if abs(mc_slope - 1.0) < 0.4 && !isnan(rd_slope_sig) && abs(rd_slope_sig - 2.0) < 0.7
    println("PASS: GI RDN0 works for fixed denominator — per-sim DS+SD-SS is O(eps²).")
    println("      Proceed to Step 7: test with data-dependent denominator.")
elseif abs(mc_slope - 1.0) < 0.4 && length(above_floor) < 2
    println("INCONCLUSIVE: RD signal below noise floor at all tested eps — increase nsims or use larger eps.")
elseif abs(rd_slope - 1.0) < 0.4 && (isnan(rd_slope_sig) || abs(rd_slope_sig - 1.0) < 0.4)
    println("WARN: RD scaling is O(eps) not O(eps²) — likely missing a cross-term from")
    println("      the gradient-dependent normalization.  Fixed-denom test passed but")
    println("      production GI needs additional corrections.")
else
    println("FAIL: unexpected scaling — check implementation or increase nsims_test.")
end
