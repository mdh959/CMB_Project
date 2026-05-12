#!/usr/bin/env julia
#
# test_gi_rdn0_phase2.jl
#
# Phase 2: per-realization denominator GI RDN0 scaling test.
#
# Phase 1 confirmed O(eps²) residual with a FROZEN fiducial denominator.
# Phase 2 tests the production case: denominator computed from the GRADIENT
# LEG of each specific call (gi_estimate style, Lgrad=2000, Lhp=4000).
#
# With per-map denominator:
#   denom_X(L) = Lx²<gx_X²> + 2LxLy<gx_X gy_X> + Ly²<gy_X²>  [from map X]
#
# Theory (per-map denominator):
#   <C(phi_DS)> = N0_fid / (1+eps)   [denom_D ~ (1+eps)*denom_fid cancels 1 factor of sc]
#   <C(phi_SD)> = (1+eps) * N0_fid   [denom_S = denom_fid; numerator hp-leg scales as sc]
#   <C(phi_SS)> = N0_fid
#   <C(phi_TT)> = N0_fid             [sc cancels in numer and denom — eps-independent!]
#
#   N0_RD - N0_true = eps² / (1+eps) * N0_fid  → O(eps²) ✓
#
# Run: julia test_gi_rdn0_phase2.jl

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using CMBLensing: m_rfft, m_irfft
using Statistics: mean, std
using Printf

include("utils.jl")
using .Utils

# ── Parameters ─────────────────────────────────────────────────────────────────
const nsims_test = 50
const seed0_A    = 5000
const seed0_B    = 5100
const seed0_T    = 5200

const Lgrad_gi = 2000    # matches production gi_estimate (gradient cutoff)
const Lhp_gi   = 4000    # HP cutoff (same as production)
const Lmax      = 12000
const Δℓ_spec   = 30

const eps_values = [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]

const Cℓ_fid = camb(r=0.05, ℓmax=33000)

const load_kw = (
    Cℓ    = Cℓ_fid,
    Cℓn   = noiseCℓs(μKarcminT=0.01, ℓknee=0, ℓmax=Lmax),
    θpix  = 0.7438046267475303, T=Float64, Nside=512, beamFWHM=1.0, pol=:I,
    bandpass_mask=LowPass(Lmax),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

println("GI RDN0 Phase 2: Per-Realization Denominator Scaling Test")
println("=" ^ 70)

# ── Reference geometry ─────────────────────────────────────────────────────────
println("Loading reference sim for geometry...")
ref_result = load_sim(; seed=seed0_A+1, load_kw...)
T_ref_map  = Map(ref_result.f)
ref_proj   = T_ref_map.proj

F_test   = m_rfft(T_ref_map.arr, (1, 2))
NyF, NxF = size(F_test)

Lx2D = repeat(ref_proj.ℓx[1:NxF]', NyF, 1)
Ly2D = repeat(ref_proj.ℓy[1:NyF],  1,   NxF)
L2   = @. Lx2D^2 + Ly2D^2
L_mask = @. (L2 > Lhp_gi^2) & (L2 ≤ Lmax^2)

# ── Per-map denominator GI estimator ──────────────────────────────────────────
# Denominator computed from gradient leg of each specific call (production style).
# Gradient uses Lgrad=2000; HP uses Lhp=4000; matching gi_estimate exactly.
function gi_twoleg_permap(X_in, Y_in)
    X_lp = Map(LowPass(Lgrad_gi) * X_in)
    Y_hp = Map(HighPass(Lhp_gi)  * Y_in)

    gx, gy, _ = grad_fft(X_lp)

    σ_xx = mean(gx.^2)
    σ_xy = mean(gx .* gy)
    σ_yy = mean(gy.^2)
    denom_X   = @. Lx2D^2 * σ_xx + 2*Lx2D*Ly2D*σ_xy + Ly2D^2*σ_yy
    thresh_X  = 1e-8 * max(σ_xx, σ_yy)

    A_F = m_rfft(gx .* Y_hp.arr, (1,2))
    B_F = m_rfft(gy .* Y_hp.arr, (1,2))
    numer = @. -im * (Lx2D * A_F + Ly2D * B_F)
    φ_F   = @. ifelse(L_mask & (abs(denom_X) > thresh_X),
                      numer / denom_X, complex(0.0))
    FlatFourier(φ_F, ref_proj)
end

# ── Pre-load base maps ─────────────────────────────────────────────────────────
println("Pre-loading $(3*nsims_test) base maps (A/B/T sets, using unlensed CMB r.f)...")
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

cl_ref = get_Cℓ(gi_twoleg_permap(maps_A[1], maps_A[1]); Δℓ=Δℓ_spec)
ℓ_axis   = Float64.(collect(cl_ref.ℓ))
n_ell    = length(ℓ_axis)
ℓ_select = findall(@. ℓ_axis >= Lhp_gi && ℓ_axis <= 10000)
println("  ℓ axis: $(round(Int,ℓ_axis[1]))..$(round(Int,ℓ_axis[end])) ($(n_ell) bins), " *
        "analysis range: L=$(Lhp_gi)–10000 ($(length(ℓ_select)) bins)")

# ── Main loop ──────────────────────────────────────────────────────────────────
println("\nRunning scaling test over eps = $eps_values ...")

N0_mc_means   = Dict{Float64, Vector{Float64}}()
N0_rd_means   = Dict{Float64, Vector{Float64}}()
N0_true_means = Dict{Float64, Vector{Float64}}()

for eps in eps_values
    sc = sqrt(1 + eps)
    println("\n── eps = $eps  (scale = $(round(sc; digits=4))) ──")

    acc_mc   = zeros(n_ell)
    acc_rd   = zeros(n_ell)
    acc_true = zeros(n_ell)

    for s in 1:nsims_test
        T_D = sc * maps_A[s]   # power ≈ (1+eps)*C_fid
        T_S = maps_B[s]        # power ≈ C_fid
        T_T = sc * maps_T[s]   # same covariance as D, independent

        ϕ_ds = gi_twoleg_permap(T_D, T_S)
        ϕ_sd = gi_twoleg_permap(T_S, T_D)
        ϕ_ss = gi_twoleg_permap(T_S, T_S)
        ϕ_tt = gi_twoleg_permap(T_T, T_T)

        cl_ss = Float64.(get_Cℓ(ϕ_ss; Δℓ=Δℓ_spec).Cℓ)
        cl_ds = Float64.(get_Cℓ(ϕ_ds; Δℓ=Δℓ_spec).Cℓ)
        cl_sd = Float64.(get_Cℓ(ϕ_sd; Δℓ=Δℓ_spec).Cℓ)
        cl_tt = Float64.(get_Cℓ(ϕ_tt; Δℓ=Δℓ_spec).Cℓ)

        acc_mc   .+= cl_ss
        acc_rd   .+= cl_ds .+ cl_sd .- cl_ss
        acc_true .+= cl_tt

        print("\r  sim $s/$nsims_test"); flush(stdout)
    end
    println()

    N0_mc_means[eps]   = acc_mc   ./ nsims_test
    N0_rd_means[eps]   = acc_rd   ./ nsims_test
    N0_true_means[eps] = acc_true ./ nsims_test

    R_mc  = N0_mc_means[eps][ℓ_select] .- N0_true_means[eps][ℓ_select]
    R_rd  = N0_rd_means[eps][ℓ_select] .- N0_true_means[eps][ℓ_select]
    N0_ref = mean(abs.(N0_true_means[eps][ℓ_select]))
    @printf "  MC residual RMS: %.3e  (%.1f%% of N0_true)\n" sqrt(mean(R_mc.^2)) 100*sqrt(mean(R_mc.^2))/N0_ref
    @printf "  RD residual RMS: %.3e  (%.1f%% of N0_true)\n" sqrt(mean(R_rd.^2)) 100*sqrt(mean(R_rd.^2))/N0_ref
end

# ── Scaling analysis ───────────────────────────────────────────────────────────
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

println()
log_eps = log.(eps_abs_vec)
function loglog_slope(log_x, log_y)
    n = length(log_x)
    (n*sum(log_x.*log_y) - sum(log_x)*sum(log_y)) /
    (n*sum(log_x.^2) - sum(log_x)^2)
end
mc_slope = loglog_slope(log_eps, log.(rms_mc_vec))
rd_slope = loglog_slope(log_eps, log.(rms_rd_vec))

rd_floor  = minimum(rms_rd_vec)
rd_signal = sqrt.(max.(rms_rd_vec.^2 .- rd_floor^2, 0.0))
above_floor = findall(rd_signal .> rd_floor)
rd_slope_sig = length(above_floor) >= 2 ?
    loglog_slope(log_eps[above_floor], log.(rd_signal[above_floor])) : NaN

# With per-map denominator, both N0_MC=<C(SS)> and N0_true=<C(TT)> are eps-independent
# (sc cancels in numer and denom for both), so MC residual should be FLAT (slope ≈ 0).
# This is different from Phase 1 (frozen denom) where MC residual scales as O(eps).
@printf "Log-log slope  MC: %.2f (expect ≈0)    RD (all): %.2f    RD (signal-dominated): %.2f (expect ≈2)\n" mc_slope rd_slope rd_slope_sig
@printf "Noise floor estimate (min RD RMS): %.3e  (%d/%d eps values above floor)\n" rd_floor length(above_floor) length(rms_rd_vec)

println()
mc_flat = abs(mc_slope) < 0.4   # per-map: MC should be flat, not O(eps)
rd_quad = !isnan(rd_slope_sig) && abs(rd_slope_sig - 2.0) < 0.7
if mc_flat && rd_quad
    println("PASS: per-realization denominator GI RDN0 is O(eps²).")
    println("      N0_true is eps-independent (sc cancels in per-map denom) ✓")
    println("      N0_MC is flat (also eps-independent with per-map denom) ✓")
    println("      DS+SD-SS residual scales as O(eps²) in signal-dominated regime ✓")
    println("      Production GI RDN0 (gi_estimate style) should work.")
elseif mc_flat && length(above_floor) < 2
    println("INCONCLUSIVE: RD signal below noise floor — increase nsims or eps range.")
elseif !mc_flat
    println("FAIL: MC residual is not flat — unexpected eps dependence in <C(SS)> or <C(TT)>.")
else
    println("FAIL: RD scaling is not O(eps²) — per-map denominator introduces higher-order terms.")
end

# ── Theory check: verify C(phi_TT) ≈ N0_fid (eps-independent) ────────────────
println("\n── Theory check: N0_true = <C(phi_TT)> vs eps (expect eps-independent) ──")
@printf "%-8s  %-14s  %-10s\n" "eps" "N0_true_mean" "ratio_to_eps0"
N0_true_at_0 = mean(abs.(N0_true_means[eps_values[findfirst(e->abs(e)==0.05, eps_values)]][ℓ_select]))
for eps in eps_values
    n0 = mean(abs.(N0_true_means[eps][ℓ_select]))
    @printf "%-8.3f  %-14.3e  %-10.4f\n" eps n0 n0/N0_true_at_0
end
println("(ratio ≈ 1 at all eps confirms denom scaling cancels the sc factor — eps-independent N0_true)")
