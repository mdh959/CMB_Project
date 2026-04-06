#!/usr/bin/env julia
# scripts/map_speed_diagnostic.jl
#
# One-sim timing benchmark for MAP_joint at 0.1 µK-arcmin.
# Tests three configurations in sequence:
#
#   A. Baseline  : QE warm-start  + Cf  preconditioner   (current state of test_gi.jl)
#   B. Warm-start: GI warm-start  + Cf  preconditioner   (better initialisation only)
#   C. Full fix  : GI warm-start  + Cf̃  preconditioner   (Phase 2 of plan)
#
# Prints per-outer-step CG iteration counts and a summary timing table.
#
# ML note: the Phase-4 learned preconditioner slot is shown as a placeholder at the
# bottom of the file.  Without training data it cannot be tested, but the hook into
# CMBLensing's Hessian_logpdf_preconditioner is shown so you know where it lives.

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))

using CMBLensing
using Statistics: mean
using Printf
using JLD2

include(joinpath(@__DIR__, "..", "utils.jl"))
using .Utils

# ── Serial fallback (needed for MAP_marg if running without workers) ──────────
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset()  = _DIST_DS[]
    end
end

# ── Parameters ────────────────────────────────────────────────────────────────
const SEED        = 9001          # single diagnostic seed
const μKarcminT   = 0.1
const beamFWHM    = 0.3
const Lmax        = 12000         # keep manageable for a diagnostic run; use 20000 for production
const MAPJ_STEPS  = 20
const NBURNIN     = 1             # steps before quasi-Newton Hessian kicks in
const CG_TOL      = 1e-3
const CG_NSTEPS   = 300

const GI_LGRAD    = 2000
const GI_LHP      = 4000
const GI_LMAX     = Lmax

const Cℓ = camb(r=0.05, ℓmax=21000)
const Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)

const load_kwargs = (
    Cℓ=Cℓ, Cℓn=Cℓn, θpix=0.7438046267475303, T=Float64, Nside=512,
    beamFWHM=beamFWHM, pol=:I,
    bandpass_mask=LowPass(Lmax),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

# ── Helpers ───────────────────────────────────────────────────────────────────

# Patch preconditioner to use Cf (unlensed) — current state
function use_Cf_preconditioner!()
    @eval CMBLensing begin
        function Hessian_logpdf_preconditioner(::Val{:f}, ds::DataSet)
            @unpack Cf, B̂, M̂, Cn̂ = ds
            v   = copy(diag(pinv(Cf) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂))
            arr = v.arr
            good = isfinite.(arr) .& (arr .> 0)
            fill!(view(arr, .!good), any(good) ? mean(arr[good]) : 1.0)
            arr .= clamp.(arr, 1e-30, 1e30)
            return Diagonal(v)
        end
    end
end

# Patch preconditioner to use Cf̃ (lensed) — Phase 2 fix
function use_Cftilde_preconditioner!()
    @eval CMBLensing begin
        function Hessian_logpdf_preconditioner(::Val{:f}, ds::DataSet)
            @unpack Cf̃, B̂, M̂, Cn̂ = ds   # ← only change: Cf → Cf̃
            v   = copy(diag(pinv(Cf̃) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂))
            arr = v.arr
            good = isfinite.(arr) .& (arr .> 0)
            fill!(view(arr, .!good), any(good) ? mean(arr[good]) : 1.0)
            arr .= clamp.(arr, 1e-30, 1e30)
            return Diagonal(v)
        end
    end
end

# Extract per-step CG iteration counts from MAP history
function cg_iters_per_step(history)
    [length(h.argmaxf_logpdf_history) for h in history]
end

# Run MAP_joint and return (result, wall_time_seconds)
function run_map(ds, ϕstart; label)
    println("\n  → Running MAP_joint ($label)...")
    t = @elapsed result = MAP_joint(
        ds, FieldTuple(ϕ=ϕstart);
        nsteps         = MAPJ_STEPS,
        nburnin_update_hessian = NBURNIN,
        conjgrad_kwargs = (tol=CG_TOL, nsteps=CG_NSTEPS),
        history_keys   = (:logpdf, :argmaxf_logpdf_history),
        progress       = false,
    )
    return result, t
end

# Print per-step table
function print_step_table(result, wall_time; label)
    iters  = cg_iters_per_step(result.history)
    logpds = [h.logpdf for h in result.history]
    println()
    println("  ── $label ──")
    @printf "  %-5s  %-8s  %-12s\n" "step" "CG_iters" "logpdf"
    for (k, (it, lp)) in enumerate(zip(iters, logpds))
        @printf "  %-5d  %-8d  %-12.2f\n" k it lp
    end
    @printf "  Total CG iters: %d  |  Total CG iters / step: %.1f  |  Wall time: %.1f s\n" \
        sum(iters) mean(iters) wall_time
end

# ── Load sim ──────────────────────────────────────────────────────────────────
println("="^65)
println("MAP speed diagnostic  —  0.1 µK-arcmin  —  seed=$SEED")
println("="^65)

println("\nLoading sim (seed=$SEED)...")
(; ϕ, ds) = load_sim(; seed=SEED, load_kwargs...)

println("Computing QE warm-start...")
ϕqe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true).ϕqe

println("Computing GI warm-start...")
ϕgi = gi_estimate(ds; Lgrad=GI_LGRAD, Lhp=GI_LHP, Lmax=GI_LMAX)

# Hybrid: QE at large scales, GI at small scales
ϕ_hybrid = Map(LowPass(GI_LHP) * ϕqe) + Map(HighPass(GI_LHP) * ϕgi)

# ── Configuration A: QE start + Cf preconditioner ────────────────────────────
use_Cf_preconditioner!()
res_A, t_A = run_map(ds, ϕqe; label="A: QE start + Cf precond")
print_step_table(res_A, t_A; label="A: QE start + Cf precond (baseline)")

# ── Configuration B: GI hybrid start + Cf preconditioner ─────────────────────
use_Cf_preconditioner!()   # same preconditioner, different warm start
res_B, t_B = run_map(ds, ϕ_hybrid; label="B: GI-hybrid start + Cf precond")
print_step_table(res_B, t_B; label="B: GI-hybrid start + Cf precond")

# ── Configuration C: GI hybrid start + Cf̃ preconditioner ─────────────────────
use_Cftilde_preconditioner!()
res_C, t_C = run_map(ds, ϕ_hybrid; label="C: GI-hybrid start + Cf̃ precond")
print_step_table(res_C, t_C; label="C: GI-hybrid start + Cf̃ precond (Phase 2)")

# ── Summary table ─────────────────────────────────────────────────────────────
println()
println("="^65)
println("SUMMARY  (0.1 µK-arcmin, Lmax=$Lmax, CG tol=$CG_TOL, max $CG_NSTEPS iters)")
println("="^65)
@printf "%-45s  %8s  %8s  %8s\n" "Configuration" "Wall(s)" "ΣCG" "CG/step"
for (label, result, t) in [
    ("A: QE start + Cf precond (baseline)", res_A, t_A),
    ("B: GI-hybrid start + Cf precond",     res_B, t_B),
    ("C: GI-hybrid start + Cf̃ precond",     res_C, t_C),
]
    iters = cg_iters_per_step(result.history)
    @printf "%-45s  %8.1f  %8d  %8.1f\n" label t sum(iters) mean(iters)
end

speedup_BC = t_A / t_C
@printf "\nSpeedup A→C (warm-start + Cf̃): %.2f×\n" speedup_BC

# ── Final logpdf check: all three should converge to same value ───────────────
println()
println("Convergence sanity check (final logpdf, should agree to < 1%):")
lp_A = res_A.history[end].logpdf
lp_B = res_B.history[end].logpdf
lp_C = res_C.history[end].logpdf
@printf "  A: %.4f\n  B: %.4f\n  C: %.4f\n" lp_A lp_B lp_C
println("(large differences → not yet converged; increase MAPJ_STEPS)")

# ── phi reconstruction quality (cross-correlation with truth) ─────────────────
println()
println("Reconstruction quality (ρ_L = cross-corr with truth):")
for (lbl, ϕhat) in [("QE start",  ϕqe),
                     ("GI hybrid", ϕ_hybrid),
                     ("MAP A",     res_A.ϕ),
                     ("MAP B",     res_B.ϕ),
                     ("MAP C",     res_C.ϕ)]
    cl_tt = get_Cℓ(ϕ;       Δℓ=200)
    cl_rr = get_Cℓ(ϕhat;    Δℓ=200)
    cl_tr = get_Cℓ(ϕ, ϕhat; Δℓ=200)
    ℓ     = Float64.(collect(cl_tt.ℓ))
    ρ     = Float64.(cl_tr.Cℓ) ./ sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_rr.Cℓ), 0))
    @printf "  %-12s:" lbl
    for ℓ_check in [3000, 5000, 7000, 9000]
        idx = argmin(abs.(ℓ .- ℓ_check))
        @printf "  ρ(L≈%d)=%.3f" ℓ_check ρ[idx]
    end
    println()
end

println()
println("Done. To test ML preconditioner: see Phase-4 stub below (needs training).")

# ════════════════════════════════════════════════════════════════════════════════
# Phase 4 stub: where a learned preconditioner would plug in
# ════════════════════════════════════════════════════════════════════════════════
#
# The Hessian_logpdf_preconditioner hook is the right place.
# A minimal Lux.jl architecture that approximates H_f^{-1}:
#
#   using Lux, Random
#   struct LearnedPrecond <: Lux.AbstractExplicitLayer
#       # small real-space conv: (Ny,Nx,1,1) input/output
#       conv::Lux.Conv
#   end
#   LearnedPrecond(k=5) = LearnedPrecond(
#       Lux.Chain(
#           Lux.Conv((k,k), 1=>8, relu; pad=k÷2),
#           Lux.Conv((k,k), 8=>1;       pad=k÷2),
#       )
#   )
#
# Training data for Phase 4:
#   For each ϕ sample from QE:
#     Draw random probe v ~ N(0,I)
#     Compute w = H_f^{-1} v  via CG (cheap: same CG you already run)
#     Training pair: (v, w)
#
# At inference: replace diagonal M*r with network(r)
# This never changes the MAP solution — only affects CG convergence speed.
#
# To plug in:
#   function use_learned_preconditioner!(ps, st, model)
#       @eval CMBLensing begin
#           function Hessian_logpdf_preconditioner(::Val{:f}, ds::DataSet)
#               # ps, st, model captured from outer scope via closure
#               f_ref = diag(ds.Cf)
#               FuncOp(r -> begin
#                   arr = r.arr[:,:,1:1]          # (Ny,Nx,1) view
#                   out, _ = model(arr, ps, st)
#                   typeof(r)(out[:,:,1], r.metadata)
#               end)
#           end
#       end
#   end
#
# For warm-start prediction (alternative to preconditioner):
#   Train a UNet: T_obs (Ny,Nx) → ϕ_hat (Ny,Nx)
#   Use ϕ_hat as Ωstart instead of ϕqe.
#   This requires (T_obs, ϕ_true) pairs — you have 1000 from run_qe_gi_wl12k.
#   But note: the stored ϕ_true maps are the right targets; T_obs = Map(ds.d).arr.
#
# Decision between the two:
#   Learned preconditioner → speeds up each outer step (fewer CG iters)
#   Learned warm-start     → reduces number of outer steps needed
#   Both are compatible and stack.
#
# Recommendation: run this diagnostic first. If CG iters/step are the bottleneck,
# do learned preconditioner. If outer step count is the bottleneck, do warm-start.
