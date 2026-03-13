#!/usr/bin/env julia
# MAP diagnostic: QE vs MAP_joint vs MAP_marg on the full 512×512 map.
# Plots ℓ⁴C_ℓ^{ϕϕ} (auto/cross) and ρ_L for each estimator.
#
# The adaptive secant Hessian in MAP_joint has been fixed in the CMBLensing source:
#   cls.jl       — smooth() now filters NaN/Inf before LOESS
#   maximization.jl — secant block wrapped in try/catch; falls back to static precon on failure
#
# REQUIRES: restart Julia after the source patches so the changes are compiled in.
#
# nburnin_update_hessian:
#   = Inf → always use static prior-based H⁻¹
#   = N   → static for steps 1..N, then switch to adaptive secant H⁻¹(L) ≈ |Δϕ(L)/Δg(L)|
#           With the source fix this no longer crashes — it silently falls back on failure.

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using LinearAlgebra
using Statistics: mean, std
using Printf
using PythonPlot

# ── Serial fallback for MAP_marg ──────────────────────────────────────────────
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset()  = _DIST_DS[]
    end
end

# ── Patch f-preconditioner ────────────────────────────────────────────────────
# H_f^{-1} = (C_f^{-1} + B̂'Cn̂^{-1}B̂)^{-1} has near-zero/NaN entries at high ℓ
# where C_f → 0. Replace bad entries with the mean of the good ones.
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

# ── Parameters ────────────────────────────────────────────────────────────────
const Cℓ            = camb(r=0.05, ℓmax=21000)
const Cℓn           = noiseCℓs(μKarcminT=1.0, ℓknee=0)
const θpix          = 0.7438046267475303
const Nside         = 512
const pol           = :I
const beamFWHM      = 1.0
const bandpass_mask = LowPass(6000)
const Δℓ            = 50
const nsims         = 50
const seed0         = 1000

# nburnin=1: use static only for step 1, then adaptive from step 2 onwards.
# The source fix means failures fall back gracefully (no more BoundsError crash).
const NBURNIN    = 1
const MAPJ_STEPS = 30   # more steps → better convergence with adaptive H
const MAPM_STEPS = 25

println("Nside=$Nside  nsims=$nsims  nburnin=$NBURNIN  MAPJ_STEPS=$MAPJ_STEPS")

# ── Per-sim spectrum storage (for jackknife SEM) ─────────────────────────────
tt_sims  = Vector{Vector{Float64}}()   # true ϕ auto (all sims)
qq_sims  = Vector{Vector{Float64}}()   # QE auto
tq_sims  = Vector{Vector{Float64}}()   # true×QE cross
tt_j_sims = Vector{Vector{Float64}}()  # true auto for sims where joint ran
jj_sims  = Vector{Vector{Float64}}()
tj_sims  = Vector{Vector{Float64}}()
tt_m_sims = Vector{Vector{Float64}}()  # true auto for sims where marg ran
mm_sims  = Vector{Vector{Float64}}()
tm_sims  = Vector{Vector{Float64}}()
ℓ_vec    = nothing
nsims_done = 0

# ── Main loop ─────────────────────────────────────────────────────────────────
for s in 1:nsims
    seed = seed0 + s
    @printf "\n=== Sim %d/%d (seed=%d) ===\n" s nsims seed

    global tt_sims, qq_sims, tq_sims, tt_j_sims, jj_sims, tj_sims
    global tt_m_sims, mm_sims, tm_sims, ℓ_vec, nsims_done

    local ds, ϕ, ϕqe, ϕ_joint, ϕ_marg

    try
        (; ϕ, ds) = load_sim(
            seed=seed, Cℓ=Cℓ, Cℓn=Cℓn,
            θpix=θpix, T=Float64, Nside=Nside,
            beamFWHM=beamFWHM, pol=pol,
            bandpass_mask=bandpass_mask,
            pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )
    catch err; println("  load_sim failed: $err"); continue; end

    # QE (Wiener-filtered) — baseline
    try
        ϕqe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true).ϕqe
    catch err; println("  QE failed: $err"); continue; end

    # MAP_joint with adaptive secant Hessian (falls back gracefully if secant fails)
    ϕ_joint = nothing
    try
        result_j = MAP_joint(
            ds, FieldTuple(ϕ=ϕqe);
            nsteps=MAPJ_STEPS, progress=true,
            nburnin_update_hessian=NBURNIN,
            conjgrad_kwargs=(tol=1e-4, nsteps=200),
            history_keys=(:logpdf, :ΔΩ°_norm),
        )
        ϕ_joint = result_j.ϕ
        logpdfs_j = [h.logpdf for h in result_j.history]
        println("  MAP_joint logpdf: $(round(logpdfs_j[1]; sigdigits=6)) → $(round(logpdfs_j[end]; sigdigits=6))")
        println("  MAP_joint ✓")
    catch err
        println("  MAP_joint failed: $err")
    end

    # MAP_marg
    ϕ_marg = nothing
    try
        ϕ_marg, _ = MAP_marg(
            ds; ϕstart=ϕqe, nsteps=MAPM_STEPS, progress=true,
            α=0.2,
            conjgrad_kwargs=(tol=1e-4, nsteps=200), pmap=map,
        )
        println("  MAP_marg ✓")
    catch err
        println("  MAP_marg failed: $err")
    end

    # ── spectra ───────────────────────────────────────────────────────────────
    cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)
    tt = Float64.(cl_tt.Cℓ)
    if ℓ_vec === nothing; ℓ_vec = Float64.(collect(cl_tt.ℓ)); end

    qq = Float64.(get_Cℓ(ϕqe; Δℓ=Δℓ).Cℓ)
    tq = Float64.(get_Cℓ(ϕ, ϕqe; Δℓ=Δℓ).Cℓ)
    push!(tt_sims, tt); push!(qq_sims, qq); push!(tq_sims, tq)

    if ϕ_joint !== nothing
        jj = Float64.(get_Cℓ(ϕ_joint; Δℓ=Δℓ).Cℓ)
        tj = Float64.(get_Cℓ(ϕ, ϕ_joint; Δℓ=Δℓ).Cℓ)
        push!(tt_j_sims, tt); push!(jj_sims, jj); push!(tj_sims, tj)
    end

    if ϕ_marg !== nothing
        mm = Float64.(get_Cℓ(ϕ_marg; Δℓ=Δℓ).Cℓ)
        tm = Float64.(get_Cℓ(ϕ, ϕ_marg; Δℓ=Δℓ).Cℓ)
        push!(tt_m_sims, tt); push!(mm_sims, mm); push!(tm_sims, tm)
    end

    nsims_done += 1
end

println("\n$nsims_done / $nsims sims completed.")
nsims_done == 0 && error("No sims succeeded.")

# ── ρ_L from mean spectra + jackknife SEM ────────────────────────────────────
function mean_rho(tt_s, rr_s, tr_s)
    n = Float64(length(tt_s))
    Σtt = sum(tt_s); Σrr = sum(rr_s); Σtr = sum(tr_s)
    clamp.((Σtr./n) ./ max.(sqrt.(max.((Σtt./n).*(Σrr./n), 0.0)), 1e-30), -1.0, 1.0)
end

function jk_sem(tt_s, rr_s, tr_s)
    n = length(tt_s); n < 2 && return nothing
    Σtt = sum(tt_s); Σrr = sum(rr_s); Σtr = sum(tr_s)
    ρ_jk = [begin
        n1 = Float64(n - 1)
        Ttt = Σtt .- tt_s[i]; Trr = Σrr .- rr_s[i]; Ttr = Σtr .- tr_s[i]
        clamp.((Ttr./n1) ./ max.(sqrt.(max.((Ttt./n1).*(Trr./n1), 0.0)), 1e-30), -1.0, 1.0)
    end for i in 1:n]
    vec(sqrt((n-1)^2 / n) .* std(reduce(hcat, ρ_jk); dims=2))
end

ρ_q  = mean_rho(tt_sims,   qq_sims, tq_sims)
ρ_j  = !isempty(jj_sims) ? mean_rho(tt_j_sims, jj_sims, tj_sims) : nothing
ρ_m  = !isempty(mm_sims) ? mean_rho(tt_m_sims, mm_sims, tm_sims) : nothing
sem_q = jk_sem(tt_sims,   qq_sims, tq_sims)
sem_j = !isempty(jj_sims) ? jk_sem(tt_j_sims, jj_sims, tj_sims) : nothing
sem_m = !isempty(mm_sims) ? jk_sem(tt_m_sims, mm_sims, tm_sims) : nothing

# means for power spectrum panels
n = Float64(nsims_done)
cltt_mean = sum(tt_sims) ./ n
clrr_q_mean = sum(qq_sims) ./ n
clrr_j_mean = !isempty(jj_sims) ? sum(jj_sims) ./ length(jj_sims) : nothing
clrr_m_mean = !isempty(mm_sims) ? sum(mm_sims) ./ length(mm_sims) : nothing
cltr_q_mean = sum(tq_sims) ./ n
cltr_j_mean = !isempty(tj_sims) ? sum(tj_sims) ./ length(tj_sims) : nothing
cltr_m_mean = !isempty(tm_sims) ? sum(tm_sims) ./ length(tm_sims) : nothing

# ── Plot ──────────────────────────────────────────────────────────────────────
PythonPlot.rc("font", family="serif", size=11)
PythonPlot.rc("xtick", direction="in", top=true)
PythonPlot.rc("ytick", direction="in", right=true)

nrows = ρ_j !== nothing || ρ_m !== nothing ? 3 : 2
fig, axs = PythonPlot.subplots(nrows, 1; figsize=(7, 3.5*nrows), sharex=true, constrained_layout=true)
ax1, ax2 = axs[0], axs[1]
ax3 = nrows == 3 ? axs[2] : nothing

ℓ4 = ℓ_vec .^ 4; ℓ4[ℓ_vec .== 0] .= 0

ax1.loglog(ℓ_vec, ℓ4 .* cltt_mean;     color="k",       label="true",      linewidth=1.5, linestyle="--")
ax1.loglog(ℓ_vec, ℓ4 .* clrr_q_mean;   color="#D62728", label="QE (WF)",   linewidth=1.5)
clrr_j_mean !== nothing && ax1.loglog(ℓ_vec, ℓ4 .* clrr_j_mean; color="#1F77B4", label="MAP joint", linewidth=1.5)
clrr_m_mean !== nothing && ax1.loglog(ℓ_vec, ℓ4 .* clrr_m_mean; color="#2CA02C", label="MAP marg",  linewidth=1.5)
ax1.set_xlim(200, 6000); ax1.set_ylabel(L"\ell^4 C_\ell^{\hat\phi\hat\phi}", fontsize=12)
ax1.legend(frameon=false, fontsize=9); ax1.spines["top"].set_visible(false); ax1.spines["right"].set_visible(false)

ax2.loglog(ℓ_vec, ℓ4 .* abs.(cltr_q_mean); color="#D62728", label="QE",       linewidth=1.5)
cltr_j_mean !== nothing && ax2.loglog(ℓ_vec, ℓ4 .* abs.(cltr_j_mean); color="#1F77B4", label="MAP joint", linewidth=1.5)
cltr_m_mean !== nothing && ax2.loglog(ℓ_vec, ℓ4 .* abs.(cltr_m_mean); color="#2CA02C", label="MAP marg",  linewidth=1.5)
ax2.set_ylabel(L"\ell^4 |C_\ell^{\phi\hat\phi}|", fontsize=12)
ax2.legend(frameon=false, fontsize=9); ax2.spines["top"].set_visible(false); ax2.spines["right"].set_visible(false)

if ax3 !== nothing
    ax3.semilogx(ℓ_vec, ρ_q; color="#D62728", label="QE", linewidth=2.0)
    sem_q !== nothing && ax3.fill_between(ℓ_vec, ρ_q .- sem_q, ρ_q .+ sem_q; color="#D62728", alpha=0.2)
    if ρ_j !== nothing
        ax3.semilogx(ℓ_vec, ρ_j; color="#1F77B4", label="MAP joint", linewidth=2.0)
        sem_j !== nothing && ax3.fill_between(ℓ_vec, ρ_j .- sem_j, ρ_j .+ sem_j; color="#1F77B4", alpha=0.2)
    end
    if ρ_m !== nothing
        ax3.semilogx(ℓ_vec, ρ_m; color="#2CA02C", label="MAP marg", linewidth=2.0)
        sem_m !== nothing && ax3.fill_between(ℓ_vec, ρ_m .- sem_m, ρ_m .+ sem_m; color="#2CA02C", alpha=0.2)
    end
    ax3.axhline(1; color="k", linewidth=0.7, linestyle=":")
    ax3.set_ylim(0, 1.1); ax3.set_ylabel(L"\rho_L", fontsize=12)
    ax3.legend(frameon=false, fontsize=9); ax3.spines["top"].set_visible(false); ax3.spines["right"].set_visible(false)
    ax3.set_xlabel(L"L", fontsize=12)
else
    ax2.set_xlabel(L"L", fontsize=12)
end

ax1.set_title("$nsims_done sims · Nside=$Nside · nburnin=$NBURNIN · MAPJ=$MAPJ_STEPS steps", fontsize=9)

outfile = "results/map_diagnostic_nburnin$(NBURNIN).png"
fig.savefig(outfile; dpi=150)
println("Saved $outfile")
PythonPlot.plotclose("all")
