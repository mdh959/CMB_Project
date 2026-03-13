#!/usr/bin/env julia
# test_gi.jl
#
# Compare QE vs Gradient-Inversion (GI) vs MAP_joint vs MAP_marg.
# GI is expected to outperform QE at L > 4000 for SO-like temperature.
# MAP_joint/MAP_marg are initialised from GI (better high-L starting point).
# Reference: Hadzhiyska et al. 2019 (arXiv:1905.04217)

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using Statistics: mean, std
using Printf
using PythonPlot
using JLD2

include("utils.jl")
using .Utils

# ── Serial fallback for MAP_marg ──────────────────────────────────────────────
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset()  = _DIST_DS[]
    end
end

# ── Patch f-preconditioner (same fix as test.jl) ─────────────────────────────
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

# ── Parameters ───────────────────────────────────────────────────────────────
const Cℓ            = camb(r=0.05, ℓmax=21000)
const Cℓn           = noiseCℓs(μKarcminT=1.0, ℓknee=0)
const θpix          = 0.7438046267475303
const Nside         = 512
const pol           = :I
const beamFWHM      = 1.0
const bandpass_mask = LowPass(8000)
const Δℓ            = 100
const nsims         = 100
const seed0         = 2000

const CHECKPOINT = "results/gi_checkpoint.jld2"

const GI_LGRAD   = 2000
const GI_LHP     = 4000
const GI_LMAX    = 18000

const NBURNIN    = 1
const MAPJ_STEPS = 30
const MAPM_STEPS = 25

const load_kwargs = (
    Cℓ=Cℓ, Cℓn=Cℓn, θpix=θpix, T=Float64, Nside=Nside,
    beamFWHM=beamFWHM, pol=pol, bandpass_mask=bandpass_mask,
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

# ── Per-sim spectrum storage (for jackknife SEM) ─────────────────────────────
tt_sims  = Vector{Vector{Float64}}()
qq_sims  = Vector{Vector{Float64}}(); tq_sims  = Vector{Vector{Float64}}()
tt_gi_sims = Vector{Vector{Float64}}(); gg_sims = Vector{Vector{Float64}}(); tg_sims = Vector{Vector{Float64}}()
tt_j_sims  = Vector{Vector{Float64}}(); jj_sims = Vector{Vector{Float64}}(); tj_sims = Vector{Vector{Float64}}()
tt_m_sims  = Vector{Vector{Float64}}(); mm_sims = Vector{Vector{Float64}}(); tm_sims = Vector{Vector{Float64}}()
ℓ_vec    = nothing
nsims_done = 0

# ── Load checkpoint if it exists ─────────────────────────────────────────────
seeds_done = Set{Int}()
if isfile(CHECKPOINT)
    println("Loading checkpoint: $CHECKPOINT")
    cp = load(CHECKPOINT)
    tt_sims    = cp["tt_sims"];    qq_sims  = cp["qq_sims"];  tq_sims  = cp["tq_sims"]
    tt_gi_sims = cp["tt_gi_sims"]; gg_sims  = cp["gg_sims"];  tg_sims  = cp["tg_sims"]
    tt_j_sims  = cp["tt_j_sims"];  jj_sims  = cp["jj_sims"];  tj_sims  = cp["tj_sims"]
    tt_m_sims  = cp["tt_m_sims"];  mm_sims  = cp["mm_sims"];  tm_sims  = cp["tm_sims"]
    ℓ_vec      = cp["ell_vec"]
    seeds_done = Set{Int}(cp["seeds_done"])
    nsims_done = length(tt_sims)
    println("  Resumed: $nsims_done sims already done (seeds: $(sort(collect(seeds_done))))")
end

# ── Main loop ─────────────────────────────────────────────────────────────────
println("=== GI vs QE vs MAP (started from GI)  ($nsims sims) ===")

for s in 1:nsims
    seed = seed0 + s
    seed in seeds_done && continue   # already checkpointed
    @printf "\n--- Sim %d/%d (seed=%d) ---\n" s nsims seed
    global tt_sims, qq_sims, tq_sims, tt_gi_sims, gg_sims, tg_sims
    global tt_j_sims, jj_sims, tj_sims, tt_m_sims, mm_sims, tm_sims
    global ℓ_vec, nsims_done, seeds_done

    local ϕ, ds, ϕqe, ϕgi, ϕ_joint, ϕ_marg

    try
        (; ϕ, ds) = load_sim(; seed=seed, load_kwargs...)
    catch err; println("load_sim failed: $err"); continue; end

    # QE (Wiener-filtered)
    try
        ϕqe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true).ϕqe
    catch err; println("QE failed: $err"); continue; end

    # GI (standalone)
    try
        ϕgi = gi_estimate(ds; Lgrad=GI_LGRAD, Lhp=GI_LHP, Lmax=GI_LMAX)
        println("  GI ✓")
    catch err; println("  GI failed: $err"); ϕgi = nothing; end

    # Hybrid starting point: QE at low-L, GI at high-L
    ϕ_start = if ϕgi !== nothing
        Map(LowPass(GI_LHP) * ϕqe) + Map(HighPass(GI_LHP) * ϕgi)
    else
        ϕqe
    end

    # MAP_joint (started from hybrid)
    ϕ_joint = nothing
    try
        result_j = MAP_joint(
            ds, FieldTuple(ϕ=ϕ_start);
            nsteps=MAPJ_STEPS, progress=true,
            nburnin_update_hessian=NBURNIN,
            conjgrad_kwargs=(tol=1e-4, nsteps=200),
            history_keys=(:logpdf,),
        )
        ϕ_joint = result_j.ϕ
        logpdfs = [h.logpdf for h in result_j.history]
        @printf "  MAP_joint logpdf: %.2f → %.2f\n" logpdfs[1] logpdfs[end]
    catch err; println("  MAP_joint failed: $err"); end

    # MAP_marg (started from hybrid)
    ϕ_marg = nothing
    try
        ϕ_marg, _ = MAP_marg(
            ds; ϕstart=ϕ_start, nsteps=MAPM_STEPS, progress=true,
            conjgrad_kwargs=(tol=1e-4, nsteps=200), pmap=map,
        )
        println("  MAP_marg ✓")
    catch err; println("  MAP_marg failed: $err"); end

    # ── spectra ───────────────────────────────────────────────────────────────
    cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)
    tt    = Float64.(cl_tt.Cℓ)
    if ℓ_vec === nothing; ℓ_vec = Float64.(collect(cl_tt.ℓ)); end

    qq = Float64.(get_Cℓ(ϕqe;    Δℓ=Δℓ).Cℓ)
    tq = Float64.(get_Cℓ(ϕ, ϕqe; Δℓ=Δℓ).Cℓ)
    push!(tt_sims, tt); push!(qq_sims, qq); push!(tq_sims, tq)

    if ϕgi !== nothing
        gg = Float64.(get_Cℓ(ϕgi;    Δℓ=Δℓ).Cℓ)
        tg = Float64.(get_Cℓ(ϕ, ϕgi; Δℓ=Δℓ).Cℓ)
        push!(tt_gi_sims, tt); push!(gg_sims, gg); push!(tg_sims, tg)
    end
    if ϕ_joint !== nothing
        jj = Float64.(get_Cℓ(ϕ_joint;    Δℓ=Δℓ).Cℓ)
        tj = Float64.(get_Cℓ(ϕ, ϕ_joint; Δℓ=Δℓ).Cℓ)
        push!(tt_j_sims, tt); push!(jj_sims, jj); push!(tj_sims, tj)
    end
    if ϕ_marg !== nothing
        mm = Float64.(get_Cℓ(ϕ_marg;    Δℓ=Δℓ).Cℓ)
        tm = Float64.(get_Cℓ(ϕ, ϕ_marg; Δℓ=Δℓ).Cℓ)
        push!(tt_m_sims, tt); push!(mm_sims, mm); push!(tm_sims, tm)
    end

    nsims_done += 1
    push!(seeds_done, seed)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    jldsave(CHECKPOINT;
        tt_sims, qq_sims, tq_sims,
        tt_gi_sims, gg_sims, tg_sims,
        tt_j_sims, jj_sims, tj_sims,
        tt_m_sims, mm_sims, tm_sims,
        ell_vec=ℓ_vec,
        seeds_done=collect(seeds_done),
    )
    println("  Checkpoint saved ($nsims_done sims).")
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

ρ_q  = mean_rho(tt_sims,    qq_sims, tq_sims)
ρ_gi = !isempty(gg_sims) ? mean_rho(tt_gi_sims, gg_sims, tg_sims) : nothing
ρ_j  = !isempty(jj_sims) ? mean_rho(tt_j_sims,  jj_sims, tj_sims) : nothing
ρ_m  = !isempty(mm_sims) ? mean_rho(tt_m_sims,  mm_sims, tm_sims) : nothing

sem_q  = jk_sem(tt_sims,    qq_sims, tq_sims)
sem_gi = !isempty(gg_sims) ? jk_sem(tt_gi_sims, gg_sims, tg_sims) : nothing
sem_j  = !isempty(jj_sims) ? jk_sem(tt_j_sims,  jj_sims, tj_sims) : nothing
sem_m  = !isempty(mm_sims) ? jk_sem(tt_m_sims,  mm_sims, tm_sims) : nothing

# means for power spectrum panels
n = Float64(nsims_done)
cltt_mean   = sum(tt_sims) ./ n
clrr_q_mean = sum(qq_sims) ./ n
cltr_q_mean = sum(tq_sims) ./ n
clrr_gi_mean = !isempty(gg_sims) ? sum(gg_sims) ./ length(gg_sims) : nothing
cltr_gi_mean = !isempty(tg_sims) ? sum(tg_sims) ./ length(tg_sims) : nothing
clrr_j_mean  = !isempty(jj_sims) ? sum(jj_sims) ./ length(jj_sims) : nothing
cltr_j_mean  = !isempty(tj_sims) ? sum(tj_sims) ./ length(tj_sims) : nothing
clrr_m_mean  = !isempty(mm_sims) ? sum(mm_sims) ./ length(mm_sims) : nothing
cltr_m_mean  = !isempty(tm_sims) ? sum(tm_sims) ./ length(tm_sims) : nothing

# ── GI transfer function diagnostics ─────────────────────────────────────────
if cltr_gi_mean !== nothing
    W_GI = cltr_gi_mean ./ max.(cltt_mean, 1e-30)
    println("\nGI transfer function W_L = <C^{true,GI}>/<C^{true,true}>:")
    for ℓ_check in [3000, 4000, 5000, 6000, 8000, 10000]
        idx = argmin(abs.(ℓ_vec .- ℓ_check))
        @printf "  L≈%5d : W_GI=%.4f  ρ_GI=%.4f  ρ_QE=%.4f\n" round(Int, ℓ_vec[idx]) real(W_GI[idx]) ρ_gi[idx] ρ_q[idx]
    end
end

# ── Plot ──────────────────────────────────────────────────────────────────────
PythonPlot.rc("font", family="serif", size=11)
PythonPlot.rc("xtick", direction="in", top=true)
PythonPlot.rc("ytick", direction="in", right=true)

fig, (ax1, ax2) = PythonPlot.subplots(2, 1; figsize=(7, 8), sharex=true, constrained_layout=true)

ℓ4 = ℓ_vec .^ 4

ax1.loglog(ℓ_vec, ℓ4 .* cltt_mean;     color="k",       label="true",                   lw=1.5, ls="--")
ax1.loglog(ℓ_vec, ℓ4 .* clrr_q_mean;   color="#D62728", label="QE (WF)",                lw=1.5)
clrr_j_mean  !== nothing && ax1.loglog(ℓ_vec, ℓ4 .* clrr_j_mean;  color="#1F77B4", label="MAP joint (GI start)",   lw=1.5)
clrr_m_mean  !== nothing && ax1.loglog(ℓ_vec, ℓ4 .* clrr_m_mean;  color="#2CA02C", label="MAP marg (GI start)",    lw=1.5)
ax1.axvline(GI_LHP; color="gray", lw=0.7, ls=":")
ax1.set_xlim(200, GI_LMAX)
ax1.set_ylabel(L"\ell^4 C_\ell^{\hat\phi\hat\phi}", fontsize=12)
ax1.legend(frameon=false, fontsize=8)

ax2.semilogx(ℓ_vec, ρ_q;  color="#D62728", label="QE (WF)", lw=2.0)
sem_q  !== nothing && ax2.fill_between(ℓ_vec, ρ_q  .- sem_q,  ρ_q  .+ sem_q;  color="#D62728", alpha=0.2)
if ρ_j !== nothing
    ax2.semilogx(ℓ_vec, ρ_j;  color="#1F77B4", label="MAP joint", lw=2.0)
    sem_j  !== nothing && ax2.fill_between(ℓ_vec, ρ_j  .- sem_j,  ρ_j  .+ sem_j;  color="#1F77B4", alpha=0.2)
end
if ρ_m !== nothing
    ax2.semilogx(ℓ_vec, ρ_m;  color="#2CA02C", label="MAP marg", lw=2.0)
    sem_m  !== nothing && ax2.fill_between(ℓ_vec, ρ_m  .- sem_m,  ρ_m  .+ sem_m;  color="#2CA02C", alpha=0.2)
end
ax2.axvline(GI_LHP; color="gray", lw=0.7, ls=":", label="GI regime")
ax2.axhline(1; color="k", lw=0.7, ls=":")
ax2.set_ylim(-0.1, 1.1)
ax2.set_ylabel(L"\rho_L", fontsize=12)
ax2.set_xlabel(L"L", fontsize=12)
ax2.legend(frameon=false, fontsize=8)

fig.suptitle("GI vs QE vs MAP — $nsims_done sims — 1µK-arcmin", fontsize=10)

outfile = "results/gi_comparison.png"
fig.savefig(outfile; dpi=150)
println("\nSaved $outfile")
PythonPlot.plotclose("all")
