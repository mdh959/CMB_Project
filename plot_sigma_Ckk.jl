import Pkg; Pkg.activate(@__DIR__)
using JLD2
using Statistics: mean, std
using PythonPlot
using Printf

# plot_sigma_Ckk.jl — σ[C_L^{κ̂κ̂}], σ[C_L^{κκ̂_deb}] vs Knox formula
#
# The checkpoint stores per-sim spectra ALREADY in κ units with debiasing:
#   Cl_auto_*_sims[s]  = C_L^{κ̂_deb, κ̂_deb}(s)  (debiased auto per sim)
#   Cl_cross_*_sims[s] = C_L^{κ_true, κ̂_deb}(s)  (debiased cross per sim)
#
# Empirical σ  = std_sims of per-sim band-averaged power spectrum
# Knox theory  = uses mean debiased cross C̄^{κκ̂} and mean debiased auto C̄^{κ̂κ̂}
#
# Both empirical σ and Knox are computed at f_sky_paper=0.4.
# Empirical σ from patch is rescaled: σ_emp × √(f_sky_patch/f_sky_paper).

# ── Geometry
θpix_rad    = 0.7438 * π / (180 * 60)
f_sky_patch = (512 * θpix_rad)^2 / (4π)   # box solid angle / full sphere
f_sky_paper = 0.4
σ_scale     = sqrt(f_sky_patch / f_sky_paper)
println("f_sky_patch = $(round(f_sky_patch; sigdigits=3))  |  σ_scale = $(round(σ_scale; sigdigits=3))")

# ── Load spectra
ea_file = "results/checkpoints/error_analysis_checkpoint_8000.jld2"
println("Loading $ea_file …")
@load ea_file ℓ_kk Cl_auto_qe_sims Cl_auto_joint_sims Cl_auto_marg_sims
ℓ = Float64.(collect(ℓ_kk))
# Aq[l,s] = C_L^{κ̂_qe_deb, κ̂_qe_deb}(s)   — debiased auto per sim, κ units
Aq, Aj, Am = reduce(hcat, Cl_auto_qe_sims), reduce(hcat, Cl_auto_joint_sims), reduce(hcat, Cl_auto_marg_sims)
println("  $(size(Aq,2)) sims, $(length(ℓ)) bins")

# Xq[l,s] = C_L^{κ_true, κ̂_qe_deb}(s)  — debiased cross per sim (already debiased in error_mean.jl)
# NOTE: do NOT divide by W_L again here; that would double-debias.
Xq, Xj, Xm, have_cross = try
    qd, jd, md = JLD2.load(ea_file, "Cl_cross_qe_sims", "Cl_cross_joint_sims", "Cl_cross_marg_sims")
    println("  Cross loaded ($(length(qd)) sims)")
    reduce(hcat, qd), reduce(hcat, jd), reduce(hcat, md), true
catch; nothing, nothing, nothing, false end

noise_qe_m, noise_joint_m, noise_marg_m, ℓ_noise_vec, have_noise = try
    nq, nj, nm, ln = JLD2.load(ea_file, "noise_qe_sims", "noise_joint_sims", "noise_marg_sims", "ℓ_noise")
    println("  Noise curves loaded")
    mean(reduce(hcat, nq); dims=2)[:], mean(reduce(hcat, nj); dims=2)[:],
    mean(reduce(hcat, nm); dims=2)[:], Float64.(collect(ln)), true
catch; nothing, nothing, nothing, nothing, false end

# ── Coarsen: per-band mean and std across sims
function coarsen(ℓ, M; edges)
    nb = length(edges) - 1
    Lc = fill(NaN, nb); μ_v = fill(NaN, nb); σ_v = fill(NaN, nb)
    for b in 1:nb
        idx = findall(x -> edges[b] <= x < edges[b+1], ℓ)
        isempty(idx) && continue
        per = filter(isfinite, vec(mean(M[idx, :]; dims=1)))
        length(per) < 2 && continue
        Lc[b] = 0.5*(edges[b]+edges[b+1]); μ_v[b] = mean(per); σ_v[b] = std(per)
    end
    return Lc, μ_v, σ_v
end

coarsen_1d(ℓs, y; edges) = [begin
    idx = findall(x -> edges[b] <= x < edges[b+1], ℓs)
    isempty(idx) ? NaN : mean(y[idx])
end for b in 1:length(edges)-1]

ΔL = 500.0; edges = collect(2500 : Int(ΔL) : 10000 + Int(ΔL))

# Auto: C̄_qe = mean over sims of C(κ̂_deb),  σ_qe = std across sims
Lc_qe,    C̄_qe,    σ_qe    = coarsen(ℓ, Aq; edges=edges); σ_qe    .*= σ_scale
Lc_joint, C̄_joint, σ_joint = coarsen(ℓ, Aj; edges=edges); σ_joint .*= σ_scale
Lc_marg,  C̄_marg,  σ_marg  = coarsen(ℓ, Am; edges=edges); σ_marg  .*= σ_scale

# Cross: C̄_xqe = mean over sims of C(κ_true, κ̂_deb),  σ_xqe = std across sims
if have_cross
    Lc_xqe,    C̄_xqe,    σ_xqe    = coarsen(ℓ, Xq; edges=edges); σ_xqe    .*= σ_scale
    Lc_xjoint, C̄_xjoint, σ_xjoint = coarsen(ℓ, Xj; edges=edges); σ_xjoint .*= σ_scale
    Lc_xmarg,  C̄_xmarg,  σ_xmarg  = coarsen(ℓ, Xm; edges=edges); σ_xmarg  .*= σ_scale
end

# C^κκ ≈ mean joint auto − mean joint noise  (both in κ units after kfac conversion)
C_signal_coarse = if have_noise
    Nkk = @. (ℓ_noise_vec^2 / 2)^2 * max(noise_joint_m, 0.0)
    C̄_joint .- coarsen_1d(ℓ_noise_vec, Nkk; edges=edges)
else
    fill(NaN, length(edges)-1)
end

# ── Knox formula at f_sky_paper=0.4
# Auto:  σ²[C_L^{κ̂κ̂}] = 2 / [(2L+1)ΔL f_sky] × (C̄^{κ̂κ̂})²
knox_auto(Lc, C̄) = @. sqrt(2 / ((2Lc + 1) * ΔL * f_sky_paper)) * abs(C̄)
σ_th_qe    = knox_auto(Lc_qe,    C̄_qe)
σ_th_joint = knox_auto(Lc_joint, C̄_joint)
σ_th_marg  = knox_auto(Lc_marg,  C̄_marg)

if have_cross
    # Cross: σ²[C_L^{κκ̂}] = [C^κκ × C̄^{κ̂κ̂} + (C̄^{κκ̂})²] / [(2L+1)ΔL f_sky]
    # C̄^{κκ̂} = mean debiased cross over sims (not the approximation Cs ≈ C^κκ)
    Cs = C_signal_coarse
    σ_th_xqe    = @. sqrt(max(abs(Cs)*abs(C̄_qe)   + C̄_xqe^2,   0.0) / ((2Lc_xqe   +1)*ΔL*f_sky_paper))
    σ_th_xjoint = @. sqrt(max(abs(Cs)*abs(C̄_joint) + C̄_xjoint^2, 0.0) / ((2Lc_xjoint+1)*ΔL*f_sky_paper))
    σ_th_xmarg  = @. sqrt(max(abs(Cs)*abs(C̄_marg)  + C̄_xmarg^2,  0.0) / ((2Lc_xmarg +1)*ΔL*f_sky_paper))
end

# ── Diagnostics
println("\n  ΔL=$(Int(ΔL)), empirical σ rescaled to f_sky=0.4:")
println("  L       σ_auto_QE   σ_auto_MAP  σ_cross_QE  σ_cross_MAP  C̄_cross/C̄_auto  Knox_cross_QE")
for b in eachindex(Lc_qe)
    isnan(Lc_qe[b]) && continue
    sx  = have_cross ? @sprintf("%10.2e  %10.2e", σ_xqe[b], σ_xjoint[b]) : "          —           —"
    rat = have_cross && isfinite(C̄_xqe[b]) && abs(C̄_qe[b]) > 0 ? C̄_xqe[b]/C̄_qe[b] : NaN
    kx  = have_cross && isfinite(σ_th_xqe[b]) ? @sprintf("%10.2e", σ_th_xqe[b]) : "         —"
    @printf "  L=%-5d  %9.2e  %9.2e  %s  ratio=%.3f  %s\n" round(Int,Lc_qe[b]) σ_qe[b] σ_joint[b] sx rat kx
end

# ── Plot setup
PythonPlot.rc("font",        family="serif", size=11)
PythonPlot.rc("axes",        linewidth=0.8)
PythonPlot.rc("xtick",       direction="in", top=true)
PythonPlot.rc("ytick",       direction="in", right=true)
PythonPlot.rc("xtick.major", width=0.8, size=4)
PythonPlot.rc("ytick.major", width=0.8, size=4)
PythonPlot.rc("xtick.minor", width=0.5, size=2.5, visible=true)
PythonPlot.rc("ytick.minor", width=0.5, size=2.5, visible=true)

colours = Dict("qe"=>"#D62728", "joint"=>"#1F77B4", "marg"=>"#2CA02C")
labels  = Dict("qe"=>"QE (WF)", "joint"=>"MAP joint", "marg"=>"MAP marg")

fig, axs_arr = PythonPlot.subplots(3, 1;
    figsize=(6.5, 10.5), sharex=true, constrained_layout=true,
    gridspec_kw=Dict("height_ratios"=>[2, 2, 2], "hspace"=>0.0))
ax_auto  = axs_arr[0]
ax_cross = axs_arr[1]
ax_mean  = axs_arr[2]

# ── Panel 1: σ[C_L^{κ̂κ̂}]
# Solid: empirical std across sims;  Dashed: Knox theory
for (key, Lc, σ_sim, σ_th) in [("qe",Lc_qe,σ_qe,σ_th_qe),
                                  ("joint",Lc_joint,σ_joint,σ_th_joint),
                                  ("marg",Lc_marg,σ_marg,σ_th_marg)]
    msk = @. !isnan(Lc) & !isnan(σ_sim) & (σ_sim > 0)
    !any(msk) && continue
    ax_auto.semilogy(Lc[msk], σ_sim[msk]; color=colours[key], linewidth=2.0, label=labels[key])
    msk_th = @. !isnan(Lc) & !isnan(σ_th) & (σ_th > 0)
    any(msk_th) && ax_auto.semilogy(Lc[msk_th], σ_th[msk_th];
        color=colours[key], linestyle="--", linewidth=1.2, alpha=0.7)
end
ax_auto.set_xlim(5000, 11000); ax_auto.set_ylim(1e-14, 1e-9)
ax_auto.set_ylabel(L"\sigma[C_L^{\hat{\kappa}\hat{\kappa}}]", fontsize=12)
ax_auto.legend(loc="upper left", frameon=false, fontsize=9,
    title="solid=empirical, dashed=Knox", title_fontsize=7)
ax_auto.spines["top"].set_visible(false); ax_auto.spines["right"].set_visible(false)

# ── Panel 2: σ[C_L^{κκ̂_deb}]
# Solid: empirical std(C(κ_true, κ̂_deb)) across sims;  Dashed: Knox theory
if have_cross
    for (key, Lc, σ_sim, σ_th) in [("qe",Lc_xqe,σ_xqe,σ_th_xqe),
                                      ("joint",Lc_xjoint,σ_xjoint,σ_th_xjoint),
                                      ("marg",Lc_xmarg,σ_xmarg,σ_th_xmarg)]
        msk = @. !isnan(Lc) & !isnan(σ_sim) & (σ_sim > 0)
        !any(msk) && continue
        ax_cross.semilogy(Lc[msk], σ_sim[msk]; color=colours[key], linewidth=2.0, label=labels[key])
        msk_th = @. !isnan(Lc) & !isnan(σ_th) & (σ_th > 0)
        any(msk_th) && ax_cross.semilogy(Lc[msk_th], σ_th[msk_th];
            color=colours[key], linestyle="--", linewidth=1.2, alpha=0.7)
    end
    ax_cross.legend(loc="upper left", frameon=false, fontsize=9,
        title="solid=empirical, dashed=Knox", title_fontsize=7)
else
    ax_cross.text(0.5, 0.5, "Cross σ not available", ha="center", va="center",
                  transform=ax_cross.transAxes, fontsize=10, color="grey")
end
ax_cross.set_xlim(3000, 11000); ax_cross.set_ylim(1e-14, 1e-9)
ax_cross.set_ylabel(L"\sigma[C_L^{\kappa\hat{\kappa}_\mathrm{deb}}]", fontsize=12)
ax_cross.spines["top"].set_visible(false); ax_cross.spines["right"].set_visible(false)

# ── Panel 3: mean C(κ_true, κ̂_deb) vs C(κ_true, κ_true) — should be equal if unbiased
if have_cross
    for (key, Lc, C̄_x) in [("qe",Lc_xqe,C̄_xqe),
                              ("joint",Lc_xjoint,C̄_xjoint),
                              ("marg",Lc_xmarg,C̄_xmarg)]
        msk = @. !isnan(Lc) & isfinite(C̄_x) & (C̄_x > 0)
        !any(msk) && continue
        ax_mean.semilogy(Lc[msk], C̄_x[msk]; color=colours[key], linewidth=2.0, label=labels[key])
    end
    # True κ auto: C̄_joint − mean noise (best estimate of C^κκ from sims)
    if have_noise
        Lc_sig = Lc_joint
        msk_s = @. !isnan(Lc_sig) & isfinite(C_signal_coarse) & (C_signal_coarse > 0)
        any(msk_s) && ax_mean.semilogy(Lc_sig[msk_s], C_signal_coarse[msk_s];
            color="k", linestyle="--", linewidth=1.5, label=L"C_L^{\kappa\kappa}\ (\mathrm{true})")
    end
    ax_mean.legend(loc="upper right", frameon=false, fontsize=9)
else
    ax_mean.text(0.5, 0.5, "Cross spectra not available", ha="center", va="center",
                 transform=ax_mean.transAxes, fontsize=10, color="grey")
end
ax_mean.set_xlim(5000, 11000)
ax_mean.set_ylabel(L"\langle C_L^{\kappa\hat{\kappa}_\mathrm{deb}}\rangle", fontsize=12)
ax_mean.set_xlabel(L"L", fontsize=12)
ax_mean.spines["top"].set_visible(false); ax_mean.spines["right"].set_visible(false)

savepath = "results/sigma_Ckk_binned.png"
fig.savefig(savepath; dpi=200)
println("\nSaved $savepath")
PythonPlot.plotclose("all")
