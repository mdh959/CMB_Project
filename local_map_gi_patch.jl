#!/usr/bin/env julia
# local_map_gi_patch.jl
#
# Local MAP/GI diagnostics:
#   A. Local convergence: ρ(GI,true), ρ(MAP,true), ρ(MAP,GI)
#   B. Gradient weighting: high-|∇T_LP| vs low-|∇T_LP| patches
#   C. Patch-level cross/auto σ
#   D. Oracle local-amplitude MAP test: 4×4 smooth patch amplitudes fitted to true high-L κ
#
# Uses CMBLensing.m_rfft/m_irfft only. No FFTW/AbstractFFTs.

import Pkg
Pkg.activate(@__DIR__)

using CMBLensing, Statistics, JLD2, PythonPlot, Printf, LinearAlgebra
import CMBLensing: m_rfft, m_irfft

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

const θpix     = 0.7438046267475303
const θpix_rad = θpix * π / (180 * 60)
const Nside    = 512
const minW     = 1e-8

const QEGI_FILE = "results/phi_maps_qe_gi_12000_ul.jld2"
const MAP_FILE  = "results/phi_maps_map_12000_ul.jld2"
const WL_QEGI   = "results/WL_qe_gi_12000_ul.jld2"
const WL_MAP    = "results/WL_map_12000_ul.jld2"
const OUT_DIR   = "results/local_patch"
mkpath(OUT_DIR)

const DO_GRADIENT_PATCHES = true
const DO_ORACLE_LOCAL_AMP = true

const Np       = 256
const ALPHA    = 0.2
const ℓ_edges  = collect(3000.0:500.0:12500.0)
const nb       = length(ℓ_edges) - 1

const AMP_GRID = 4
const AMP_MIN  = 0.5
const AMP_MAX  = 1.5

# ─────────────────────────────────────────────────────────────────────────────
# FFT / map helpers
# ─────────────────────────────────────────────────────────────────────────────

function smooth_wl(W::Vector{Float64}; window::Int=9)
    Ws = similar(W)
    hw = window ÷ 2
    for i in eachindex(W)
        vals = filter(isfinite, W[max(firstindex(W), i-hw):min(lastindex(W), i+hw)])
        Ws[i] = isempty(vals) ? 0.0 : mean(vals)
    end
    return Ws
end

function tukey1d(N::Int, α::Float64=0.2)
    w = ones(Float64, N)
    width = max(1, floor(Int, α * N / 2))
    for i in 1:width
        c = 0.5 * (1 - cos(π * (i - 1) / width))
        w[i] = c
        w[N + 1 - i] = c
    end
    return w
end

tukey2d(N::Int, α::Float64=0.2) = tukey1d(N, α) .* tukey1d(N, α)'

function fftfreq_ints(N::Int)
    iseven(N) ? vcat(0:(N ÷ 2), (-(N ÷ 2 - 1)):-1) :
                vcat(0:((N - 1) ÷ 2), (-((N - 1) ÷ 2)):-1)
end

function make_ellxy_grid_rfft(N::Int, θpix_rad::Float64)
    NyF  = N ÷ 2 + 1
    Lbox = N * θpix_rad

    ℓy = 2π .* collect(0:(NyF - 1)) ./ Lbox
    ℓx = 2π .* fftfreq_ints(N) ./ Lbox

    ℓx2D = repeat(ℓx', NyF, 1)
    ℓy2D = repeat(ℓy, 1, N)
    ℓ2D  = sqrt.(ℓx2D.^2 .+ ℓy2D.^2)

    return ℓx2D, ℓy2D, ℓ2D
end

make_ell_grid_rfft(N::Int, θpix_rad::Float64) = make_ellxy_grid_rfft(N, θpix_rad)[3]

function rfft_weights(N::Int)
    NyF = N ÷ 2 + 1
    w = ones(Float64, NyF, N)
    iseven(N) ? (w[2:end-1, :] .*= 2.0) : (w[2:end, :] .*= 2.0)
    return w
end

function debias_arr(arr::Matrix{Float64}, WL, ℓ2d::Matrix{Float64}; minW_local=minW)
    Ny = size(arr, 1)
    F = m_rfft(arr, (1, 2))

    ℓmin = minimum(WL.ℓ)
    ℓmax = maximum(WL.ℓ)

    for j in axes(F, 2), i in axes(F, 1)
        L = ℓ2d[i, j]
        if L < ℓmin || L > ℓmax
            F[i, j] = 0.0 + 0.0im
        else
            w = Float64(WL(L))
            F[i, j] = (isfinite(w) && abs(w) > minW_local) ? F[i, j] / w : 0.0 + 0.0im
        end
    end

    return real.(m_irfft(F, Ny, (1, 2)))
end

function bandpass_kappa_arr(phi::Matrix{Float64}, ℓ2d::Matrix{Float64};
                            Llo::Float64=4000.0, Lhi::Float64=12000.0)
    Ny = size(phi, 1)
    F = m_rfft(phi, (1, 2))

    for j in axes(F, 2), i in axes(F, 1)
        L = ℓ2d[i, j]
        if Llo <= L <= Lhi
            F[i, j] *= -0.5 * L^2
        else
            F[i, j] = 0.0 + 0.0im
        end
    end

    return real.(m_irfft(F, Ny, (1, 2)))
end

function cross_spec(a::Matrix, b::Matrix, W2d::Matrix, ℓ2d::Matrix)
    N  = size(a, 1)
    Fa = m_rfft(W2d .* a, (1, 2)) .* θpix_rad^2
    Fb = m_rfft(W2d .* b, (1, 2)) .* θpix_rad^2

    Ωw  = sum(W2d.^2) * θpix_rad^2
    wt  = rfft_weights(N)
    C2d = real.(conj.(Fa) .* Fb) ./ Ωw

    Lc = fill(NaN, nb)
    Cl = fill(NaN, nb)

    for bi in 1:nb
        mask = (ℓ2d .>= ℓ_edges[bi]) .& (ℓ2d .< ℓ_edges[bi+1])
        sum(mask) < 2 && continue

        ww = wt[mask]
        Lc[bi] = sum(ww .* ℓ2d[mask]) / sum(ww)
        Cl[bi] = sum(ww .* C2d[mask]) / sum(ww)
    end

    return Lc, Cl
end

# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

function init_specs(nsim::Int, nview::Int)
    return (
        tt = fill(NaN, nb, nsim, nview),
        tg = fill(NaN, nb, nsim, nview),
        tm = fill(NaN, nb, nsim, nview),
        gg = fill(NaN, nb, nsim, nview),
        mm = fill(NaN, nb, nsim, nview),
        gm = fill(NaN, nb, nsim, nview),
    )
end

function fill_view!(S, si, vi, ta, ga, ma, W2d, ℓ2d)
    S.tt[:, si, vi] .= cross_spec(ta, ta, W2d, ℓ2d)[2]
    S.tg[:, si, vi] .= cross_spec(ta, ga, W2d, ℓ2d)[2]
    S.tm[:, si, vi] .= cross_spec(ta, ma, W2d, ℓ2d)[2]
    S.gg[:, si, vi] .= cross_spec(ga, ga, W2d, ℓ2d)[2]
    S.mm[:, si, vi] .= cross_spec(ma, ma, W2d, ℓ2d)[2]
    S.gm[:, si, vi] .= cross_spec(ga, ma, W2d, ℓ2d)[2]
end

function rho_stats(Cab, Caa, Cbb)
    nb_, ns, nv = size(Cab)
    μ = fill(NaN, nb_, nv)
    σ = fill(NaN, nb_, nv)

    for vi in 1:nv, bi in 1:nb_
        vals = Float64[]
        for si in 1:ns
            denom = sqrt(max(Caa[bi, si, vi], 0.0) * max(Cbb[bi, si, vi], 0.0))
            if denom > 0 && isfinite(Cab[bi, si, vi])
                push!(vals, clamp(Cab[bi, si, vi] / denom, -1.0, 1.0))
            end
        end
        length(vals) < 2 && continue
        μ[bi, vi] = mean(vals)
        σ[bi, vi] = std(vals)
    end

    return μ, σ
end

function cl_std(C)
    nb_, ns, nv = size(C)
    σ = fill(NaN, nb_, nv)

    for vi in 1:nv, bi in 1:nb_
        vals = filter(isfinite, C[bi, :, vi])
        length(vals) < 2 && continue
        σ[bi, vi] = std(vals)
    end

    return σ
end

function compute_bundle_stats(S)
    return (
        ρ_tg = rho_stats(S.tg, S.tt, S.gg),
        ρ_tm = rho_stats(S.tm, S.tt, S.mm),
        ρ_gm = rho_stats(S.gm, S.gg, S.mm),
        σ_tg = cl_std(S.tg),
        σ_tm = cl_std(S.tm),
        σ_gg = cl_std(S.gg),
        σ_mm = cl_std(S.mm),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Gradient-patch helpers
# ─────────────────────────────────────────────────────────────────────────────

const Cℓ_CACHE  = Ref{Any}(nothing)
const Cℓn_CACHE = Ref{Any}(nothing)

function get_cached_cls()
    if Cℓ_CACHE[] === nothing
        Cℓ_CACHE[] = camb(r=0.05, ℓmax=35000)
    end
    if Cℓn_CACHE[] === nothing
        Cℓn_CACHE[] = noiseCℓs(μKarcminT=0.1, ℓknee=0, ℓmax=12000)
    end
    return Cℓ_CACHE[], Cℓn_CACHE[]
end

function try_load_ds_for_gradient(seed::Int)
    Cℓ, Cℓn = get_cached_cls()

    r = load_sim(
        seed=seed,
        Cℓ=Cℓ,
        Cℓn=Cℓn,
        θpix=θpix,
        T=Float64,
        Nside=Nside,
        beamFWHM=0.3,
        pol=:I,
        bandpass_mask=LowPass(12000),
        pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
    )

    return r.ds
end

function grad_strength_from_data(d)
    arr = Float64.(Map(LowPass(2000) * d).arr)
    N = size(arr, 1)

    ℓx2D, ℓy2D, _ = make_ellxy_grid_rfft(N, θpix_rad)
    F = m_rfft(arr, (1, 2))

    gx = real.(m_irfft((im .* ℓx2D) .* F, N, (1, 2)))
    gy = real.(m_irfft((im .* ℓy2D) .* F, N, (1, 2)))

    return gx.^2 .+ gy.^2
end

function choose_gradient_patches(g2::Matrix{Float64}, Np::Int; stride::Int=64)
    N = size(g2, 1)
    starts = collect(1:stride:(N - Np + 1))

    best_score, worst_score = -Inf, Inf
    best_patch, worst_patch = nothing, nothing

    for r0 in starts, c0 in starts
        rows = r0:(r0 + Np - 1)
        cols = c0:(c0 + Np - 1)
        score = mean(@view g2[rows, cols])

        if score > best_score
            best_score = score
            best_patch = ("high |∇TLP|", rows, cols)
        end
        if score < worst_score
            worst_score = score
            worst_patch = ("low |∇TLP|", rows, cols)
        end
    end

    return best_patch, worst_patch, best_score, worst_score
end

# ─────────────────────────────────────────────────────────────────────────────
# Oracle local-amplitude MAP helpers
# ─────────────────────────────────────────────────────────────────────────────

function make_smooth_amp_windows(N::Int, ngrid::Int; σ_frac::Float64=0.65)
    tile = N / ngrid
    σ = σ_frac * tile

    y = collect(1:N)
    x = collect(1:N)
    Y = repeat(y, 1, N)
    X = repeat(x', N, 1)

    wins = Matrix{Float64}[]

    for iy in 1:ngrid, ix in 1:ngrid
        cy = (iy - 0.5) * tile
        cx = (ix - 0.5) * tile
        push!(wins, @. exp(-0.5 * (((Y - cy) / σ)^2 + ((X - cx) / σ)^2)))
    end

    Wsum = zeros(Float64, N, N)
    for W in wins
        Wsum .+= W
    end
    Wsum .= max.(Wsum, 1e-30)

    for i in eachindex(wins)
        wins[i] = wins[i] ./ Wsum
    end

    return wins
end

function fit_oracle_local_amplitudes(ktrue::Matrix{Float64},
                                     kmap::Matrix{Float64},
                                     wins::Vector{Matrix{Float64}};
                                     amin::Float64=0.5,
                                     amax::Float64=1.5,
                                     ridge::Float64=1e-10)

    npar = length(wins)
    M = zeros(Float64, npar, npar)
    y = zeros(Float64, npar)
    basis = Vector{Matrix{Float64}}(undef, npar)

    for p in 1:npar
        basis[p] = wins[p] .* kmap
        y[p] = sum(basis[p] .* ktrue)
    end

    for p in 1:npar, q in p:npar
        v = sum(basis[p] .* basis[q])
        M[p, q] = v
        M[q, p] = v
    end

    scale = maximum(abs.(diag(M)))
    scale = isfinite(scale) && scale > 0 ? scale : 1.0

    A = (M + ridge * scale * I) \ y
    A = clamp.(A, amin, amax)

    Amap = zeros(Float64, size(kmap))
    for p in 1:npar
        Amap .+= A[p] .* wins[p]
    end

    return A, Amap
end

# ─────────────────────────────────────────────────────────────────────────────
# Load W_L and geometry
# ─────────────────────────────────────────────────────────────────────────────

println("Loading W_L files...")
d_qegi = JLD2.load(WL_QEGI)
d_map  = JLD2.load(WL_MAP)

WL_gi = Cℓs(Float64.(d_qegi["ℓ_template"]), smooth_wl(Float64.(d_qegi["W_gi_b"])))
WL_mj = Cℓs(Float64.(d_map["ℓ_template"]),  smooth_wl(Float64.(d_map["W_mj"])))

ℓ2d_patch = make_ell_grid_rfft(Np, θpix_rad)
ℓ2d_full  = make_ell_grid_rfft(Nside, θpix_rad)
W2d_patch = tukey2d(Np, ALPHA)
W2d_full  = tukey2d(Nside, ALPHA)

amp_windows = make_smooth_amp_windows(Nside, AMP_GRID)
n_amp = length(amp_windows)

Lc_ref = [0.5 * (ℓ_edges[b] + ℓ_edges[b+1]) for b in 1:nb]
kfac   = @. (Lc_ref^2 / 2)^2

quadrant_patches = [
    ("NW", 1:Np,     1:Np),
    ("NE", 1:Np,     Np+1:2Np),
    ("SW", Np+1:2Np, 1:Np),
    ("SE", Np+1:2Np, Np+1:2Np),
]

quad_labels = [p[1] for p in quadrant_patches]
push!(quad_labels, "full map")

grad_labels = ["high |∇TLP|", "low |∇TLP|", "full map"]
amp_labels  = ["MAP + oracle local amp"]

println("Patch size: $(Np)×$(Np) px = $(round(Np * θpix / 60; sigdigits=3))°")
println("Oracle local amplitude grid: $(AMP_GRID)×$(AMP_GRID) = $n_amp amplitudes")

# ─────────────────────────────────────────────────────────────────────────────
# Find shared sims
# ─────────────────────────────────────────────────────────────────────────────

function sim_ids(file)
    ids = Int[]
    jldopen(file, "r") do f
        for k in keys(f)
            m = match(r"^sim_(\d+)$", k)
            m !== nothing && push!(ids, parse(Int, m.captures[1]))
        end
    end
    return sort(ids)
end

shared = sort(collect(intersect(Set(sim_ids(QEGI_FILE)), Set(sim_ids(MAP_FILE)))))

jldopen(MAP_FILE, "r") do f
    filter!(s -> haskey(f, "sim_$s/ϕ_mj"), shared)
end

Nsim = length(shared)
@assert Nsim > 0 "No shared sims with ϕ_mj found."
println("Shared sims with ϕ_mj: $Nsim")

# ─────────────────────────────────────────────────────────────────────────────
# Process sims
# ─────────────────────────────────────────────────────────────────────────────

quad = init_specs(Nsim, length(quad_labels))
grad = init_specs(Nsim, length(grad_labels))
amp  = init_specs(Nsim, 1)

grad_ok = falses(Nsim)
grad_scores_high = fill(NaN, Nsim)
grad_scores_low  = fill(NaN, Nsim)

amp_ok = falses(Nsim)
amp_A  = fill(NaN, Nsim, n_amp)

gradient_enabled = DO_GRADIENT_PATCHES
warned_gradient = false

println("Processing sims...")
jldopen(QEGI_FILE, "r") do fqegi
    jldopen(MAP_FILE, "r") do fmap
        for (si, s) in enumerate(shared)
            print("\r  sim $si/$Nsim")
            flush(stdout)

            t = Float64.(read(fqegi, "sim_$s/ϕ_true"))
            g = Float64.(read(fqegi, "sim_$s/ϕ_gi_b"))
            m = Float64.(read(fmap,  "sim_$s/ϕ_mj"))

            seed = haskey(fqegi, "sim_$s/seed") ? read(fqegi, "sim_$s/seed") : 1000 + s

            g_deb = debias_arr(g, WL_gi, ℓ2d_full; minW_local=minW)
            m_deb = debias_arr(m, WL_mj, ℓ2d_full; minW_local=minW)

            # A/C: fixed quadrant local convergence and σ.
            for (vi, (_, rows, cols)) in enumerate(quadrant_patches)
                fill_view!(quad, si, vi, t[rows, cols], g_deb[rows, cols], m_deb[rows, cols],
                           W2d_patch, ℓ2d_patch)
            end
            fill_view!(quad, si, length(quad_labels), t, g_deb, m_deb, W2d_full, ℓ2d_full)

            # B: gradient-selected local weighting diagnostic.
            if gradient_enabled
                try
                    ds = try_load_ds_for_gradient(seed)
                    g2 = grad_strength_from_data(ds.d)
                    high_patch, low_patch, high_score, low_score =
                        choose_gradient_patches(g2, Np; stride=64)

                    grad_ok[si] = true
                    grad_scores_high[si] = high_score
                    grad_scores_low[si]  = low_score

                    for (vi, patch) in enumerate([high_patch, low_patch])
                        _, rows, cols = patch
                        fill_view!(grad, si, vi, t[rows, cols], g_deb[rows, cols], m_deb[rows, cols],
                                   W2d_patch, ℓ2d_patch)
                    end

                    fill_view!(grad, si, length(grad_labels), t, g_deb, m_deb, W2d_full, ℓ2d_full)
                catch err
                    gradient_enabled = false
                    if !warned_gradient
                        println()
                        @warn "Gradient-selected patches disabled after failure. Quadrant/oracle plots still produced." err
                        warned_gradient = true
                    end
                end
            end

            # D: oracle 4×4 local amplitude correction for MAP.
            if DO_ORACLE_LOCAL_AMP
                kt = bandpass_kappa_arr(t,     ℓ2d_full; Llo=4000.0, Lhi=12000.0)
                km = bandpass_kappa_arr(m_deb, ℓ2d_full; Llo=4000.0, Lhi=12000.0)

                A_loc, Amap = fit_oracle_local_amplitudes(
                    kt, km, amp_windows;
                    amin=AMP_MIN, amax=AMP_MAX,
                )

                amp_A[si, :] .= A_loc
                amp_ok[si] = true

                # Smooth local-amplitude correction applied to MAP φ field.
                # The fit is done on high-L κ, so this is an oracle diagnostic only.
                m_amp_deb = Amap .* m_deb

                fill_view!(amp, si, 1, t, g_deb, m_amp_deb, W2d_full, ℓ2d_full)
            end
        end
    end
end
println("\nDone.")

quad_stats = compute_bundle_stats(quad)

Ngrad = count(grad_ok)
grad_stats = nothing
if Ngrad >= 2
    grad_sub = (
        tt = grad.tt[:, grad_ok, :],
        tg = grad.tg[:, grad_ok, :],
        tm = grad.tm[:, grad_ok, :],
        gg = grad.gg[:, grad_ok, :],
        mm = grad.mm[:, grad_ok, :],
        gm = grad.gm[:, grad_ok, :],
    )
    grad_stats = compute_bundle_stats(grad_sub)
else
    println("Gradient-selected diagnostic unavailable: Ngrad=$Ngrad")
end

Namp = count(amp_ok)
amp_stats = nothing
if Namp >= 2
    amp_sub = (
        tt = amp.tt[:, amp_ok, :],
        tg = amp.tg[:, amp_ok, :],
        tm = amp.tm[:, amp_ok, :],
        gg = amp.gg[:, amp_ok, :],
        mm = amp.mm[:, amp_ok, :],
        gm = amp.gm[:, amp_ok, :],
    )
    amp_stats = compute_bundle_stats(amp_sub)

    valsA = filter(isfinite, vec(amp_A[amp_ok, :]))
    println("Oracle local-amplitude MAP diagnostic: Namp=$Namp")
    @printf "  A mean/min/max = %.3f / %.3f / %.3f\n" mean(valsA) minimum(valsA) maximum(valsA)
else
    println("Oracle local-amplitude diagnostic unavailable: Namp=$Namp")
end

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

PythonPlot.rc("font", family="serif", size=10)
PythonPlot.rc("axes", linewidth=0.8)
PythonPlot.rc("xtick", direction="in", top=true)
PythonPlot.rc("ytick", direction="in", right=true)

const quad_cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "k"]
const quad_lws  = [1.5, 1.5, 1.5, 1.5, 2.5]
const quad_alph = [0.7, 0.7, 0.7, 0.7, 1.0]

const grad_cols = ["#d62728", "#1f77b4", "k"]
const grad_lws  = [2.0, 2.0, 2.5]
const grad_alph = [0.9, 0.9, 1.0]

function plot_rho_triplet(filename, title, labels, cols, lws, alphas, stats)
    fig, axs = PythonPlot.subplots(1, 3; figsize=(13, 4.5), constrained_layout=true)

    panels = [
        (axs[0], stats.ρ_tg[1], stats.ρ_tg[2], L"\rho_L(\mathrm{GI},\,\mathrm{true})"),
        (axs[1], stats.ρ_tm[1], stats.ρ_tm[2], L"\rho_L(\mathrm{MAP},\,\mathrm{true})"),
        (axs[2], stats.ρ_gm[1], stats.ρ_gm[2], L"\rho_L(\mathrm{MAP},\,\mathrm{GI})"),
    ]

    for (ax, μ, σ, ttl) in panels
        ax.axhline(1.0; color="grey", ls=":", lw=1, alpha=0.6)
        ax.axhline(0.0; color="grey", ls="--", lw=0.6, alpha=0.4)

        for vi in 1:length(labels)
            msk = @. isfinite(μ[:, vi])
            !any(msk) && continue

            ax.plot(Lc_ref[msk], μ[msk, vi]; color=cols[vi], lw=lws[vi],
                    alpha=alphas[vi], label=labels[vi])
            ax.fill_between(Lc_ref[msk], μ[msk, vi] .- σ[msk, vi], μ[msk, vi] .+ σ[msk, vi];
                            color=cols[vi], alpha=0.12, linewidth=0)
        end

        ax.set_xlabel(L"L"; fontsize=12)
        ax.set_title(ttl; fontsize=11)
        ax.set_xlim(3000, 12000)
        ax.set_ylim(-0.1, 1.15)
    end

    axs[0].legend(frameon=false, fontsize=8, loc="lower left")
    fig.suptitle(title, fontsize=10)
    fig.savefig(joinpath(OUT_DIR, filename); dpi=200, bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved $filename")
end

function plot_sigma_pair(filename, title, ylabel, labels, cols, lws, alphas, σ_gi, σ_map)
    fig, axs = PythonPlot.subplots(1, 2; figsize=(10, 4.5), constrained_layout=true)

    for (ax, σ, lbl) in [(axs[0], σ_gi, "GI"), (axs[1], σ_map, "MAP")]
        for vi in 1:length(labels)
            msk = @. isfinite(σ[:, vi]) & (σ[:, vi] > 0)
            !any(msk) && continue

            ax.semilogy(Lc_ref[msk], kfac[msk] .* σ[msk, vi];
                        color=cols[vi], lw=lws[vi], alpha=alphas[vi], label=labels[vi])
        end

        ax.set_xlabel(L"L"; fontsize=12)
        ax.set_ylabel(ylabel; fontsize=10)
        ax.set_title(lbl; fontsize=11)
        ax.set_xlim(3000, 12000)
        ax.legend(frameon=false, fontsize=8)
    end

    fig.suptitle(title, fontsize=10)
    fig.savefig(joinpath(OUT_DIR, filename); dpi=200, bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved $filename")
end

function plot_oracle_amp_comparison(filename, base_stats, amp_stats)
    full_i = length(quad_labels)

    ρ_gi  = base_stats.ρ_tg[1][:, full_i]
    ρ_map = base_stats.ρ_tm[1][:, full_i]
    ρ_amp = amp_stats.ρ_tm[1][:, 1]
    ρ_mg  = base_stats.ρ_gm[1][:, full_i]
    ρ_ag  = amp_stats.ρ_gm[1][:, 1]

    σ_gi  = base_stats.σ_tg[:, full_i]
    σ_map = base_stats.σ_tm[:, full_i]
    σ_amp = amp_stats.σ_tm[:, 1]

    fig, axs = PythonPlot.subplots(1, 2; figsize=(11, 4.5), constrained_layout=true)

    ax = axs[0]
    ax.plot(Lc_ref, ρ_gi;  color="#1f77b4", lw=2, label="GI vs true")
    ax.plot(Lc_ref, ρ_map; color="#9467bd", lw=2, label="MAP vs true")
    ax.plot(Lc_ref, ρ_amp; color="#d62728", lw=2, label="MAP+local amp vs true")
    ax.plot(Lc_ref, ρ_mg;  color="0.4", lw=1.5, ls="--", label="MAP vs GI")
    ax.plot(Lc_ref, ρ_ag;  color="0.1", lw=1.5, ls=":", label="MAP+amp vs GI")
    ax.axhline(1.0; color="grey", ls=":", lw=1)
    ax.set_xlabel(L"L")
    ax.set_ylabel(L"\rho_L")
    ax.set_xlim(3000, 12000)
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Correlation")
    ax.legend(frameon=false, fontsize=8)

    ax = axs[1]
    for (σ, col, lbl) in [
        (σ_gi,  "#1f77b4", "GI"),
        (σ_map, "#9467bd", "MAP"),
        (σ_amp, "#d62728", "MAP+local amp"),
    ]
        msk = @. isfinite(σ) & (σ > 0)
        any(msk) && ax.semilogy(Lc_ref[msk], kfac[msk] .* σ[msk];
                                color=col, lw=2, label=lbl)
    end

    ax.set_xlabel(L"L")
    ax.set_ylabel(L"(L^2/2)^2\,\sigma[C_L^{\kappa_{\rm true},\hat\kappa}]")
    ax.set_xlim(3000, 12000)
    ax.set_title("Cross σ")
    ax.legend(frameon=false, fontsize=8)

    fig.suptitle("Oracle MAP local-amplitude test — 4×4 smooth amplitudes fitted to true high-L κ",
                 fontsize=10)
    fig.savefig(joinpath(OUT_DIR, filename); dpi=200, bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved $filename")
end

# A/C: quadrants
plot_rho_triplet(
    "rho_map_gi_true_quadrants.png",
    "A/C: Quadrant patch diagnostic — correlations  ($Nsim sims, $Np×$Np px patches)",
    quad_labels, quad_cols, quad_lws, quad_alph, quad_stats,
)

plot_sigma_pair(
    "sigma_cross_patch_quadrants.png",
    "C: Quadrant patch diagnostic — cross σ  ($Nsim sims)",
    L"(L^2/2)^2\,\sigma[C_L^{\kappa_{\rm true},\hat\kappa}]",
    quad_labels, quad_cols, quad_lws, quad_alph,
    quad_stats.σ_tg, quad_stats.σ_tm,
)

plot_sigma_pair(
    "sigma_auto_patch_quadrants.png",
    "C: Quadrant patch diagnostic — auto σ  ($Nsim sims, no N⁰ subtraction)",
    L"(L^2/2)^2\,\sigma[C_L^{\hat\kappa\hat\kappa}]",
    quad_labels, quad_cols, quad_lws, quad_alph,
    quad_stats.σ_gg, quad_stats.σ_mm,
)

# B: gradient-selected patches
if grad_stats !== nothing
    plot_rho_triplet(
        "rho_map_gi_true_gradient_patches.png",
        "B: Gradient-selected diagnostic — high vs low |∇TLP|  ($Ngrad sims)",
        grad_labels, grad_cols, grad_lws, grad_alph, grad_stats,
    )

    plot_sigma_pair(
        "sigma_cross_gradient_patches.png",
        "B/C: Gradient-selected diagnostic — cross σ  ($Ngrad sims)",
        L"(L^2/2)^2\,\sigma[C_L^{\kappa_{\rm true},\hat\kappa}]",
        grad_labels, grad_cols, grad_lws, grad_alph,
        grad_stats.σ_tg, grad_stats.σ_tm,
    )

    plot_sigma_pair(
        "sigma_auto_gradient_patches.png",
        "B/C: Gradient-selected diagnostic — auto σ  ($Ngrad sims, no N⁰ subtraction)",
        L"(L^2/2)^2\,\sigma[C_L^{\hat\kappa\hat\kappa}]",
        grad_labels, grad_cols, grad_lws, grad_alph,
        grad_stats.σ_gg, grad_stats.σ_mm,
    )
end

# D: oracle local-amplitude MAP test
if amp_stats !== nothing
    plot_oracle_amp_comparison(
        "oracle_local_amp_map_comparison.png",
        quad_stats,
        amp_stats,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Example high-L κ maps
# ─────────────────────────────────────────────────────────────────────────────

let
    s0 = first(shared)

    t0 = jldopen(QEGI_FILE, "r") do f
        Float64.(read(f, "sim_$s0/ϕ_true"))
    end
    g0 = jldopen(QEGI_FILE, "r") do f
        Float64.(read(f, "sim_$s0/ϕ_gi_b"))
    end
    m0 = jldopen(MAP_FILE, "r") do f
        Float64.(read(f, "sim_$s0/ϕ_mj"))
    end

    g0_deb = debias_arr(g0, WL_gi, ℓ2d_full; minW_local=minW)
    m0_deb = debias_arr(m0, WL_mj, ℓ2d_full; minW_local=minW)

    kt = bandpass_kappa_arr(t0,     ℓ2d_full; Llo=4000.0, Lhi=12000.0)
    kg = bandpass_kappa_arr(g0_deb, ℓ2d_full; Llo=4000.0, Lhi=12000.0)
    km = bandpass_kappa_arr(m0_deb, ℓ2d_full; Llo=4000.0, Lhi=12000.0)

    maps = [kt, kg, km, km .- kg]
    lbls = [
        L"\kappa_{\rm true}\;(4000<L<12000)",
        L"\hat\kappa_{\rm GI}\;(4000<L<12000)",
        L"\hat\kappa_{\rm MAP}\;(4000<L<12000)",
        L"\hat\kappa_{\rm MAP}-\hat\kappa_{\rm GI}",
    ]
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r", "PiYG"]

    fig, axs = PythonPlot.subplots(1, 4; figsize=(16, 4.5), constrained_layout=true)

    for (ax, mp, lbl, cm) in [
        (axs[0], maps[1], lbls[1], cmaps[1]),
        (axs[1], maps[2], lbls[2], cmaps[2]),
        (axs[2], maps[3], lbls[3], cmaps[3]),
        (axs[3], maps[4], lbls[4], cmaps[4]),
    ]
        vm = max(quantile(abs.(vec(mp)), 0.995), 1e-30)
        im = ax.imshow(mp; cmap=cm, vmin=-vm, vmax=vm, origin="lower",
                       extent=[0, Nside * θpix / 60, 0, Nside * θpix / 60])

        fig.colorbar(im; ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(lbl; fontsize=10)
        ax.set_xlabel("deg"; fontsize=9)
        ax.set_ylabel("deg"; fontsize=9)

        for (_, rows, cols) in quadrant_patches
            r0 = (first(rows) - 1) * θpix / 60
            r1 = last(rows) * θpix / 60
            c0 = (first(cols) - 1) * θpix / 60
            c1 = last(cols) * θpix / 60
            ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0];
                    color="k", lw=0.7, ls="--", alpha=0.5)
        end
    end

    fig.suptitle("Example sim $s0 — high-L κ fields, W-debiased GI/MAP", fontsize=10)
    fig.savefig(joinpath(OUT_DIR, "example_maps_true_gi_map_residual.png");
                dpi=200, bbox_inches="tight")
    PythonPlot.plotclose("all")
    println("Saved example_maps_true_gi_map_residual.png")
end

# ─────────────────────────────────────────────────────────────────────────────
# Save summary
# ─────────────────────────────────────────────────────────────────────────────

jldsave(joinpath(OUT_DIR, "local_patch_stats.jld2");
    shared,
    Lc_ref,
    quad_labels,
    grad_labels,
    grad_ok,
    grad_scores_high,
    grad_scores_low,
    amp_ok,
    amp_A,
    quad_stats,
    grad_stats,
    amp_stats,
)

println("\nAll plots saved to $OUT_DIR/")
println("Shared sims: $Nsim")
println("Quadrant patch size: $(Np)×$(Np) px = $(round(Np * θpix / 60; sigdigits=3))°")
println("Gradient-selected sims used: $Ngrad")
println("Oracle local-amplitude sims used: $Namp")
println("Notes:")
println("  A: local convergence is tested by ρ(GI,true), ρ(MAP,true), ρ(MAP,GI).")
println("  B: gradient plots test GI spatial weighting using high/low-|∇TLP| patches.")
println("  C: σ plots compare local cross/auto scatter; auto σ has no N0 subtraction.")
println("  D: oracle local-amplitude MAP test uses true κ, so it is diagnostic only.")