#!/usr/bin/env julia

# diag_map_zero_gihess.jl
#
# Standard MAP-from-zero diagnostic with optional GI-informed phi Hessian.
#
# Baseline:
#   GI_PHI_HESS=0 julia diag_map_zero_gihess.jl
#
# GI-Hessian tests:
#   GI_PHI_HESS=1 GI_HESS_ETA=0.25 julia diag_map_zero_gihess.jl
#   GI_PHI_HESS=1 GI_HESS_ETA=0.5  julia diag_map_zero_gihess.jl
#   GI_PHI_HESS=1 GI_HESS_ETA=1.0  julia diag_map_zero_gihess.jl
#
# Useful overrides:
#   NSIMS=5 MAPJ_STEPS=10 julia diag_map_zero_gihess.jl
#   MUKARCMIN_T=0.1 BEAM_FWHM=0.3 LMAX=12000 julia diag_map_zero_gihess.jl
#
# Output:
#   results/diag_map_zero_gihess_<tag>.jld2
#
# Main diagnostic row saved:
#   sim, seed, hessian, eta, beta, final_logpdf, mean_alpha,
#   final_rho_hiL, final_W_hiL, sigma_cross_hiL

using CMBLensing
using JLD2
using LinearAlgebra
using Statistics
using Printf
using Dates

# ================================================================
# Environment/config
# ================================================================

const GI_PHI_HESS  = get(ENV, "GI_PHI_HESS", "0") == "1"
const GI_HESS_ETA  = parse(Float64, get(ENV, "GI_HESS_ETA",  "0.5"))
const GI_HESS_BETA = parse(Float64, get(ENV, "GI_HESS_BETA", "1.0"))
const GI_LGRAD     = parse(Float64, get(ENV, "GI_LGRAD", "2000"))
const GI_HESS_FLOOR_FRAC = parse(Float64, get(ENV, "GI_HESS_FLOOR_FRAC", "1e-6"))

const NSIMS       = parse(Int, get(ENV, "NSIMS", "5"))
const MAPJ_STEPS  = parse(Int, get(ENV, "MAPJ_STEPS", "10"))
const CG_TOL      = parse(Float64, get(ENV, "CG_TOL", "1e-3"))
const CG_STEPS    = parse(Int, get(ENV, "CG_STEPS", "500"))

const SEED0       = parse(Int, get(ENV, "SEED0", "1000"))
const MUKARCMIN_T = parse(Float64, get(ENV, "MUKARCMIN_T", "0.1"))
const BEAM_FWHM   = parse(Float64, get(ENV, "BEAM_FWHM", "0.3"))
const LMAX        = parse(Int, get(ENV, "LMAX", "12000"))

const Nside       = parse(Int, get(ENV, "NSIDE", "512"))
const θpix        = parse(Float64, get(ENV, "THETAPIX", "0.7438046267475303"))
const pol         = :I
const Ttype       = Float64

const LHI_MIN     = parse(Float64, get(ENV, "LHI_MIN", "4000"))
const LHI_MAX     = parse(Float64, get(ENV, "LHI_MAX", string(LMAX)))

const OUTDIR      = get(ENV, "OUTDIR", "results")
mkpath(OUTDIR)

const TAG = @sprintf(
    "hess%s_eta%.3g_beta%.3g_ns%d_steps%d_lmax%d_noise%.3g_beam%.3g",
    GI_PHI_HESS ? "GI" : "base",
    GI_HESS_ETA,
    GI_HESS_BETA,
    NSIMS,
    MAPJ_STEPS,
    LMAX,
    MUKARCMIN_T,
    BEAM_FWHM,
)

const OUTFILE = joinpath(OUTDIR, "diag_map_zero_gihess_$(TAG).jld2")

println("============================================================")
println("MAP zero-start GI-Hessian diagnostic")
println("time           = ", now())
println("GI_PHI_HESS    = ", GI_PHI_HESS)
println("GI_HESS_ETA    = ", GI_HESS_ETA)
println("GI_HESS_BETA   = ", GI_HESS_BETA)
println("GI_LGRAD       = ", GI_LGRAD)
println("NSIMS          = ", NSIMS)
println("MAPJ_STEPS     = ", MAPJ_STEPS)
println("CG             = tol=$(CG_TOL), nsteps=$(CG_STEPS)")
println("noise/beam     = $(MUKARCMIN_T) μK-arcmin, $(BEAM_FWHM) arcmin")
println("Nside, θpix    = $(Nside), $(θpix) arcmin")
println("Lmax           = ", LMAX)
println("high-L band    = $(LHI_MIN) < L < $(LHI_MAX)")
println("OUTFILE        = ", OUTFILE)
println("============================================================")

# ================================================================
# Helpers
# ================================================================

finite_mean(x) = mean(x[isfinite.(x)])

function safe_median_positive(x)
    good = isfinite.(x) .& (x .> 0)
    return any(good) ? median(x[good]) : one(eltype(x))
end

function fftfreq_ell(N::Int, θpix_arcmin::Float64)
    arcmin_to_rad = π / (180.0 * 60.0)
    pixrad = θpix_arcmin * arcmin_to_rad
    side_rad = N * pixrad
    k = Vector{Float64}(undef, N)
    for i in 1:N
        n = i - 1
        if n <= div(N, 2)
            k[i] = 2π * n / side_rad
        else
            k[i] = 2π * (n - N) / side_rad
        end
    end
    return k
end

function ell_grids(N::Int=Nside, θpix_arcmin::Float64=θpix)
    k = fftfreq_ell(N, θpix_arcmin)
    Lx = repeat(reshape(k, N, 1), 1, N)
    Ly = repeat(reshape(k, 1, N), N, 1)
    L2 = Lx.^2 .+ Ly.^2
    L  = sqrt.(L2)
    return Lx, Ly, L2, L
end

const Lx_grid, Ly_grid, L2_grid, L_grid = ell_grids()

function as_array(x)
    return Array(x)
end

function fourier_array(x)
    return Array(Fourier(x).arr)
end

function get_Tmap(d)
    # Temperature-only simulations usually have d as a scalar map.
    # If your local CMBLensing stores temperature in a field tuple,
    # these fallbacks catch the common names.
    if hasproperty(d, :I)
        return getproperty(d, :I)
    elseif hasproperty(d, :T)
        return getproperty(d, :T)
    else
        return d
    end
end

function map_like(template, arr)
    # Construct a CMBLensing map/Fourier-map-like object using template projection.
    # This works in most CMBLensing.jl versions.
    return Map(arr, template.proj)
end

function zero_phi_start(ds)
    return FieldTuple(ϕ = zero(diag(ds.Cϕ)))
end

function safe_alpha_hist(history)
    vals = Float64[]
    for h in history
        if hasproperty(h, :α)
            push!(vals, Float64(getproperty(h, :α)))
        end
    end
    return vals
end

function safe_final_logpdf(history)
    isempty(history) && return NaN
    h = history[end]
    if hasproperty(h, :total_logpdf)
        return Float64(getproperty(h, :total_logpdf))
    elseif hasproperty(h, :logpdf)
        lp = getproperty(h, :logpdf)
        try
            return Float64(sum(lp))
        catch
            return NaN
        end
    else
        return NaN
    end
end

# ================================================================
# GI-informed Hessian pieces
# ================================================================

function realised_gradient_tensor_from_data(ds; Lgrad=GI_LGRAD)
    Tmap = get_Tmap(ds.d)

    # Low-pass observed temperature.
    # If this fails locally, replace with the same low-pass operation used in your GI script.
    Tlp = LowPass(Lgrad) * Tmap

    # Gradient of low-pass map.
    g = ∇ * Tlp

    # Common CMBLensing layouts:
    #   g[1], g[2]
    # or properties x/y.
    Tx = try
        g[1]
    catch
        getproperty(g, :x)
    end

    Ty = try
        g[2]
    catch
        getproperty(g, :y)
    end

    Sxx = finite_mean(as_array(Tx .* Tx))
    Sxy = finite_mean(as_array(Tx .* Ty))
    Syy = finite_mean(as_array(Ty .* Ty))

    return Sxx, Sxy, Syy
end

function safe_diag_array(C)
    return Array(diag(C))
end

function build_gi_phi_hessian(ds; beta=GI_HESS_BETA, eta=GI_HESS_ETA, Lgrad=GI_LGRAD)
    Sxx, Sxy, Syy = realised_gradient_tensor_from_data(ds; Lgrad=Lgrad)

    @info "GI phi Hessian realised gradient tensor" Sxx Sxy Syy beta eta Lgrad

    phi_diag = diag(ds.Cϕ)
    Cphi = safe_diag_array(ds.Cϕ)
    Cphi_inv = pinv.(Cphi)

    # Lensed temperature spectrum term.
    # For temperature-only :I this should be the lensed TT covariance diagonal.
    Ctt_lensed = safe_diag_array(ds.Cf̃)

    # Deconvolved noise N/B^2.
    # These names match the objects you have used elsewhere.
    # If your CMBLensing version stores beam/noise differently, replace these two lines
    # by the same Ctt_tot denominator used in your GI N0 code:
    #     Ctt_tot = C_lensed_TT + N_TT / B^2
    Bdiag = safe_diag_array(ds.B̂)
    Ndiag = safe_diag_array(ds.Cn̂)
    Ntt_deconv = Ndiag ./ max.(Bdiag.^2, eps(Float64))

    Ctt_tot = Ctt_lensed .+ Ntt_deconv

    # GI Fisher likelihood curvature:
    # Ngi_inv(L) = (L_i S_ij L_j) / (C_lensed_TT + N/B^2)
    LSL = Lx_grid.^2 .* Sxx .+ 2 .* Lx_grid .* Ly_grid .* Sxy .+ Ly_grid.^2 .* Syy
    Ngi_inv = LSL ./ max.(Ctt_tot, eps(Float64))

    # Total phi Hessian in physical/mixed phi coordinates.
    H = beta .* Cphi_inv .+ eta .* Ngi_inv

    # Safety cleanup/floor.
    Hmed = safe_median_positive(H)
    good = isfinite.(H) .& (H .> 0)
    H .= ifelse.(good, H, Hmed)
    H .= max.(H, GI_HESS_FLOOR_FRAC * Hmed)

    Hmap = map_like(phi_diag, H)

    # MAP_joint asks for a Hessian operator over Ω°, whose key is usually :ϕ°.
    return Diagonal(FieldTuple(ϕ° = Hmap))
end

# ================================================================
# Monkey patch only the phi Hessian.
# Leaves Wiener/f solve unchanged.
# ================================================================

@eval CMBLensing begin
    function Hessian_logpdf_preconditioner(Ω::Tuple, ds::DataSet)
        if Main.GI_PHI_HESS && (Ω == (:ϕ°,) || Ω == (:ϕ,))
            return Main.build_gi_phi_hessian(
                ds;
                beta = Main.GI_HESS_BETA,
                eta  = Main.GI_HESS_ETA,
                Lgrad = Main.GI_LGRAD,
            )
        else
            return invoke(
                Hessian_logpdf_preconditioner,
                Tuple{Union{Symbol,Tuple}, DataSet},
                Ω,
                ds,
            )
        end
    end
end

# ================================================================
# Diagnostics: high-L rho, W, residual sigma
# ================================================================

function hiL_metrics(phi_hat, phi_true; Lmin=LHI_MIN, Lmax=LHI_MAX)
    fh = fourier_array(phi_hat)
    ft = fourier_array(phi_true)

    mask = (L_grid .>= Lmin) .& (L_grid .< Lmax) .& isfinite.(real.(fh)) .& isfinite.(real.(ft))

    nh = count(mask)
    if nh == 0
        return (rho=NaN, W=NaN, sigma_cross=NaN, Chat=NaN, Ctrue=NaN, Ccross=NaN, nmodes=0)
    end

    ah = fh[mask]
    at = ft[mask]

    Chat   = mean(real.(ah .* conj.(ah)))
    Ctrue  = mean(real.(at .* conj.(at)))
    Ccross = mean(real.(ah .* conj.(at)))

    rho = Ccross / sqrt(max(Chat * Ctrue, eps(Float64)))
    W   = Ccross / max(Ctrue, eps(Float64))

    # Per-sim diagnostic: fractional RMS residual after response correction.
    # This is not a paper bandpower sigma; it is a compact high-L map-level residual diagnostic.
    Wsafe = abs(W) > 1e-8 ? W : sign(W + eps(Float64)) * 1e-8
    resid = ah ./ Wsafe .- at
    sigma_cross = sqrt(mean(abs2.(resid)) / max(mean(abs2.(at)), eps(Float64)))

    return (
        rho = rho,
        W = W,
        sigma_cross = sigma_cross,
        Chat = Chat,
        Ctrue = Ctrue,
        Ccross = Ccross,
        nmodes = nh,
    )
end

# ================================================================
# Main simulation/MAP loop
# ================================================================

println("Building spectra...")
Cℓ  = camb(r=0.05, ℓmax=max(21000, LMAX + 2000))
Cℓn = noiseCℓs(μKarcminT=MUKARCMIN_T, ℓknee=0, ℓmax=LMAX)

rows = NamedTuple[]
histories = Vector{Any}(undef, NSIMS)
phi_hats = Vector{Any}(undef, NSIMS)
phi_trues = Vector{Any}(undef, NSIMS)

for isim in 1:NSIMS
    seed = SEED0 + isim

    println()
    println("------------------------------------------------------------")
    @printf("Sim %d/%d  seed=%d\n", isim, NSIMS, seed)
    println("------------------------------------------------------------")

    (; f, f̃, ϕ, ds) = load_sim(
        seed = seed,
        Cℓ = Cℓ,
        Cℓn = Cℓn,
        θpix = θpix,
        T = Ttype,
        Nside = Nside,
        beamFWHM = BEAM_FWHM,
        pol = pol,
        bandpass_mask = LowPass(LMAX),
        pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0),
    )

    # Standard MAP from zero.
    Ωstart = zero_phi_start(ds)

    t_map = @elapsed begin
        out = MAP_joint(
            ds,
            Ωstart;
            nsteps = MAPJ_STEPS,
            minsteps = 0,
            fstart = nothing,
            αmax = nothing,
            nburnin_update_hessian = Inf,
            conjgrad_kwargs = (tol=CG_TOL, nsteps=CG_STEPS),
            history_keys = (:total_logpdf, :α, :ΔΩ°_norm),
            progress = true,
        )
    end

    phi_hat = out.ϕ
    phi_true = ϕ

    m = hiL_metrics(phi_hat, phi_true)
    alpha_hist = safe_alpha_hist(out.history)

    final_logpdf = safe_final_logpdf(out.history)
    mean_alpha = isempty(alpha_hist) ? NaN : mean(alpha_hist)

    row = (
        sim = isim,
        seed = seed,
        hessian = GI_PHI_HESS ? "GI_phi_hessian" : "baseline",
        eta = GI_HESS_ETA,
        beta = GI_HESS_BETA,
        Lgrad = GI_LGRAD,
        final_logpdf = final_logpdf,
        mean_alpha = mean_alpha,
        final_rho_hiL = m.rho,
        final_W_hiL = m.W,
        sigma_cross_hiL = m.sigma_cross,
        Chat_hiL = m.Chat,
        Ctrue_hiL = m.Ctrue,
        Ccross_hiL = m.Ccross,
        nmodes_hiL = m.nmodes,
        map_runtime_sec = t_map,
    )

    push!(rows, row)
    histories[isim] = out.history
    phi_hats[isim] = phi_hat
    phi_trues[isim] = phi_true

    @printf("done sim=%d seed=%d runtime=%.1fs\n", isim, seed, t_map)
    @printf("final_logpdf = %.6e\n", final_logpdf)
    @printf("mean_alpha   = %.4g\n", mean_alpha)
    @printf("rho_hiL      = %.4f\n", m.rho)
    @printf("W_hiL        = %.4f\n", m.W)
    @printf("sigma_hiL    = %.4f\n", m.sigma_cross)

    # Save incrementally, so a crash does not lose earlier sims.
    @save OUTFILE rows histories phi_hats phi_trues TAG GI_PHI_HESS GI_HESS_ETA GI_HESS_BETA GI_LGRAD NSIMS MAPJ_STEPS CG_TOL CG_STEPS MUKARCMIN_T BEAM_FWHM LMAX Nside θpix LHI_MIN LHI_MAX
end

# ================================================================
# Summary
# ================================================================

rho_vals = [r.final_rho_hiL for r in rows if isfinite(r.final_rho_hiL)]
W_vals   = [r.final_W_hiL for r in rows if isfinite(r.final_W_hiL)]
sig_vals = [r.sigma_cross_hiL for r in rows if isfinite(r.sigma_cross_hiL)]
alp_vals = [r.mean_alpha for r in rows if isfinite(r.mean_alpha)]

println()
println("============================================================")
println("SUMMARY")
println("outfile = ", OUTFILE)
println("hessian = ", GI_PHI_HESS ? "GI_phi_hessian" : "baseline")
println("eta     = ", GI_HESS_ETA)
println("beta    = ", GI_HESS_BETA)

if !isempty(rho_vals)
    @printf("mean rho_hiL   = %.5f ± %.5f\n", mean(rho_vals), std(rho_vals) / sqrt(length(rho_vals)))
end
if !isempty(W_vals)
    @printf("mean W_hiL     = %.5f ± %.5f\n", mean(W_vals), std(W_vals) / sqrt(length(W_vals)))
end
if !isempty(sig_vals)
    @printf("mean sigma_hiL = %.5f ± %.5f\n", mean(sig_vals), std(sig_vals) / sqrt(length(sig_vals)))
end
if !isempty(alp_vals)
    @printf("mean alpha     = %.5g\n", mean(alp_vals))
end

println("============================================================")