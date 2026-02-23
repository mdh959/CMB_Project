using CMBLensing
using LinearAlgebra
using Statistics: mean
using JLD2: @save, @load

###############################################################################
# Iterative QE-based W_L for TT (analytic-ish, using CMBLensing.jl internals)
#
# ASSUMPTIONS / LIMITATIONS
# - We approximate "partial delensing" by replacing ds.Cϕ with a residual spectrum
#     Cϕ_res = (1 - ρ^2) * Cϕ_fid
#   This is a proxy for the LensIt / plancklens iteration (which updates CMB spectra).
# - We use QE with weights=:unlensed so diag(qe.Nϕ) can be interpreted as the
#   analytic reconstruction noise N_L for the φ estimator in this implementation.
# - For this QE-only proxy, we take the effective response as R_L ≈ 1/N_L.
###############################################################################

"""
    compute_WL_iterative_QE_TT(ds; niter=10, clampρ=(0.0,0.999), keep_history=true)

Returns:
- WL      : final ρ_L curve (proxy Wiener/transfer)
- N0      : final iteration QE noise curve (interpreted as N_L)
- RL      : final iteration effective response, defined as RL = 1/N0
- Cϕ_res  : final residual φφ used in the last iteration
- hist    : (ρ_history, N0_history) if keep_history=true
"""
function compute_WL_iterative_QE_TT(
    ds::CMBLensing.DataSet;
    niter::Int = 10,
    clampρ = (0.0, 0.999),
    keep_history::Bool = true,
)

    Cϕ_fid = Float64.(diag(ds.Cϕ))
    Cϕ_res = copy(Cϕ_fid)

    ρmin, ρmax = clampρ
    N0 = similar(Cϕ_fid)
    ρ  = similar(Cϕ_fid)

    hist_ρ  = keep_history ? Vector{Vector{Float64}}() : nothing
    hist_N0 = keep_history ? Vector{Vector{Float64}}() : nothing

    for it in 1:niter
        ds_tmp = ds(Cϕ = Diagonal(Cϕ_res))

        # TT QE, and interpret qe.Nϕ as N_L only for weights=:unlensed
        qe = quadratic_estimate(ds_tmp, :TT; wiener_filtered=false, weights=:unlensed)
        N0 .= Float64.(diag(qe.Nϕ))   # == diag(qe.AL) here

        # ρ_L = C / (C + N), handle zero/negative power modes
        for i in eachindex(ρ)
            if Cϕ_fid[i] > 0 && N0[i] > 0
                ρ[i] = Cϕ_fid[i] / (Cϕ_fid[i] + N0[i])
            else
                ρ[i] = 0.0
            end
        end
        ρ .= clamp.(ρ, ρmin, ρmax)

        # update residual lensing power
        @. Cϕ_res = (1.0 - ρ^2) * Cϕ_fid

        if keep_history
            push!(hist_ρ, copy(ρ))
            push!(hist_N0, copy(N0))
        end
    end

    WL = copy(ρ)
    RL = similar(N0)
    for i in eachindex(RL)
        RL[i] = N0[i] > 0 ? 1.0 / N0[i] : 0.0
    end

    return (; WL, N0=copy(N0), RL=copy(RL), Cϕ_res=copy(Cϕ_res), hist=(hist_ρ, hist_N0))
end


# ──────────────────────────────────────────────────────────────────
# Empirical W_L from Monte Carlo simulations
# ──────────────────────────────────────────────────────────────────

"""
    empirical_WL_maps_loadsim(Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask; ...)

Run `nsims` simulations, compute MAP_joint, MAP_marg, and QE for each,
and return the sim-averaged transfer functions W_L and mean error maps.

Returns a NamedTuple with fields:
- `ℓ`:  bin centres
- `WL_joint`, `WL_marg`, `WL_qe`:  empirical W_L for each estimator
- `mean_Δϕ²_joint`, `mean_Δϕ²_marg`, `mean_Δϕ²_qe`:  mean squared error maps (Nside×Nside)

Checkpoints to `checkpoint_file` after each sim for resumability.
"""
function empirical_WL_maps_loadsim(
    Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
    nsims=50, nbins=300,
    checkpoint_file="results/WL_checkpoint.jld2"
)
    # Always run sims in Float64 for numerical stability
    T = Float64

    R_joint_sims = Vector{Vector{Float64}}()
    R_marg_sims  = Vector{Vector{Float64}}()
    R_qe_sims    = Vector{Vector{Float64}}()
    ℓ_template      = nothing
    W_joint_running  = nothing
    W_marg_running   = nothing
    W_qe_running     = nothing

    # Storage for individual phi maps (Float64, Nside×Nside each)
    ϕ_true_all  = Vector{Matrix{Float64}}()
    ϕ_qe_all    = Vector{Matrix{Float64}}()
    ϕ_joint_all = Vector{Matrix{Float64}}()
    ϕ_marg_all  = Vector{Matrix{Float64}}()

    # Running sums for mean error maps (Float64 to avoid truncation)
    sum_Δϕ²_joint = zeros(Float64, Nside, Nside)
    sum_Δϕ²_marg  = zeros(Float64, Nside, Nside)
    sum_Δϕ²_qe    = zeros(Float64, Nside, Nside)
    nsims_completed = 0

    # Resume from checkpoint if available
    start_from = 1
    if isfile(checkpoint_file)
        println("Resuming from checkpoint: $checkpoint_file")
        @load checkpoint_file R_joint_sims R_marg_sims ℓ_template W_joint_running W_marg_running
        # Load new fields if they exist in checkpoint
        try @load checkpoint_file R_qe_sims W_qe_running catch end
        try @load checkpoint_file sum_Δϕ²_joint sum_Δϕ²_marg sum_Δϕ²_qe nsims_completed catch end
        if R_qe_sims === nothing; R_qe_sims = Vector{Vector{Float64}}() end
        # Load phi maps from separate file if it exists
        phi_file = replace(checkpoint_file, ".jld2" => "_phis.jld2")
        if isfile(phi_file)
            try @load phi_file ϕ_true_all ϕ_qe_all ϕ_joint_all ϕ_marg_all catch end
        end
        start_from = length(R_joint_sims) + 1
        nsims_completed = length(R_joint_sims)
        println("→ Starting from simulation $start_from")
    end

    for s in start_from:nsims
        println("→ Simulation $s / $nsims")

        (; f, f̃, ϕ, ds) = load_sim(
            seed=1000 + s,
            Cℓ=Cℓ, Cℓn=Cℓn,
            θpix=θpix, T=T, Nside=Nside,
            beamFWHM=1.0, pol=pol,
            bandpass_mask=bandpass_mask,
            pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )

        ϕ_true = Map(ϕ)

        # QE (Wiener-filtered)
        qe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true)
        ϕqe = qe.ϕqe

        # MAP_marg
        ϕ_marg = nothing
        try
            ϕ_marg, _ = MAP_marg(
                ds; ϕstart=ϕqe,
                nsteps=15, progress=false,
                conjgrad_kwargs=(tol=1e-3, nsteps=100),
            )
        catch err
            println("  MAP_marg failed: $err — skipping sim $s")
            continue
        end

        # MAP_joint
        Ωstart = FieldTuple(ϕ=ϕqe)
        ϕ_joint = nothing
        try
            result = MAP_joint(
                ds, Ωstart;
                nsteps=15, progress=false,
                conjgrad_kwargs=(tol=1e-3, nsteps=100),
            )
            ϕ_joint = result.ϕ
        catch err
            println("  MAP_joint failed: $err — skipping sim $s")
            continue
        end

        # Cross-spectra ratios for W_L
        cl_tj = get_Cℓ(ϕ, ϕ_joint; nbins=nbins)
        cl_tm = get_Cℓ(ϕ, ϕ_marg;  nbins=nbins)
        cl_tq = get_Cℓ(ϕ, ϕqe;     nbins=nbins)
        cl_tt = get_Cℓ(ϕ, ϕ_true;   nbins=nbins)

        if ℓ_template === nothing
            ℓ_template = cl_tt.ℓ
        end

        push!(R_joint_sims, cl_tj.Cℓ ./ cl_tt.Cℓ)
        push!(R_marg_sims,  cl_tm.Cℓ ./ cl_tt.Cℓ)
        push!(R_qe_sims,    cl_tq.Cℓ ./ cl_tt.Cℓ)

        # Running averages
        W_joint_running = mean(reduce(hcat, R_joint_sims); dims=2)[:]
        W_marg_running  = mean(reduce(hcat, R_marg_sims);  dims=2)[:]
        W_qe_running    = mean(reduce(hcat, R_qe_sims);    dims=2)[:]

        # Checkpoint (W_L stats only — small file, safe to push)
        @save checkpoint_file R_joint_sims R_marg_sims R_qe_sims ℓ_template W_joint_running W_marg_running W_qe_running sum_Δϕ²_joint sum_Δϕ²_marg sum_Δϕ²_qe nsims_completed

        # Save phi maps separately (large file — keep on cluster, don't push)
        phi_file = replace(checkpoint_file, ".jld2" => "_phis.jld2")
        @save phi_file ϕ_true_all ϕ_qe_all ϕ_joint_all ϕ_marg_all
        println("  Saved checkpoint after sim $s  ($nsims_completed completed)")
    end

    mean_Δϕ²_joint = sum_Δϕ²_joint ./ max(nsims_completed, 1)
    mean_Δϕ²_marg  = sum_Δϕ²_marg  ./ max(nsims_completed, 1)
    mean_Δϕ²_qe    = sum_Δϕ²_qe    ./ max(nsims_completed, 1)

    return (
        ℓ = ℓ_template,
        WL_joint = W_joint_running,
        WL_marg  = W_marg_running,
        WL_qe    = W_qe_running,
        mean_Δϕ²_joint = mean_Δϕ²_joint,
        mean_Δϕ²_marg  = mean_Δϕ²_marg,
        mean_Δϕ²_qe    = mean_Δϕ²_qe,
        nsims = nsims_completed,
    )
end

function empirical_WL_qe_loadsim(
    Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
    nsims=50, nbins=300,
    checkpoint_file="results/WL_checkpoint_qe.jld2"
)
    # Always run sims in Float64 for numerical stability
    T = Float64

    R_qe_sims    = Vector{Vector{Float64}}()
    ℓ_template   = nothing
    W_qe_running = nothing

    # Resume from checkpoint if available
    start_from = 1
    if isfile(checkpoint_file)
        println("Resuming from checkpoint: $checkpoint_file")
        @load checkpoint_file R_qe_sims ℓ_template W_qe_running
        start_from = length(R_qe_sims) + 1
        println("→ Starting from simulation $start_from")
    end

    for s in start_from:nsims
        println("→ Simulation $s / $nsims")

        (; f, f̃, ϕ, ds) = load_sim(
            seed=1000 + s,
            Cℓ=Cℓ, Cℓn=Cℓn,
            θpix=θpix, T=T, Nside=Nside,
            beamFWHM=1.0, pol=pol,
            bandpass_mask=bandpass_mask,
            pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )

        ϕ_true = Map(ϕ)

        # QE (Wiener-filtered)
        qe  = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true)
        ϕqe = qe.ϕqe

        # Cross-spectrum ratio for W_L (QE)
        cl_tq = get_Cℓ_fft(ϕ, ϕqe;    nbins=nbins)
        cl_tt = get_Cℓ_fft(ϕ, ϕ_true; nbins=nbins)

        if ℓ_template === nothing
            ℓ_template = cl_tt.ℓ
        end

        push!(R_qe_sims, cl_tq.Cℓ ./ (cl_tt.Cℓ .+ eps(Float64)))

        # Running average
        W_qe_running = mean(reduce(hcat, R_qe_sims); dims=2)[:]

        # Checkpoint
        @save checkpoint_file R_qe_sims ℓ_template W_qe_running
    end

    return (
        ℓ     = ℓ_template,
        WL_qe = W_qe_running,
        nsims = length(R_qe_sims),
    )
end




#=  ── plancklens-based normalisation (disabled: requires plancklens) ──────

using PythonCall

# Python imports
const np      = pyimport("numpy")
const utils   = pyimport("plancklens.utils")
const qresp   = pyimport("plancklens.qresp")
const nhl     = pyimport("plancklens.nhl")
const camb_lc = pyimport("camb.correlations")  # lensed_cls lives here


Convert φφ -> dd for CAMB lensed_cls, and back.

CAMB's `lensed_cls(dls_unl, cldd)` expects:
- dls_unl: array of unlensed Dl's (TT, EE, BB, TE) in CAMB convention
- cldd: deflection power spectrum C_L^{dd}

Many pipelines store C_L^{φφ}. They are related by:
  d = ∇φ  ⇒  C_L^{dd} = L(L+1) C_L^{φφ}
(conventionally; check your exact definition)

We'll use:
  cldd[L] = L*(L+1)*clpp[L]

function clpp_to_cldd(clpp::Vector{Float64})
    L = collect(0:length(clpp)-1)
    return (L .* (L .+ 1.0)) .* clpp
end

function cldd_to_clpp(cldd::Vector{Float64})
    L = collect(0:length(cldd)-1)
    out = similar(cldd)
    out[1] = 0.0
    @. out[2:end] = cldd[2:end] / (L[2:end] * (L[2:end] + 1.0))
    return out
end


Build CAMB-style dls_unl array from cls_unl dict.

Expected input cls_unl keys: "tt","ee","bb","te"
Each is a Vector length >= lmax+1 containing C_ℓ.

CAMB lensed_cls wants Dl = ℓ(ℓ+1)Cℓ/(2π)
in columns [TT, EE, BB, TE].

function cls_to_dls_unl(cls_unl::Dict{String,Vector{Float64}}, lmax::Int)
    ℓ = collect(0:lmax)
    fac = @. ℓ*(ℓ+1)/(2π)
    dls = zeros(Float64, lmax+1, 4)
    dls[:,1] .= fac .* cls_unl["tt"][1:lmax+1]
    dls[:,2] .= fac .* cls_unl["ee"][1:lmax+1]
    dls[:,3] .= fac .* cls_unl["bb"][1:lmax+1]
    dls[:,4] .= fac .* cls_unl["te"][1:lmax+1]
    return dls
end


Inverse: CAMB Dl array -> cls dict

function dls_to_cls(dls::Matrix{Float64})
    lmax = size(dls,1)-1
    ℓ = collect(0:lmax)
    fac = similar(ℓ, Float64)
    fac[1] = 0.0
    @. fac[2:end] = (2π) / (ℓ[2:end]*(ℓ[2:end]+1))
    cls = Dict{String,Vector{Float64}}()
    cls["tt"] = fac .* dls[:,1]
    cls["ee"] = fac .* dls[:,2]
    cls["bb"] = fac .* dls[:,3]
    cls["te"] = fac .* dls[:,4]
    return cls
end


Compute partially-lensed CMB cls given unlensed cls and residual clpp_res.

Uses CAMB's non-perturbative lensed_cls(dls_unl, cldd).

function partially_lensed_cls(cls_unl::Dict{String,Vector{Float64}},
                              clpp_res::Vector{Float64},
                              lmax_cmb::Int)
    dls_unl = cls_to_dls_unl(cls_unl, lmax_cmb)
    cldd_res = clpp_to_cldd(clpp_res)[1:lmax_cmb+1]  # CAMB expects length lmax+1
    dls_len = camb_lc.lensed_cls(np.array(dls_unl), np.array(cldd_res))
    # dls_len is python array -> convert back
    dls_len_j = Array{Float64}(dls_len)
    return dls_to_cls(dls_len_j)
end


Compute (R_L, N0_L) for TT using plancklens, given:
- cls_cmb_filt: fiducial partially-lensed CMB spectra (weights)
- cls_cmb_dat : sampled partially-lensed CMB spectra (response side)
- cls_noise_filt/dat: noise spectra (beam-deconvolved) dict with "tt" etc
- lmin_ivf, lmax_ivf, lmax_qlm: multipole cuts

function RL_N0_TT_plancklens(cls_cmb_filt::Dict{String,Vector{Float64}},
                             cls_cmb_dat::Dict{String,Vector{Float64}},
                             cls_noise_filt::Dict{String,Vector{Float64}},
                             cls_noise_dat::Dict{String,Vector{Float64}},
                             lmin_ivf::Int, lmax_ivf::Int, lmax_qlm::Int)

    qe_key = "ptt"

    # Build fals (= inverse of total filtering Cls)
    # plancklens expects python dicts of numpy arrays
    fals = Dict{String,Any}()
    dat_cls = Dict{String,Any}()

    # TT-only
    fals["tt"] = np.array(cls_cmb_filt["tt"][1:lmax_ivf+1] .+ cls_noise_filt["tt"][1:lmax_ivf+1])
    dat_cls["tt"] = np.array(cls_cmb_dat["tt"][1:lmax_ivf+1] .+ cls_noise_dat["tt"][1:lmax_ivf+1])

    fals = utils.cl_inverse(fals)

    # zero below lmin
    fals["tt"][1:lmin_ivf] .= 0.0
    dat_cls["tt"][1:lmin_ivf] .= 0.0

    # QE weights and "response" spectra
    cls_w = Dict("tt" => np.array(cls_cmb_filt["tt"][1:lmax_ivf+1]))
    cls_f = Dict("tt" => np.array(cls_cmb_dat["tt"][1:lmax_ivf+1]))
    cls_w["tt"][1:lmin_ivf] .= 0.0
    cls_f["tt"][1:lmin_ivf] .= 0.0

    # cls_ivfs = fals * dat_cls * fals (scalar TT case)
    cls_ivfs = Dict("tt" => fals["tt"] .* dat_cls["tt"] .* fals["tt"])

    # nhl "n_gg"
    n_gg = nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]

    # response r_gg_true (this is your R_L)
    r_gg_true = qresp.get_response(qe_key, lmax_ivf, "p", cls_w, cls_f, fals, lmax_qlm=lmax_qlm)[0]

    n_gg_j = Array{Float64}(n_gg)
    r_gg_j = Array{Float64}(r_gg_true)

    # unbiased N0 (normalized with true response)
    N0 = n_gg_j ./ (r_gg_j .^ 2)

    return r_gg_j, N0
end


Full iterative loop (paper-like) for TT:
- iterate residual clpp -> partially-lensed CMB cls -> compute (R_L, N0) -> rho -> update residual
- returns final WL and RL, plus histories

function iterative_WL_TT_paperlike(
    cls_unl_fid::Dict{String,Vector{Float64}},
    cls_noise::Dict{String,Vector{Float64}},
    clpp_fid::Vector{Float64};
    lmin_ivf::Int = 30,
    lmax_ivf::Int = 3000,
    lmax_qlm::Int = 3000,
    niter::Int = 8,
)

    # In your “no chain” case, sampled == fid
    clpp_res = copy(clpp_fid)

    hist_rho = Vector{Vector{Float64}}()
    hist_RL  = Vector{Vector{Float64}}()
    hist_N0  = Vector{Vector{Float64}}()

    for it in 1:niter
        # 1) partially-lensed CMB spectra from residual lensing
        cls_plen = partially_lensed_cls(cls_unl_fid, clpp_res, lmax_ivf)

        # 2) compute R_L and N0_L from plancklens
        RL, N0 = RL_N0_TT_plancklens(cls_plen, cls_plen, cls_noise, cls_noise,
                                     lmin_ivf, lmax_ivf, lmax_qlm)

        # 3) rho_L = C / (C + N0)
        rho = similar(N0)
        @. rho = clpp_fid[1:length(N0)] / (clpp_fid[1:length(N0)] + N0)

        push!(hist_rho, rho)
        push!(hist_RL, RL)
        push!(hist_N0, N0)

        # 4) update residual lensing clpp_res = (1 - rho^2) * clpp
        clpp_res_new = copy(clpp_res)
        @. clpp_res_new[1:length(rho)] = (1.0 - rho^2) * clpp_fid[1:length(rho)]
        clpp_res = clpp_res_new
    end

    RL_final = hist_RL[end]
    # Eq (2.8): W_L = C / (C + 1/R_L)
    WL_final = similar(RL_final)
    @. WL_final = clpp_fid[1:length(RL_final)] / (clpp_fid[1:length(RL_final)] + 1.0 / RL_final)

    return (; WL=WL_final, RL=RL_final, N0=hist_N0[end], rho=hist_rho[end],
            clpp_res=clpp_res, hist=(hist_rho, hist_RL, hist_N0))
end
=#