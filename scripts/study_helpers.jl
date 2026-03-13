# Shared configuration and helper functions for thesis lensing reconstruction study

using CMBLensing, JLD2, Statistics, Printf

# -----------------------------------------------------------------------
# Map configuration — matches user's existing setup
# -----------------------------------------------------------------------
const θpix  = 0.7438046267475303
const Nside = 512
const pol   = :I
const T     = Float64

# -----------------------------------------------------------------------
# ℓ bins for ϕ reconstruction (wide enough to cover small scales)
# -----------------------------------------------------------------------
const ℓedges_ϕ = [50, 100, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000]

# -----------------------------------------------------------------------
# MAP settings — tighter than defaults for better small-scale convergence
# -----------------------------------------------------------------------
const MAP_JOINT_KWARGS = (
    nsteps           = 15,
    conjgrad_kwargs  = (tol=1e-3, nsteps=100),
    progress         = false,
)

const MAP_MARG_KWARGS = (
    nsteps                      = 15,
    nsteps_with_meanfield_update = 6,
    Nsims                       = 30,
    conjgrad_kwargs             = (tol=1e-3, nsteps=100),
    progress                    = false,
)

# Define serial fallback only once (if the installed CMBLensing lacks these)
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset() = _DIST_DS[]
    end
end

# -----------------------------------------------------------------------
# Metrics: ρ(L) and N(L) from a single simulation
# -----------------------------------------------------------------------

"""
    compute_rho(ϕ_est, ϕ_true; ℓedges)

Cross-correlation coefficient ρ(L) in ℓ bins.
Normalization-independent — does not require empirical renormalization.
"""
function compute_rho(ϕ_est, ϕ_true; ℓedges=ℓedges_ϕ)
    Cl_cross = get_Cℓ(ϕ_est, ϕ_true; ℓedges=ℓedges)
    Cl_est   = get_Cℓ(ϕ_est;         ℓedges=ℓedges)
    Cl_true  = get_Cℓ(ϕ_true;        ℓedges=ℓedges)
    ℓ = Cl_cross.ℓ
    ρ = @. Cl_cross.Cℓ / sqrt(abs(Cl_est.Cℓ) * abs(Cl_true.Cℓ))
    return ℓ, ρ
end

"""
    compute_NL(ϕ_est, ϕ_true; ℓedges)

Reconstruction noise N(L) = C_est + C_true - 2*C_cross per ℓ bin.
Averaged over seeds this gives the mean-field noise power.
"""
function compute_NL(ϕ_est, ϕ_true; ℓedges=ℓedges_ϕ)
    Cl_cross = get_Cℓ(ϕ_est, ϕ_true; ℓedges=ℓedges)
    Cl_est   = get_Cℓ(ϕ_est;         ℓedges=ℓedges)
    Cl_true  = get_Cℓ(ϕ_true;        ℓedges=ℓedges)
    ℓ  = Cl_cross.ℓ
    NL = @. Cl_est.Cℓ + Cl_true.Cℓ - 2*Cl_cross.Cℓ
    return ℓ, NL
end

# -----------------------------------------------------------------------
# Core runner: all 4 estimators for one simulation
# -----------------------------------------------------------------------

"""
    run_one_sim(; seed, μKarcminT, lmax_cmb, Cℓ_theory, beamFWHM=1.0)

Run QE_raw, QE_WF, MAP_joint, MAP_marg for a single simulation.
Returns named tuple of ρ and N(L) for each method, plus the ℓ bin centres.
`Cℓ_theory` should be pre-computed with camb(ℓmax=21000) and reused across calls.
"""
function run_one_sim(; seed, μKarcminT, lmax_cmb, Cℓ_theory, beamFWHM=1.0)

    @printf("  seed=%d  μK=%.1f  lmax=%d\n", seed, μKarcminT, lmax_cmb)

    Cℓn = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=lmax_cmb)

    (;f, f̃, ϕ, ds) = load_sim(
        seed              = seed,
        Cℓ                = Cℓ_theory,
        Cℓn               = Cℓn,
        θpix              = θpix,
        T                 = T,
        Nside             = Nside,
        beamFWHM          = beamFWHM,
        pol               = pol,
        bandpass_mask     = LowPass(lmax_cmb),
        pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
    )

    # QE raw (normalized, not Wiener filtered)
    ϕ_qe_raw = quadratic_estimate(ds; weights=:lensed, wiener_filtered=false).ϕqe

    # QE Wiener filtered
    ϕ_qe_wf  = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true).ϕqe

    # MAP joint (start from QE WF — much better conditioned than QE raw)
    ϕ_map_joint = try
        Ωstart = FieldTuple(ϕ=ϕ_qe_wf)
        MAP_joint(ds, Ωstart; MAP_JOINT_KWARGS...).ϕ
    catch err
        @warn "MAP_joint failed (seed=$seed μK=$μKarcminT lmax=$lmax_cmb): $err — using QE as fallback"
        ϕ_qe_raw
    end

    # MAP marg (start from QE WF)
    ϕ_map_marg = try
        MAP_marg(ds; ϕstart=ϕ_qe_wf, MAP_MARG_KWARGS...)[1]
    catch err
        @warn "MAP_marg failed (seed=$seed μK=$μKarcminT lmax=$lmax_cmb): $err — using QE as fallback"
        ϕ_qe_raw
    end

    # Metrics
    ℓ, _   = compute_rho(ϕ_qe_raw, ϕ)   # ℓ is same for all

    rho = (;
        qe_raw    = compute_rho(ϕ_qe_raw,    ϕ)[2],
        qe_wf     = compute_rho(ϕ_qe_wf,     ϕ)[2],
        map_joint = compute_rho(ϕ_map_joint,  ϕ)[2],
        map_marg  = compute_rho(ϕ_map_marg,   ϕ)[2],
    )
    NL = (;
        qe_raw    = compute_NL(ϕ_qe_raw,    ϕ)[2],
        qe_wf     = compute_NL(ϕ_qe_wf,     ϕ)[2],
        map_joint = compute_NL(ϕ_map_joint,  ϕ)[2],
        map_marg  = compute_NL(ϕ_map_marg,   ϕ)[2],
    )

    return (;ℓ, rho, NL)
end
