using CMBLensing
using LinearAlgebra
using Statistics: mean, std
using JLD2
using PythonPlot
const plt_em = PythonPlot


"""
    run_error_analysis(Cℓ, Cℓn, θpix, Nside, pol, bandpass_mask; ...)

Run `nsims` simulations and compute:

1. **Noise curves**:
   N_L = < C_ℓ^{ϕ_rec, ϕ_rec} - C_ℓ^{ϕ_input, ϕ_input} >  averaged over sims
   for QE (wiener_filtered=false), MAP_joint (WL-debiased), and MAP_marg (WL-debiased).

2. **Mean squared error maps** (pixelwise):
   <|ϕ_true - ϕ_rec|²> averaged over sims.

Loads W_L from a normalization checkpoint and constructs Cℓs objects for debiasing.

Returns a NamedTuple and checkpoints after each sim.
"""


# -------------------------------------------------------------------
function run_error_analysis_all_debiased(
    Cℓ, Cℓn, θpix, Nside, pol, bandpass_mask;
    Δℓ::Int = 30,
    # NOTE: Cl_auto_*_sims are stored at the same Δℓ bins as the noise curves
    # (i.e. the same get_Cℓ call is reused — no separate Δℓ_kk parameter needed).
    checkpoint_file::String = "results/error_analysis_checkpoint_100.jld2",
    WL_checkpoint::String   = "results/WL_checkpoint_100.jld2",
    phi_maps_file::String   = "results/phi_maps.jld2",
    beamFWHM::Real = 1.0,
    progress::Bool = true,
    T::Type = Float64,
    grad_nbins::Int = 20,
    minW::Real = 1e-8,
    max_sims::Int = typemax(Int),
    raw_qe_phi_file::String = "",           # phi_maps_raw_qe.jld2 from run_raw_qe_sims.jl
    WL_raw_qe_checkpoint::String = "",      # WL_raw_qe_checkpoint.jld2
)

    # W_L is the empirical transfer function from normalization.jl:
    #   W_L = <C_L^{ϕ_true, ϕ_rec}> / <C_L^{ϕ_true, ϕ_true}>
    # Wrapping as Cℓs gives a callable interpolator so debias_phi_with_WL
    # can evaluate W_L on the 2D Fourier grid (each mode gets its own value)
    @load WL_checkpoint ℓ_template W_joint_running W_marg_running W_qe_running
    WL_joint_Cℓs = Cℓs(ℓ_template, W_joint_running)
    WL_marg_Cℓs  = Cℓs(ℓ_template, W_marg_running)
    WL_qe_Cℓs    = Cℓs(ℓ_template, W_qe_running)

    progress && println("Loaded W_L from $WL_checkpoint ($(length(ℓ_template)) bins)")
    progress && println("  using QE wiener_filtered=true + debias by W_qe_running")

    # Raw QE W_L (optional — only if run_raw_qe_sims.jl has been run)
    do_raw_qe = !isempty(raw_qe_phi_file)      && isfile(raw_qe_phi_file) &&
                !isempty(WL_raw_qe_checkpoint) && isfile(WL_raw_qe_checkpoint)
    WL_raw_qe_Cℓs = nothing
    if do_raw_qe
        # JLD2.load(file, "k1", "k2") returns a Tuple — unpack directly.
        # run_raw_qe_sims.jl saves the key as "ℓ_template" (not "ℓ_template_raw")
        d = JLD2.load(WL_raw_qe_checkpoint)
        WL_raw_qe_Cℓs = Cℓs(Float64.(d["ℓ_template"]), Float64.(d["W_qe_raw"]))
        progress && println("Loaded raw QE W_L from $WL_raw_qe_checkpoint")
    end

    # Noise curves: N_L = <C_L^{ϕ_rec, ϕ_rec}> - <C_L^{ϕ_true, ϕ_true}>
    # After debiasing by W_L the estimator is mean-unbiased, so any excess
    # auto-power over the truth is reconstruction noise.
    # Stored per-sim so convergence with nsims can be checked post-hoc.
    noise_joint_sims = Vector{Vector{Float64}}()
    noise_marg_sims  = Vector{Vector{Float64}}()
    noise_qe_sims    = Vector{Vector{Float64}}()
    ℓ_noise          = nothing

    # Pixel-space MSE: sum_s |ϕ_rec(x) - ϕ_true(x)|² / nsims
    # Kept as running sums to avoid storing all per-sim maps
    sum_Δϕ²_joint = zeros(Float64, Nside, Nside)
    sum_Δϕ²_marg  = zeros(Float64, Nside, Nside)
    sum_Δϕ²_qe    = zeros(Float64, Nside, Nside)
    nsims_completed = 0

    # MSE binned by |∇T| magnitude: tests whether reconstruction error
    # is spatially uniform or enhanced near large temperature gradients
    # (lensing signal is proportional to ∇T, so QE error correlates with it)
    grad_binned_qe_sims    = Vector{Vector{Float64}}()
    grad_binned_joint_sims = Vector{Vector{Float64}}()
    grad_binned_marg_sims  = Vector{Vector{Float64}}()
    grad_bin_cen           = nothing
    grad_counts_sims       = Vector{Vector{Int}}()   # pixel count per gradient bin per sim
    # Within-bin pixel std of Δϕ²: measures scatter of errors within each gradient bin
    # (complements the bin-mean; large std → high spatial variance within the bin)
    grad_binned_qe_stds    = Vector{Vector{Float64}}()
    grad_binned_joint_stds = Vector{Vector{Float64}}()
    grad_binned_marg_stds  = Vector{Vector{Float64}}()

    # Per-sim κ auto bandpowers at ΔL=Δℓ_kk (default 2000).
    # C_L^{κ̂κ̂}(s): raw auto-spectrum, NO noise subtraction — this is the quantity
    # whose std across sims gives σ[C_L^{κ̂κ̂}] as plotted in Fig. 3 of the GI paper.
    # φ→κ conversion applied: C_L^κκ = (L²/2)² × C_L^φφ.
    # (Cross σ is derived separately from the WL checkpoint R_sims × C_theory.)
    Cl_auto_qe_sims    = Vector{Vector{Float64}}()
    Cl_auto_joint_sims = Vector{Vector{Float64}}()
    Cl_auto_marg_sims  = Vector{Vector{Float64}}()
    ℓ_kk               = nothing

    # Per-sim κ cross bandpowers C_L^{κ_true, κ̂}(s) = (L²/2)² × C_L^{ϕ_true, ϕ̂_raw}(s)
    # Uses raw (pre-debiasing) φ̂. σ[C_L^{κ_true,κ̂}] is the right-panel of Fig. 3 in the paper.
    Cl_cross_qe_sims    = Vector{Vector{Float64}}()
    Cl_cross_joint_sims = Vector{Vector{Float64}}()
    Cl_cross_marg_sims  = Vector{Vector{Float64}}()

    # Per-sim cross-correlation coefficient ρ_L(s) = C_L^{ϕ_true,ϕ̂_raw}(s) / sqrt(C_L^{ϕ_true}(s)×C_L^{ϕ̂_raw}(s))
    # Uses raw (pre-debiasing) φ̂ — no W_L amplification, extends cleanly to full L range.
    # mean_s[ρ_L] ≈ W_L by construction; std_s[ρ_L] shows per-bin reconstruction quality spread.
    ρ_qe_sims    = Vector{Vector{Float64}}()
    ρ_joint_sims = Vector{Vector{Float64}}()
    ρ_marg_sims  = Vector{Vector{Float64}}()
    ℓ_ρ          = nothing

    # Raw QE arrays (populated only if do_raw_qe; always initialized so @save is unconditional)
    Cl_auto_raw_qe_sims  = Vector{Vector{Float64}}()
    Cl_cross_raw_qe_sims = Vector{Vector{Float64}}()
    ρ_raw_qe_sims        = Vector{Vector{Float64}}()

    # ---- Resume ----
    if isfile(checkpoint_file)
        @load checkpoint_file noise_joint_sims noise_marg_sims noise_qe_sims ℓ_noise sum_Δϕ²_joint sum_Δϕ²_marg sum_Δϕ²_qe nsims_completed
        try
            @load checkpoint_file grad_binned_qe_sims grad_binned_joint_sims grad_binned_marg_sims grad_bin_cen
        catch
        end
        try
            @load checkpoint_file grad_counts_sims
        catch
        end
        try
            @load checkpoint_file grad_binned_qe_stds grad_binned_joint_stds grad_binned_marg_stds
        catch
        end

        # Per-sim κ bandpowers: if absent (old checkpoint format) restart from scratch
        # so that all arrays remain in sync across sims.
        kk_ok = false
        try
            @load checkpoint_file Cl_auto_qe_sims Cl_auto_joint_sims Cl_auto_marg_sims ℓ_kk
            @load checkpoint_file Cl_cross_qe_sims Cl_cross_joint_sims Cl_cross_marg_sims
            @load checkpoint_file ρ_qe_sims ρ_joint_sims ρ_marg_sims ℓ_ρ
            if do_raw_qe
                @load checkpoint_file Cl_auto_raw_qe_sims Cl_cross_raw_qe_sims ρ_raw_qe_sims
            end
            # Check that ℓ_kk has the same number of bins as ℓ_noise.
            # A mismatch (different Δℓ) or missing arrays (old checkpoint format) → restart.
            if ℓ_noise !== nothing && length(ℓ_kk) != length(collect(ℓ_noise))
                progress && @warn "ℓ_kk has $(length(ℓ_kk)) bins but ℓ_noise has $(length(collect(ℓ_noise))) — restarting"
            else
                kk_ok = true
            end
        catch
        end
        if !kk_ok
            progress && @warn "κ bandpower arrays absent or incompatible in $checkpoint_file — restarting from sim 1"
            noise_joint_sims       = Vector{Vector{Float64}}()
            noise_marg_sims        = Vector{Vector{Float64}}()
            noise_qe_sims          = Vector{Vector{Float64}}()
            ℓ_noise                = nothing
            Cl_auto_qe_sims        = Vector{Vector{Float64}}()   # must reset: old entries
            Cl_auto_joint_sims     = Vector{Vector{Float64}}()   # may have different bin count
            Cl_auto_marg_sims      = Vector{Vector{Float64}}()   # than what new code produces
            ℓ_kk                   = nothing
            ρ_qe_sims              = Vector{Vector{Float64}}()   # reset alongside Cl_auto
            ρ_joint_sims           = Vector{Vector{Float64}}()
            ρ_marg_sims            = Vector{Vector{Float64}}()
            ℓ_ρ                    = nothing
            Cl_cross_qe_sims       = Vector{Vector{Float64}}()
            Cl_cross_joint_sims    = Vector{Vector{Float64}}()
            Cl_cross_marg_sims     = Vector{Vector{Float64}}()
            Cl_auto_raw_qe_sims    = Vector{Vector{Float64}}()
            Cl_cross_raw_qe_sims   = Vector{Vector{Float64}}()
            ρ_raw_qe_sims          = Vector{Vector{Float64}}()
            sum_Δϕ²_joint          = zeros(Float64, Nside, Nside)
            sum_Δϕ²_marg           = zeros(Float64, Nside, Nside)
            sum_Δϕ²_qe             = zeros(Float64, Nside, Nside)
            grad_binned_qe_sims    = Vector{Vector{Float64}}()
            grad_binned_joint_sims = Vector{Vector{Float64}}()
            grad_binned_marg_sims  = Vector{Vector{Float64}}()
            grad_bin_cen           = nothing
            grad_counts_sims       = Vector{Vector{Int}}()
            grad_binned_qe_stds    = Vector{Vector{Float64}}()
            grad_binned_joint_stds = Vector{Vector{Float64}}()
            grad_binned_marg_stds  = Vector{Vector{Float64}}()
            nsims_completed        = 0
        end
        progress && println("Resuming from $checkpoint_file ($nsims_completed completed)")
    end

    # ---- Load pre-computed phi maps from normalization run (per-sim key format) ----
    ϕ_qe_maps    = Vector{Matrix{Float64}}()
    ϕ_joint_maps = Vector{Matrix{Float64}}()
    ϕ_marg_maps  = Vector{Matrix{Float64}}()
    phi_seeds    = Int[]
    jldopen(phi_maps_file, "r") do f
        idxs = sort([parse(Int, m.captures[1])
                     for key in keys(f)
                     for m in [match(r"^sim_(\d+)$", key)]
                     if m !== nothing])
        for idx in idxs
            push!(phi_seeds,    read(f, "sim_$idx/seed"))
            push!(ϕ_qe_maps,   read(f, "sim_$idx/ϕ_qe"))
            push!(ϕ_joint_maps, read(f, "sim_$idx/ϕ_joint"))
            push!(ϕ_marg_maps,  read(f, "sim_$idx/ϕ_marg"))
        end
    end
    nsims = min(length(phi_seeds), max_sims)
    progress && println("Loaded phi maps: $nsims sims from $phi_maps_file (max_sims=$max_sims)")

    # Load raw QE phi maps (one entry per sim; nothing if that sim is missing)
    ϕ_raw_qe_maps = Vector{Union{Matrix{Float64}, Nothing}}(nothing, nsims)
    if do_raw_qe
        jldopen(raw_qe_phi_file, "r") do f
            for i in 1:nsims
                key = "sim_$i/ϕ_qe_raw"
                if haskey(f, key)
                    ϕ_raw_qe_maps[i] = read(f, key)
                end
            end
        end
        n_raw = count(!isnothing, ϕ_raw_qe_maps)
        progress && println("Loaded $n_raw / $nsims raw QE phi maps from $raw_qe_phi_file")
    end

    start_from = nsims_completed + 1

    for i in start_from:nsims
        seed = phi_seeds[i]
        progress && (println("→ Sim $i / $nsims  (seed=$seed)"); flush(stdout))

        # load_sim with the same seed gives the same ϕ_true and unlensed CMB f.
        # No reconstruction is done here — ϕ_rec maps come from phi_maps_file.
        (; f, ϕ) = load_sim(
            seed = seed,
            Cℓ = Cℓ, Cℓn = Cℓn,
            θpix = θpix, T = T, Nside = Nside,
            beamFWHM = beamFWHM, pol = pol,
            bandpass_mask = bandpass_mask,
            pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )

        # Reconstruction arrays were saved as raw Float64 matrices.
        # Wrap them as CMBLensing Map fields (pixel-space) reusing ϕ's projection
        # metadata (grid spacing, Nside, etc.) so downstream CMBLensing operations work.
        ϕ_ref   = Map(ϕ)
        ϕ_qe_field   = typeof(ϕ_ref)(Float64.(ϕ_qe_maps[i]),    ϕ_ref.metadata)
        ϕ_joint_field = typeof(ϕ_ref)(Float64.(ϕ_joint_maps[i]), ϕ_ref.metadata)
        ϕ_marg_field  = typeof(ϕ_ref)(Float64.(ϕ_marg_maps[i]),  ϕ_ref.metadata)

        # Debiasing: ϕ_deb(k) = ϕ_rec(k) / W_L(|k|)
        # Removes the multiplicative transfer-function bias so that
        # <C_L^{ϕ_deb, ϕ_true}> = C_L^{ϕ_true}. minW floors the division to
        # avoid amplifying noise at modes where W_L → 0 (near and beyond lmax).
        ϕ_qe_deb    = debias_phi_with_WL(ϕ_qe_field,    WL_qe_Cℓs;    minW=minW)
        ϕ_joint_deb = debias_phi_with_WL(ϕ_joint_field, WL_joint_Cℓs; minW=minW)
        ϕ_marg_deb  = debias_phi_with_WL(ϕ_marg_field,  WL_marg_Cℓs;  minW=minW)

        # N_L per sim: N_L^{(s)} = C_L^{ϕ_rec,ϕ_rec}(s) - C_L^{ϕ_true,ϕ_true}(s)
        # Subtracting per-sim (not the ensemble mean) removes cosmic variance
        # in C_L^{ϕϕ} from the noise estimate, matching the Eq. (A1) convention
        cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)
        if ℓ_noise === nothing
            ℓ_noise = cl_tt.ℓ
        end

        cl_rr_qe    = get_Cℓ(ϕ_qe_deb;    Δℓ=Δℓ)
        cl_rr_joint = get_Cℓ(ϕ_joint_deb; Δℓ=Δℓ)
        cl_rr_marg  = get_Cℓ(ϕ_marg_deb;  Δℓ=Δℓ)

        push!(noise_qe_sims,    cl_rr_qe.Cℓ    .- cl_tt.Cℓ)
        push!(noise_joint_sims, cl_rr_joint.Cℓ .- cl_tt.Cℓ)
        push!(noise_marg_sims,  cl_rr_marg.Cℓ  .- cl_tt.Cℓ)

        # Per-sim κ auto bandpowers — reuse cl_rr_* already computed above.
        # C_L^{κ̂κ̂}(s) = (L²/2)² × C_L^{ϕ̂,ϕ̂}(s)  at the same Δℓ bins as the noise curves.
        # Storing the RAW auto-spectrum (no noise subtraction) means
        # std_s[Cl_auto_qe_sims] = std_s[C_L^{κ̂κ̂}(s)], the correct Fig. 3 quantity.
        if ℓ_kk === nothing
            ℓ_kk = Float64.(collect(cl_rr_qe.ℓ))   # same ℓ grid as noise curves
        end
        # φ→κ: C_L^κκ = (L²/2)² × C_L^φφ  (flat-sky approximation, exact in 2D Fourier)
        kfac = @. (Float64(cl_rr_qe.ℓ)^2 / 2)^2
        push!(Cl_auto_qe_sims,    kfac .* Float64.(cl_rr_qe.Cℓ))
        
        push!(Cl_auto_joint_sims, kfac .* Float64.(cl_rr_joint.Cℓ))
        push!(Cl_auto_marg_sims,  kfac .* Float64.(cl_rr_marg.Cℓ))

        # Cross-correlation coefficient ρ_L using RAW (pre-debiasing) φ̂.
        # ρ_L(s) = C_L^{ϕ_true, ϕ̂_raw}(s) / sqrt(C_L^{ϕ_true}(s) × C_L^{ϕ̂_raw}(s))
        # Why raw? Avoids the 1/W_L² noise amplification of the debiased auto-spectrum.
        # At high L where W_L→0, the debiased σ blows up; ρ_L stays bounded in [-1,1].
        # Note: mean_s[ρ_L] ≈ W_L by construction (cross/auto ≈ response function).
        cl_raw_qe    = get_Cℓ(ϕ_qe_field;    Δℓ=Δℓ)   # C_L^{ϕ̂_qe,raw}
        cl_raw_joint = get_Cℓ(ϕ_joint_field; Δℓ=Δℓ)   # C_L^{ϕ̂_joint,raw}
        cl_raw_marg  = get_Cℓ(ϕ_marg_field;  Δℓ=Δℓ)   # C_L^{ϕ̂_marg,raw}
        # Raw cross — for ρ_L only (mean ≈ W_L × C^ϕϕ, NOT on paper σ scale)
        cl_cross_qe    = get_Cℓ(ϕ, ϕ_qe_field;    Δℓ=Δℓ)
        cl_cross_joint = get_Cℓ(ϕ, ϕ_joint_field; Δℓ=Δℓ)
        cl_cross_marg  = get_Cℓ(ϕ, ϕ_marg_field;  Δℓ=Δℓ)
        # Debiased cross — for σ[C_L^{κκ̂}]: mean ≈ C^ϕϕ, matches paper normalization
        cl_cross_qe_deb    = get_Cℓ(ϕ, ϕ_qe_deb;    Δℓ=Δℓ)
        cl_cross_joint_deb = get_Cℓ(ϕ, ϕ_joint_deb; Δℓ=Δℓ)
        cl_cross_marg_deb  = get_Cℓ(ϕ, ϕ_marg_deb;  Δℓ=Δℓ)
        if ℓ_ρ === nothing
            ℓ_ρ = Float64.(collect(cl_raw_qe.ℓ))
        end
        # denom: sqrt(C_tt × C_raw). Guard against negative numerical noise with max(.,0).
        denom_qe    = sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_raw_qe.Cℓ),    0.0))
        denom_joint = sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_raw_joint.Cℓ), 0.0))
        denom_marg  = sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_raw_marg.Cℓ),  0.0))
        push!(ρ_qe_sims,    clamp.(Float64.(cl_cross_qe.Cℓ)    ./ max.(denom_qe,    1e-30), -1.0, 1.0))
        push!(ρ_joint_sims, clamp.(Float64.(cl_cross_joint.Cℓ) ./ max.(denom_joint, 1e-30), -1.0, 1.0))
        push!(ρ_marg_sims,  clamp.(Float64.(cl_cross_marg.Cℓ)  ./ max.(denom_marg,  1e-30), -1.0, 1.0))

        # Cross κ bandpowers using DEBIASED phi: C_L^{κ_true, κ̂_deb}(s)
        # mean ≈ C_L^{κκ} (unbiased); std gives σ[C_L^{κκ̂}] on paper scale.
        kfac_cross = @. (Float64(cl_cross_qe_deb.ℓ)^2 / 2)^2
        push!(Cl_cross_qe_sims,    kfac_cross .* Float64.(cl_cross_qe_deb.Cℓ))
        push!(Cl_cross_joint_sims, kfac_cross .* Float64.(cl_cross_joint_deb.Cℓ))
        push!(Cl_cross_marg_sims,  kfac_cross .* Float64.(cl_cross_marg_deb.Cℓ))

        # Raw QE (wiener_filtered=false): debias + auto-spectrum + ρ
        # W_L for raw QE is the QE normalization A_L (stays non-zero at high L,
        # unlike WF QE whose W_L → 0 beyond the bandpass). This is the estimator
        # plotted in Millea & Farren 2021 Fig. 3.
        if do_raw_qe && ϕ_raw_qe_maps[i] !== nothing
            ϕ_raw_qe_field = typeof(ϕ_ref)(Float64.(ϕ_raw_qe_maps[i]), ϕ_ref.metadata)
            cl_rr_raw_qe   = get_Cℓ(ϕ_raw_qe_field; Δℓ=Δℓ)
            kfac_raw       = @. (Float64(cl_rr_raw_qe.ℓ)^2 / 2)^2
            push!(Cl_auto_raw_qe_sims, kfac_raw .* Float64.(cl_rr_raw_qe.Cℓ))
            # raw QE cross (no debiasing — QE normalization A_L already applied → W_L ≈ 1)
            cl_cross_raw_qe = get_Cℓ(ϕ, ϕ_raw_qe_field; Δℓ=Δℓ)
            kfac_cross_rqe  = @. (Float64(cl_cross_raw_qe.ℓ)^2 / 2)^2
            push!(Cl_cross_raw_qe_sims, kfac_cross_rqe .* Float64.(cl_cross_raw_qe.Cℓ))
            # ρ for raw QE
            cl_auto_rqe_raw = get_Cℓ(ϕ_raw_qe_field; Δℓ=Δℓ)
            denom_rqe = sqrt.(max.(Float64.(cl_tt.Cℓ) .* Float64.(cl_auto_rqe_raw.Cℓ), 0.0))
            push!(ρ_raw_qe_sims, clamp.(Float64.(cl_cross_raw_qe.Cℓ) ./ max.(denom_rqe, 1e-30), -1.0, 1.0))
        end

        # Pixel-space MSE: accumulate sum_s Δϕ(x)² for each estimator.
        # Dividing by nsims_completed at the end gives the per-pixel mean squared error.
        # This is NOT the same as N_L — it captures spatial variation in noise.
        ϕt = Map(ϕ).arr
        Δϕ_qe    = Float64.(Map(ϕ_qe_deb).arr    .- ϕt)
        Δϕ_joint = Float64.(Map(ϕ_joint_deb).arr .- ϕt)
        Δϕ_marg  = Float64.(Map(ϕ_marg_deb).arr  .- ϕt)

        sum_Δϕ²_qe    .+= Δϕ_qe.^2
        sum_Δϕ²_joint .+= Δϕ_joint.^2
        sum_Δϕ²_marg  .+= Δϕ_marg.^2

        # MSE vs |∇T|: bin pixels by local temperature gradient magnitude.
        # grad_fft returns (∂T/∂x, ∂T/∂y, |∇T|) on the unlensed CMB f.
        # bin_stat_err returns (bin centres, bin means, SEM, pixel counts).
        # SEM = σ/√N where N is pixels per bin; std = SEM × √N recovers the
        # within-bin spread of Δϕ² values (used to check bin homogeneity).
        _, _, grad_mag = grad_fft(f)

        cen, m_qe,    sem_qe,    counts_bin = bin_stat_err(Float64.(grad_mag), Δϕ_qe.^2;    nbins=grad_nbins)
        cen, m_joint, sem_joint, _          = bin_stat_err(Float64.(grad_mag), Δϕ_joint.^2; nbins=grad_nbins)
        cen, m_marg,  sem_marg,  _          = bin_stat_err(Float64.(grad_mag), Δϕ_marg.^2;  nbins=grad_nbins)

        # recover within-bin std from SEM: std = sem × √N
        n_pix = Float64.(counts_bin)
        std_qe    = sem_qe    .* sqrt.(n_pix)
        std_joint = sem_joint .* sqrt.(n_pix)
        std_marg  = sem_marg  .* sqrt.(n_pix)

        if grad_bin_cen === nothing; grad_bin_cen = cen; end
        push!(grad_binned_qe_sims,    m_qe)
        push!(grad_binned_joint_sims, m_joint)
        push!(grad_binned_marg_sims,  m_marg)
        push!(grad_counts_sims,       counts_bin)
        push!(grad_binned_qe_stds,    std_qe)
        push!(grad_binned_joint_stds, std_joint)
        push!(grad_binned_marg_stds,  std_marg)

        nsims_completed += 1

        @save checkpoint_file noise_joint_sims noise_marg_sims noise_qe_sims ℓ_noise sum_Δϕ²_joint sum_Δϕ²_marg sum_Δϕ²_qe nsims_completed grad_binned_qe_sims grad_binned_joint_sims grad_binned_marg_sims grad_bin_cen grad_counts_sims grad_binned_qe_stds grad_binned_joint_stds grad_binned_marg_stds Cl_auto_qe_sims Cl_auto_joint_sims Cl_auto_marg_sims ℓ_kk Cl_cross_qe_sims Cl_cross_joint_sims Cl_cross_marg_sims ρ_qe_sims ρ_joint_sims ρ_marg_sims ℓ_ρ Cl_auto_raw_qe_sims Cl_cross_raw_qe_sims ρ_raw_qe_sims
        progress && println("  Checkpoint saved ($nsims_completed completed)")
    end

    # ---- Final ensemble averages ----
    # N_L = mean over sims of per-sim N_L^{(s)}
    N_qe    = mean(reduce(hcat, noise_qe_sims);    dims=2)[:]
    N_joint = mean(reduce(hcat, noise_joint_sims); dims=2)[:]
    N_marg  = mean(reduce(hcat, noise_marg_sims);  dims=2)[:]

    # Pixel-space MSE maps: <|Δϕ(x)|²>_sims
    denom = max(nsims_completed, 1)
    mean_Δϕ²_qe    = sum_Δϕ²_qe    ./ denom
    mean_Δϕ²_joint = sum_Δϕ²_joint ./ denom
    mean_Δϕ²_marg  = sum_Δϕ²_marg  ./ denom

    # Gradient-binned MSE: matrix is (nbins × nsims), mean over sims gives curve
    M_qe    = reduce(hcat, grad_binned_qe_sims)
    M_joint = reduce(hcat, grad_binned_joint_sims)
    M_marg  = reduce(hcat, grad_binned_marg_sims)

    mse_vs_grad_qe    = mean(M_qe;    dims=2)[:]
    mse_vs_grad_joint = mean(M_joint; dims=2)[:]
    mse_vs_grad_marg  = mean(M_marg;  dims=2)[:]

    # SEM across sims: uncertainty on the mean gradient-binned MSE curve
    # SEM = std(sims) / √nsims — not the within-bin pixel scatter
    n = max(nsims_completed, 2)
    mse_vs_grad_sem_qe    = std(M_qe;    dims=2)[:] ./ sqrt(n)
    mse_vs_grad_sem_joint = std(M_joint; dims=2)[:] ./ sqrt(n)
    mse_vs_grad_sem_marg  = std(M_marg;  dims=2)[:] ./ sqrt(n)

    mean_grad_counts = mean(reduce(hcat, Float64.(c) for c in grad_counts_sims); dims=2)[:]

    # Within-bin pixel std averaged over sims — shows how homogeneous the error
    # distribution is within each gradient-magnitude bin
    Sq_qe    = reduce(hcat, grad_binned_qe_stds)
    Sq_joint = reduce(hcat, grad_binned_joint_stds)
    Sq_marg  = reduce(hcat, grad_binned_marg_stds)
    mse_vs_grad_std_qe    = mean(Sq_qe;    dims=2)[:]
    mse_vs_grad_std_joint = mean(Sq_joint; dims=2)[:]
    mse_vs_grad_std_marg  = mean(Sq_marg;  dims=2)[:]

    return (
        ℓ = ℓ_noise,
        N_qe    = N_qe,
        N_joint = N_joint,
        N_marg  = N_marg,
        mean_Δϕ²_qe    = mean_Δϕ²_qe,
        mean_Δϕ²_joint = mean_Δϕ²_joint,
        mean_Δϕ²_marg  = mean_Δϕ²_marg,
        grad_bin_cen          = grad_bin_cen,
        mse_vs_grad_qe        = mse_vs_grad_qe,
        mse_vs_grad_joint     = mse_vs_grad_joint,
        mse_vs_grad_marg      = mse_vs_grad_marg,
        mse_vs_grad_sem_qe    = mse_vs_grad_sem_qe,
        mse_vs_grad_sem_joint = mse_vs_grad_sem_joint,
        mse_vs_grad_sem_marg  = mse_vs_grad_sem_marg,
        mse_vs_grad_std_qe    = mse_vs_grad_std_qe,
        mse_vs_grad_std_joint = mse_vs_grad_std_joint,
        mse_vs_grad_std_marg  = mse_vs_grad_std_marg,
        mean_grad_counts      = mean_grad_counts,
        nsims = nsims_completed,
        Δℓ = Δℓ,
        ℓ_kk           = ℓ_kk,
        Cl_auto_qe_sims    = Cl_auto_qe_sims,
        Cl_auto_joint_sims = Cl_auto_joint_sims,
        Cl_auto_marg_sims  = Cl_auto_marg_sims,
        Cl_cross_qe_sims    = Cl_cross_qe_sims,
        Cl_cross_joint_sims = Cl_cross_joint_sims,
        Cl_cross_marg_sims  = Cl_cross_marg_sims,
        ρ_qe_sims    = ρ_qe_sims,
        ρ_joint_sims = ρ_joint_sims,
        ρ_marg_sims  = ρ_marg_sims,
        ℓ_ρ          = ℓ_ρ,
        Cl_auto_raw_qe_sims  = Cl_auto_raw_qe_sims,
        Cl_cross_raw_qe_sims = Cl_cross_raw_qe_sims,
        ρ_raw_qe_sims        = ρ_raw_qe_sims,
    )
end


"""
    plot_noise_curves_L4(ℓ, N_joint, N_marg, N_qe, Δℓ;
                         Nside=512, θpix=..., savepath="results/noise_curves_L4.png",
                         use_abs=true)

Two-panel plot:
- Top: (ℓ^4) × noise curves  vs ℓ  (log scale by default)
- Bottom: approximate number of Fourier modes per ℓ bin
The noise curves are multiplied by ℓ^4 to better visualise their shape and differences.
Negative bins (over-debiased) are shown dashed on the log scale.
"""
function plot_noise_curves_L4(ℓ, N_joint, N_marg, N_qe, Δℓ;
    Nside::Int = 512,
    θpix::Float64 = 0.7438046267475303,
    savepath::String = "results/noise_curves_L4.png",
    Lcut::Real = 12000,         # <-- add this
)

    # ... rcParams etc ...

    ℓv = Float64.(collect(ℓ))
    lmask = (ℓv .> 0) .& (ℓv .<= Lcut)   # <-- this is the ℓ mask

    ℓ4 = ℓv .^ 4
    ℓ4[ℓv .== 0.0] .= 0.0

    y_qe    = ℓ4 .* Float64.(N_qe)
    y_joint = ℓ4 .* Float64.(N_joint)
    y_marg  = ℓ4 .* Float64.(N_marg)

    θpix_rad = θpix * (π / 180) / 60
    L_box    = Nside * θpix_rad
    Δℓ_fund  = 2π / L_box
    mode_counts = @. 2π * ℓv * Δℓ / (Δℓ_fund^2)

    fig, (ax_top, ax_bot) = plt_em.subplots(
        2, 1, figsize=(7, 6),
        gridspec_kw=Dict("height_ratios" => [3, 1], "hspace" => 0.08),
        sharex=true
    )

    for (y, col, lab) in [(y_qe, "#4477AA", "QE (WF, debiased)"),
                          (y_joint, "#EE6677", "MAP joint (debiased)"),
                          (y_marg,  "#228833", "MAP marg (debiased)")]

        yy = y[lmask]
        ll = ℓv[lmask]
        pos = yy .> 0
        

        ax_top.semilogy(ll[pos], yy[pos],       color=col, label=lab,      linewidth=1.5)
    end

    ax_top.set_ylabel(L"L^4\,\langle C_L^{\hat\phi\hat\phi} - C_L^{\phi\phi} \rangle")
    
    ax_top.legend(frameon=false, fontsize=9)
    ax_top.tick_params(labelsize=11, labelbottom=false)

    ax_bot.bar(ℓv[lmask], mode_counts[lmask], width=0.8*Δℓ, color="grey", alpha=0.6, edgecolor="none")
    ax_bot.set_xlabel(L"L")
    ax_bot.set_ylabel(L"N_\mathrm{modes}", fontsize=11)
    ax_bot.tick_params(labelsize=10)

    ax_top.set_xlim(0, Lcut)   # <-- important: don’t show beyond cutoff

    fig.savefig(savepath, dpi=150)
    plt_em.plotclose("all")
    println("Saved ℓ^4 noise curve plot to $savepath (L ≤ $Lcut)")
end


"""
    save_error_results(results, Δℓ, Nside, θpix; results_file=..., plot_path=...)

Saves the combined (QE+joint+marg) results and makes the ℓ^4 noise curve plot.
"""
function save_error_results(
    results::NamedTuple,
    Δℓ::Int, Nside::Int, θpix::Float64;
    results_file::String = "results/error_analysis_final.jld2",
    plot_path::String = "results/noise_curves_L4.png",
)
    ℓ = results.ℓ
    N_joint = results.N_joint
    N_marg  = results.N_marg
    N_qe    = results.N_qe
    mean_Δϕ²_joint   = results.mean_Δϕ²_joint
    mean_Δϕ²_marg    = results.mean_Δϕ²_marg
    mean_Δϕ²_qe      = results.mean_Δϕ²_qe
    grad_bin_cen          = results.grad_bin_cen
    mse_vs_grad_qe        = results.mse_vs_grad_qe
    mse_vs_grad_joint     = results.mse_vs_grad_joint
    mse_vs_grad_marg      = results.mse_vs_grad_marg
    mse_vs_grad_sem_qe    = results.mse_vs_grad_sem_qe
    mse_vs_grad_sem_joint = results.mse_vs_grad_sem_joint
    mse_vs_grad_sem_marg  = results.mse_vs_grad_sem_marg
    mse_vs_grad_std_qe    = results.mse_vs_grad_std_qe
    mse_vs_grad_std_joint = results.mse_vs_grad_std_joint
    mse_vs_grad_std_marg  = results.mse_vs_grad_std_marg
    mean_grad_counts      = results.mean_grad_counts
    nsims                 = results.nsims

    @save results_file ℓ N_joint N_marg N_qe mean_Δϕ²_joint mean_Δϕ²_marg mean_Δϕ²_qe grad_bin_cen mse_vs_grad_qe mse_vs_grad_joint mse_vs_grad_marg mse_vs_grad_sem_qe mse_vs_grad_sem_joint mse_vs_grad_sem_marg mse_vs_grad_std_qe mse_vs_grad_std_joint mse_vs_grad_std_marg mean_grad_counts nsims
    println("Saved final results to $results_file")

    plot_noise_curves_L4(ℓ, N_joint, N_marg, N_qe, Δℓ;
        Nside=Nside, θpix=θpix, savepath=plot_path)
end
