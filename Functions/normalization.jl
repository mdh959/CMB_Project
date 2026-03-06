using CMBLensing
using LinearAlgebra
using Statistics: mean
using JLD2: @save, @load


# ──────────────────────────────────────────────────────────────────
# Empirical W_L from Monte Carlo simulations
# ──────────────────────────────────────────────────────────────────

"""
    empirical_WL_all_loadsim(Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
                            nsims=50, Δℓ=50, seed0=1000, beamFWHM=1.0,
                            qe_wiener_filtered=true,
                            checkpoint_file="results/WL_checkpoint.jld2",
                            phi_maps_file="results/phi_maps.jld2")

Empirical normalization curves W_L for QE, MAP_joint, MAP_marg:

    W_L = < C_L^{ϕ_true, ϕ_hat} / C_L^{ϕ_true, ϕ_true} >_sims

Uses `get_Cℓ` (binned by Δℓ) for all spectra.

Checkpoint stores per-sim ratio curves (R_*_sims), running means (W_*_running),
ℓ_template, nsims_completed, seeds_done.

phi_maps_file stores the raw pixel-space phi arrays (Float64 matrices) for all sims
so that error_mean.jl can load them directly without re-running MAP. Seeds are also
saved so error_mean.jl knows which load_sim seeds to use for the gradient field f.
"""

# Define serial fallback only once (if the installed CMBLensing lacks these)
if !isdefined(CMBLensing, :set_distributed_dataset)
    @eval CMBLensing begin
        const _DIST_DS = Ref{Any}(nothing)
        set_distributed_dataset(x) = (_DIST_DS[] = x)
        get_distributed_dataset() = _DIST_DS[]
    end
end

function empirical_WL_all_loadsim(
    Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
    nsims::Int = 50,
    Δℓ::Int = 50,
    seed0::Int = 1000,
    beamFWHM::Real = 1.0,
    qe_wiener_filtered::Bool = true,
    checkpoint_file::String = "results/WL_checkpoint.jld2",
    phi_maps_file::String   = "results/phi_maps.jld2",
)

    # Float64 throughout: at low noise ϕ values at high ℓ are O(1e-6), Float32 loses
    # precision in the cross-spectrum ratios R_L
    T = Float64

    # Per-sim cross/auto ratio vectors; W_* are their running means
    # R_L^{(s)} = C_L^{ϕ_true, ϕ_rec}(s) / C_L^{ϕ_true, ϕ_true}(s)
    # W_L = <R_L>_sims  → transfer function / response of each estimator
    R_joint_sims = Vector{Vector{Float64}}()
    R_marg_sims  = Vector{Vector{Float64}}()
    R_qe_sims    = Vector{Vector{Float64}}()
    ℓ_template   = nothing
    W_joint_running = nothing
    W_marg_running  = nothing
    W_qe_running    = nothing
    nsims_completed = 0
    last_s_attempted = 0   # tracks loop index s, not just successful completions
    seeds_done = Int[]

    # --- resume from checkpoint ---
    if isfile(checkpoint_file)
        println("Resuming from checkpoint: $checkpoint_file")
        jldopen(checkpoint_file, "r") do f
            # haskey guards make this robust to checkpoints written by older code versions
            if haskey(f, "R_joint_sims");      R_joint_sims = read(f, "R_joint_sims"); end
            if haskey(f, "R_marg_sims");       R_marg_sims  = read(f, "R_marg_sims");  end
            if haskey(f, "R_qe_sims");         R_qe_sims    = read(f, "R_qe_sims");    end
            if haskey(f, "ℓ_template");        ℓ_template  = read(f, "ℓ_template");   end
            if haskey(f, "W_joint_running");   W_joint_running = read(f, "W_joint_running"); end
            if haskey(f, "W_marg_running");    W_marg_running  = read(f, "W_marg_running");  end
            if haskey(f, "W_qe_running");      W_qe_running    = read(f, "W_qe_running");    end
            if haskey(f, "nsims_completed");   nsims_completed = read(f, "nsims_completed"); end
            if haskey(f, "last_s_attempted");  last_s_attempted = read(f, "last_s_attempted"); end
            if haskey(f, "seeds_done");        seeds_done = read(f, "seeds_done"); end
        end
        # old checkpoints didn't save last_s_attempted; fall back to success count
        if last_s_attempted == 0
            last_s_attempted = nsims_completed
        end
        println("→ Successes = $nsims_completed; last s attempted = $last_s_attempted; next s = $(last_s_attempted + 1)")
    end

    # --- check which phi maps are already saved (per-sim key format) ---
    # phi maps are written to a separate file (append-only, ~8 MB per sim) so that
    # error_mean.jl can load precomputed reconstructions without re-running MAP
    phi_sims_done = Set{Int}()
    if isfile(phi_maps_file)
        try
            jldopen(phi_maps_file, "r") do f
                for key in keys(f)
                    m = match(r"^sim_(\d+)$", key)
                    if m !== nothing
                        push!(phi_sims_done, parse(Int, m.captures[1]))
                    end
                end
            end
            println("→ $(length(phi_sims_done)) phi maps already saved in $phi_maps_file")
        catch err
            @warn "phi_maps_file appears corrupted ($err) — deleting and starting fresh"
            rm(phi_maps_file)
        end
    end

    start_from = last_s_attempted + 1   # skip sims already attempted (including failed ones)

    for s in start_from:nsims
        seed = seed0 + s
        println("→ Simulation $s / $nsims   (seed=$seed)")
        last_s_attempted = s   # write before anything that can throw, so resume skips this s

        (; ϕ, ds) = load_sim(
            seed=seed,
            Cℓ=Cℓ, Cℓn=Cℓn,
            θpix=θpix, T=T, Nside=Nside,
            beamFWHM=beamFWHM, pol=pol,
            bandpass_mask=bandpass_mask,
            pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )

        # --- reconstructions ---
        # QE starting point; used as ϕstart for both MAP methods
        qe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=qe_wiener_filtered)
        ϕqe = qe.ϕqe

        # MAP_marg: wrapped in try/catch because at very low noise the mean-field
        # gradient can fail if CG doesn't converge — skip the sim rather than crash
        ϕ_marg = nothing
        try
            ϕ_marg, _ = MAP_marg(
                ds; ϕstart=ϕqe,
                nsteps=15, progress=false,
                conjgrad_kwargs=(tol=1e-3, nsteps=100),
                pmap=map,
            )
        catch err
            println("  MAP_marg failed: $err — skipping this sim (no checkpoint increment)")
            @save checkpoint_file R_qe_sims R_joint_sims R_marg_sims ℓ_template W_qe_running W_joint_running W_marg_running nsims_completed last_s_attempted seeds_done
            continue
        end

        # MAP_joint: nburnin_update_hessian deliberately omitted — the secant Hessian
        # update (Δϕ/Δg in Fourier space) produces all-NaN Cℓ at low noise because
        # step sizes α are tiny and gradient differences are numerically near zero
        ϕ_joint = nothing
        try
            Ωstart = FieldTuple(ϕ=ϕqe)
            result = MAP_joint(
                ds, Ωstart;
                nsteps=15, progress=false,
                conjgrad_kwargs=(tol=1e-3, nsteps=100),
            )
            ϕ_joint = result.ϕ
        catch err
            println("  MAP_joint failed: $err — skipping this sim (no checkpoint increment)")
            @save checkpoint_file R_qe_sims R_joint_sims R_marg_sims ℓ_template W_qe_running W_joint_running W_marg_running nsims_completed last_s_attempted seeds_done
            continue
        end

        # --- spectra: C_L^{ϕ_true, ϕ_rec} and C_L^{ϕ_true, ϕ_true} ---
        # Δℓ binning averages over ~Δℓ modes; wide enough to reduce scatter but narrow
        # enough to resolve the scale-dependent W_L shape
        cl_tt = get_Cℓ(ϕ; Δℓ=Δℓ)           # true auto:  C_L^{ϕϕ}
        cl_tq = get_Cℓ(ϕ, ϕqe; Δℓ=Δℓ)       # true×QE:   C_L^{ϕ ϕ_QE}
        cl_tj = get_Cℓ(ϕ, ϕ_joint; Δℓ=Δℓ)   # true×joint: C_L^{ϕ ϕ_J}
        cl_tm = get_Cℓ(ϕ, ϕ_marg;  Δℓ=Δℓ)   # true×marg:  C_L^{ϕ ϕ_M}

        if ℓ_template === nothing
            ℓ_template = cl_tt.ℓ
        end

        # R_L^{(s)} = C_L^{ϕ_true, ϕ_rec}(s) / C_L^{ϕ_true, ϕ_true}(s)
        # Dividing per-sim (rather than averaging numerator and denominator separately)
        # removes cosmic variance in C_L^{ϕϕ} from the ratio at no extra cost
        denom = cl_tt.Cℓ
        push!(R_qe_sims,    cl_tq.Cℓ ./ denom)
        push!(R_joint_sims, cl_tj.Cℓ ./ denom)
        push!(R_marg_sims,  cl_tm.Cℓ ./ denom)

        # running means — converge to W_L as nsims grows
        W_qe_running    = mean(reduce(hcat, R_qe_sims);    dims=2)[:]
        W_joint_running = mean(reduce(hcat, R_joint_sims); dims=2)[:]
        W_marg_running  = mean(reduce(hcat, R_marg_sims);  dims=2)[:]

        nsims_completed = length(R_joint_sims)
        push!(seeds_done, seed)

        # Append pixel-space maps to the phi file (per-sim group key).
        # "a+" mode appends without reading the whole file — important for large nsims.
        if s ∉ phi_sims_done
            jldopen(phi_maps_file, "a+") do f
                f["sim_$s/ϕ_true"]  = Float64.(Map(ϕ).arr)
                f["sim_$s/ϕ_qe"]    = Float64.(Map(ϕqe).arr)
                f["sim_$s/ϕ_joint"] = Float64.(Map(ϕ_joint).arr)
                f["sim_$s/ϕ_marg"]  = Float64.(Map(ϕ_marg).arr)
                f["sim_$s/seed"]    = seed
            end
            push!(phi_sims_done, s)
        end

        # checkpoint
        @save checkpoint_file R_qe_sims R_joint_sims R_marg_sims ℓ_template W_qe_running W_joint_running W_marg_running nsims_completed last_s_attempted seeds_done
        println("  Checkpoint saved ($nsims_completed successful sims, last s=$last_s_attempted)")
    end

    return (
        ℓ = ℓ_template,
        WL_qe    = W_qe_running,
        WL_joint = W_joint_running,
        WL_marg  = W_marg_running,
        nsims    = nsims_completed,
        Δℓ = Δℓ,
    )
end
