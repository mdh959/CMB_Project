using CMBLensing
using LinearAlgebra
using Statistics: mean
using JLD2: @save, @load
using PythonPlot
const plt_em = PythonPlot
using Statistics: median



"""
    run_error_analysis(Cℓ, Cℓn, θpix, Nside, pol, bandpass_mask; ...)

Run `nsims` simulations and compute:

1. **Noise curves**:
   N_L = < C_ℓ^{ϕ_rec, ϕ_rec} - C_ℓ^{ϕ_input, ϕ_input} >  averaged over sims
   for QE (wiener_filtered=false), MAP_joint (WL-debiased), and MAP_marg (WL-debiased).

2. **Mean squared error maps** (pixelwise):
   <|ϕ_true - ϕ_rec|²> averaged over sims.

Loads W_L from a normalization checkpoint and constructs Cℓs objects for debiasing.

QE uses `wiener_filtered=false` so it is already correctly normalized.
MAP estimates are debiased by dividing by W_L in Fourier space.

Returns a NamedTuple and checkpoints after each sim.
"""


function run_error_analysis(
    Cℓ, Cℓn, θpix, Nside, pol, bandpass_mask;
    nsims::Int = 100,
    Δℓ::Int = 50,
    checkpoint_file::String = "results/error_analysis_checkpoint.jld2",
    WL_checkpoint::String = "results/WL_checkpoint.jld2",
    seed0::Int = 1000,
    beamFWHM::Real = 1.0,
    progress::Bool = true,
    T::Type = Float64,
    grad_nbins::Int = 20,
)

    # ── Load W_L from normalization checkpoint ──
    @load WL_checkpoint ℓ_template W_joint_running W_marg_running
    WL_joint_Cℓs = Cℓs(ℓ_template, W_joint_running)
    WL_marg_Cℓs  = Cℓs(ℓ_template, W_marg_running)
    progress && println("Loaded W_L from $WL_checkpoint ($(length(ℓ_template)) bins)")

    # ── Accumulators ──
    # Noise curves: accumulate (C_ℓ^{rec,rec} - C_ℓ^{true,true}) per sim
    noise_joint_sims = Vector{Vector{Float64}}()
    noise_marg_sims  = Vector{Vector{Float64}}()
    noise_qe_sims    = Vector{Vector{Float64}}()
    ℓ_noise          = nothing

    # MSE maps
    sum_Δϕ²_joint = zeros(Float64, Nside, Nside)
    sum_Δϕ²_marg  = zeros(Float64, Nside, Nside)
    sum_Δϕ²_qe    = zeros(Float64, Nside, Nside)
    nsims_completed = 0

    # MSE vs |∇T| binned curves (per-sim, then averaged)
    grad_binned_qe_sims    = Vector{Vector{Float64}}()
    grad_binned_joint_sims = Vector{Vector{Float64}}()
    grad_binned_marg_sims  = Vector{Vector{Float64}}()
    grad_bin_cen           = nothing

    # ── Resume from checkpoint ──
    if isfile(checkpoint_file)
        @load checkpoint_file noise_joint_sims noise_marg_sims noise_qe_sims ℓ_noise sum_Δϕ²_joint sum_Δϕ²_marg sum_Δϕ²_qe nsims_completed
        try @load checkpoint_file grad_binned_qe_sims grad_binned_joint_sims grad_binned_marg_sims grad_bin_cen catch end
        progress && println("Resuming from $checkpoint_file ($nsims_completed completed)")
    end
    start_from = nsims_completed + 1

    for s in start_from:nsims
        progress && (println("→ Simulation $s / $nsims"); flush(stdout))

        (; f, f̃, ϕ, ds) = load_sim(
            seed = seed0 + s,
            Cℓ = Cℓ, Cℓn = Cℓn,
            θpix = θpix, T = T, Nside = Nside,
            beamFWHM = beamFWHM, pol = pol,
            bandpass_mask = bandpass_mask,
            pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )

        # ── QE (no Wiener filter — already correctly normalized) ──
        qe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=false)
        ϕ_qe = qe.ϕqe

        # ── Wiener-filtered QE as starting point for MAP ──
        qe_start = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true)
        ϕqe_start = qe_start.ϕqe

        # MAP_marg
        ϕ_marg = nothing
        try
            ϕ_marg, _ = MAP_marg(
                ds; ϕstart= ϕqe_start,
                nsteps=15, progress=false,
                pmap = map,
                conjgrad_kwargs=(tol=1e-3, nsteps=50),
            )
        catch err
            println("  MAP_marg failed: $err — skipping sim $s")
            continue
        end

        # MAP_joint
        Ωstart = FieldTuple(ϕ=ϕqe_start)
        ϕ_joint = nothing
        try
            result = MAP_joint(
                ds, Ωstart;
                nsteps=15, progress=false,
                conjgrad_kwargs=(tol=1e-3, nsteps=50),
            )
            ϕ_joint = result.ϕ
        catch err
            println("  MAP_joint failed: $err — skipping sim $s")
            continue
        end

        # ── WL debias MAP estimates ──
        ϕ_joint_deb = debias_phi_with_WL(ϕ_joint, WL_joint_Cℓs)
        ϕ_marg_deb  = ϕ_marg !== nothing ? debias_phi_with_WL(ϕ_marg, WL_marg_Cℓs) : nothing


        # ── Noise curves: C_ℓ^{rec,rec} - C_ℓ^{true,true} ──
        cl_rr_joint = get_Cℓ(ϕ_joint_deb; Δℓ=Δℓ)
        cl_rr_qe    = get_Cℓ(ϕ_qe;        Δℓ=Δℓ)
        cl_tt       = get_Cℓ(ϕ;            Δℓ=Δℓ)

        if ℓ_noise === nothing
            ℓ_noise = cl_tt.ℓ
        end

        push!(noise_joint_sims, cl_rr_joint.Cℓ .- cl_tt.Cℓ)
        push!(noise_qe_sims,    cl_rr_qe.Cℓ    .- cl_tt.Cℓ)
        if ϕ_marg_deb !== nothing
            cl_rr_marg = get_Cℓ(ϕ_marg_deb; Δℓ=Δℓ)
            push!(noise_marg_sims, cl_rr_marg.Cℓ .- cl_tt.Cℓ)
        end

        # ── MSE maps ──

        ϕt = Map(ϕ).arr
        Δϕ_joint = Float64.(Map(ϕ_joint_deb).arr .- ϕt)
        Δϕ_qe    = Float64.(Map(ϕ_qe).arr        .- ϕt)

        sum_Δϕ²_joint .+= Δϕ_joint .^ 2
        sum_Δϕ²_qe    .+= Δϕ_qe    .^ 2
        if ϕ_marg_deb !== nothing
            Δϕ_marg = Float64.(Map(ϕ_marg_deb).arr .- ϕt)
            sum_Δϕ²_marg .+= Δϕ_marg .^ 2
        end

        # ── MSE vs |∇T| binned curves (per-sim) ──
        _, _, grad_mag = grad_fft(f)
        cen, m_qe,    _, _ = bin_stat_err(Float64.(grad_mag), Δϕ_qe.^2;    nbins=grad_nbins)
        cen, m_joint, _, _ = bin_stat_err(Float64.(grad_mag), Δϕ_joint.^2; nbins=grad_nbins)
        if grad_bin_cen === nothing; grad_bin_cen = cen; end
        push!(grad_binned_qe_sims,    m_qe)
        push!(grad_binned_joint_sims, m_joint)
        if ϕ_marg_deb !== nothing
            _, m_marg, _, _ = bin_stat_err(Float64.(grad_mag), Δϕ_marg.^2; nbins=grad_nbins)
            push!(grad_binned_marg_sims, m_marg)
        end

        nsims_completed += 1

        # ── Checkpoint ──
        @save checkpoint_file noise_joint_sims noise_marg_sims noise_qe_sims ℓ_noise sum_Δϕ²_joint sum_Δϕ²_marg sum_Δϕ²_qe nsims_completed grad_binned_qe_sims grad_binned_joint_sims grad_binned_marg_sims grad_bin_cen
        progress && println("  Checkpoint saved ($nsims_completed completed)")
    end

    # ── Average noise curves ──
    N_joint = mean(reduce(hcat, noise_joint_sims); dims=2)[:]
    N_qe    = mean(reduce(hcat, noise_qe_sims);    dims=2)[:]
    N_marg  = isempty(noise_marg_sims) ? fill(NaN, length(N_joint)) :
              mean(reduce(hcat, noise_marg_sims); dims=2)[:]

    # ── Average MSE maps ──
    denom = max(nsims_completed, 1)
    mean_Δϕ²_joint = sum_Δϕ²_joint ./ denom
    mean_Δϕ²_qe    = sum_Δϕ²_qe    ./ denom
    nmarg = length(noise_marg_sims)
    mean_Δϕ²_marg  = nmarg > 0 ? sum_Δϕ²_marg ./ nmarg : fill(NaN, size(sum_Δϕ²_marg))

    # ── Average MSE vs |∇T| binned curves ──
    mse_vs_grad_qe    = mean(reduce(hcat, grad_binned_qe_sims);    dims=2)[:]
    mse_vs_grad_joint = mean(reduce(hcat, grad_binned_joint_sims); dims=2)[:]
    mse_vs_grad_marg  = isempty(grad_binned_marg_sims) ? fill(NaN, length(mse_vs_grad_qe)) :
                        mean(reduce(hcat, grad_binned_marg_sims);  dims=2)[:]

    return (
        ℓ = ℓ_noise,
        N_joint = N_joint,
        N_marg  = N_marg,
        N_qe    = N_qe,
        mean_Δϕ²_joint    = mean_Δϕ²_joint,
        mean_Δϕ²_marg     = mean_Δϕ²_marg,
        mean_Δϕ²_qe       = mean_Δϕ²_qe,
        grad_bin_cen       = grad_bin_cen,
        mse_vs_grad_qe     = mse_vs_grad_qe,
        mse_vs_grad_joint  = mse_vs_grad_joint,
        mse_vs_grad_marg   = mse_vs_grad_marg,
        nsims = nsims_completed,
    )
end



function run_error_analysis_qe(
    Cℓ, Cℓn, θpix, Nside, pol, bandpass_mask;
    nsims::Int = 100,
    Δℓ::Int = 50,
    checkpoint_file::String = "results/error_analysis_checkpoint_qe.jld2",
    WL_checkpoint::String = "results/WL_checkpoint_qe.jld2",
    seed0::Int = 1000,
    beamFWHM::Real = 1.0,
    progress::Bool = true,
    T::Type = Float64,
    grad_nbins::Int = 20,
)

    # ── Load W_L from normalization checkpoint ──
    @load WL_checkpoint ℓ_template W_qe_running
    WL_qe_Cℓs = Cℓs(ℓ_template, W_qe_running)
    progress && println("Loaded W_L from $WL_checkpoint ($(length(ℓ_template)) bins)")

    # ── Accumulators ──
    # Noise curves: accumulate (C_ℓ^{rec,rec} - C_ℓ^{true,true}) per sim
    noise_qe_sims    = Vector{Vector{Float64}}()
    ℓ_noise          = nothing

    # MSE maps
    
    sum_Δϕ²_qe    = zeros(Float64, Nside, Nside)
    nsims_completed = 0

    # MSE vs |∇T| binned curves (per-sim, then averaged)
    grad_binned_qe_sims    = Vector{Vector{Float64}}()
    
    grad_bin_cen           = nothing

    # ── Resume from checkpoint ──
    if isfile(checkpoint_file)
        @load checkpoint_file noise_qe_sims ℓ_noise sum_Δϕ²_qe nsims_completed
        try @load checkpoint_file grad_binned_qe_sims grad_bin_cen catch end
        progress && println("Resuming from $checkpoint_file ($nsims_completed completed)")
    end

    start_from = nsims_completed + 1

    for s in start_from:nsims
        progress && (println("→ Simulation $s / $nsims"); flush(stdout))

        (; f, f̃, ϕ, ds) = load_sim(
            seed = seed0 + s,
            Cℓ = Cℓ, Cℓn = Cℓn,
            θpix = θpix, T = T, Nside = Nside,
            beamFWHM = beamFWHM, pol = pol,
            bandpass_mask = bandpass_mask,
            pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )

        # ── QE (no Wiener filter — already correctly normalized) ──
        qe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true)
        ϕ_qe = qe.ϕqe
        ϕ_qe_deb = debias_phi_with_WL(ϕ_qe, WL_qe_Cℓs)  

        cl_rr_qe    = get_Cℓ(ϕ_qe_deb;        Δℓ=Δℓ)
        cl_tt       = get_Cℓ(ϕ;            Δℓ=Δℓ)

        if ℓ_noise === nothing
            ℓ_noise = cl_tt.ℓ
        end

        push!(noise_qe_sims,    cl_rr_qe.Cℓ    .- cl_tt.Cℓ)

        # ── MSE maps ──

        ϕt = Map(ϕ).arr
        
        Δϕ_qe    = Float64.(Map(ϕ_qe_deb).arr        .- ϕt)

        
        sum_Δϕ²_qe    .+= Δϕ_qe    .^ 2
        

        # ── MSE vs |∇T| binned curves (per-sim) ──
        _, _, grad_mag = grad_fft(f)
        cen, m_qe,    _, _ = bin_stat_err(Float64.(grad_mag), Δϕ_qe.^2;    nbins=grad_nbins)
        
        if grad_bin_cen === nothing; grad_bin_cen = cen; end
        push!(grad_binned_qe_sims,    m_qe)

        nsims_completed += 1

        # ── Checkpoint ──
        @save checkpoint_file noise_qe_sims ℓ_noise  sum_Δϕ²_qe nsims_completed grad_binned_qe_sims grad_bin_cen
        progress && println("  Checkpoint saved ($nsims_completed completed)")
    end

    # ── Average noise curves ──
    
    N_qe    = mean(reduce(hcat, noise_qe_sims);    dims=2)[:]

    # ── Average MSE maps ──
    denom = max(nsims_completed, 1)
    mean_Δϕ²_qe    = sum_Δϕ²_qe    ./ denom
    
    # ── Average MSE vs |∇T| binned curves ──
    mse_vs_grad_qe    = mean(reduce(hcat, grad_binned_qe_sims);    dims=2)[:]
    
    return (
        ℓ = ℓ_noise,
        N_qe    = N_qe,
        mean_Δϕ²_qe       = mean_Δϕ²_qe,
        grad_bin_cen       = grad_bin_cen,
        mse_vs_grad_qe     = mse_vs_grad_qe,
        nsims = nsims_completed,
    )
end

"""
    plot_noise_curves(ℓ, N_joint, N_marg, N_qe, Δℓ; savepath="results/noise_curves.png")

Two-panel plot:
- Top: noise curves N_L = <C_ℓ^{rec,rec} - C_ℓ^{true,true}> vs ℓ  (log scale)
- Bottom: histogram of number of Fourier modes per ℓ bin

`Δℓ` should match what was used in `get_Cℓ` so the mode count is consistent.
"""


function plot_noise_curves(ℓ, N_joint, N_marg, N_qe, Δℓ;
    Nside::Int = 512,
    θpix::Float64 = 0.7438046267475303,
    savepath::String = "results/noise_curves.png",
)

    plt_em.rc("font",   size=12, family="serif")
    plt_em.rc("axes",   linewidth=0.8)
    plt_em.rc("xtick",  direction="in", top=true)
    plt_em.rc("ytick",  direction="in", right=true)
    plt_em.rc("xtick.major", width=0.8, size=4)
    plt_em.rc("ytick.major", width=0.8, size=4)

    colours = ["#4477AA", "#EE6677", "#228833"]
    ℓv = collect(ℓ)

    # ── Estimate mode counts per ℓ bin ──
    θpix_rad = θpix * (π / 180) / 60
    L_box = Nside * θpix_rad
    Δℓ_fund = 2π / L_box
    mode_counts = @. 2π * ℓv * Δℓ / Δℓ_fund^2

    fig, (ax_top, ax_bot) = plt_em.subplots(2, 1, figsize=(7, 6),
        gridspec_kw=Dict("height_ratios" => [3, 1], "hspace" => 0.08),
        sharex=true)

    # ── Top panel: noise curves ──
    ax_top.semilogy(ℓv, abs.(N_qe),     color=colours[1], label="QE",        linewidth=1.5)
    ax_top.semilogy(ℓv, abs.(N_joint),  color=colours[2], label="MAP joint", linewidth=1.5)
    ax_top.semilogy(ℓv, abs.(N_marg),   color=colours[3], label="MAP marg",  linewidth=1.5)

    ax_top.set_ylabel(L"\langle C_\ell^{\hat\phi\hat\phi} - C_\ell^{\phi\phi} \rangle", fontsize=13)
    ax_top.legend(frameon=false, fontsize=11)
    ax_top.tick_params(labelsize=11, labelbottom=false)

    # ── Bottom panel: mode counts histogram ──
    ax_bot.bar(ℓv, mode_counts, width=0.8*Δℓ, color="grey", alpha=0.6, edgecolor="none")
    ax_bot.set_xlabel(L"\ell", fontsize=13)
    ax_bot.set_ylabel(L"N_\mathrm{modes}", fontsize=11)
    ax_bot.tick_params(labelsize=10)

    ax_top.set_xlim(0, maximum(ℓv))

    # Avoid tight_layout warning from some matplotlib backends
    fig.subplots_adjust(hspace=0.08)

    fig.savefig(savepath, dpi=150)

    # PythonPlot.jl: close via plotclose (don’t pass fig)
    plt_em.plotclose("all")

    println("Saved noise curve plot to $savepath")
end


function save_error_results(
    results::NamedTuple,
    Δℓ::Int, Nside::Int, θpix::Float64;
    results_file::String = "results/error_analysis_final.jld2",
    plot_path::String = "results/noise_curves_onesim.png",
)
    ℓ = results.ℓ
    N_joint = results.N_joint
    N_marg  = results.N_marg
    N_qe    = results.N_qe
    mean_Δϕ²_joint   = results.mean_Δϕ²_joint
    mean_Δϕ²_marg    = results.mean_Δϕ²_marg
    mean_Δϕ²_qe      = results.mean_Δϕ²_qe
    grad_bin_cen      = results.grad_bin_cen
    mse_vs_grad_qe    = results.mse_vs_grad_qe
    mse_vs_grad_joint = results.mse_vs_grad_joint
    mse_vs_grad_marg  = results.mse_vs_grad_marg
    nsims             = results.nsims

    @save results_file ℓ N_joint N_marg N_qe mean_Δϕ²_joint mean_Δϕ²_marg mean_Δϕ²_qe grad_bin_cen mse_vs_grad_qe mse_vs_grad_joint mse_vs_grad_marg nsims
    println("Saved final results to $results_file")

    plot_noise_curves(ℓ, N_joint, N_marg, N_qe, Δℓ;
        Nside=Nside, θpix=θpix, savepath=plot_path)
end


function plot_noise_curves_qe(ℓ, N_qe, Δℓ;
    Nside::Int = 512,
    θpix::Float64 = 0.7438046267475303,
    savepath::String = "results/noise_curves_qe.png",
)

    plt_em.rc("font",   size=12, family="serif")
    plt_em.rc("axes",   linewidth=0.8)
    plt_em.rc("xtick",  direction="in", top=true)
    plt_em.rc("ytick",  direction="in", right=true)
    plt_em.rc("xtick.major", width=0.8, size=4)
    plt_em.rc("ytick.major", width=0.8, size=4)

    ℓv = collect(ℓ)

    # ── Estimate mode counts per ℓ bin ──
    θpix_rad = θpix * (π / 180) / 60
    L_box = Nside * θpix_rad
    Δℓ_fund = 2π / L_box
    mode_counts = @. 2π * ℓv * Δℓ / Δℓ_fund^2

    fig, (ax_top, ax_bot) = plt_em.subplots(2, 1, figsize=(7, 6),
        gridspec_kw=Dict("height_ratios" => [3, 1], "hspace" => 0.08),
        sharex=true)

    # ── Top panel: QE noise curve ──
    ax_top.semilogy(ℓv, abs.(N_qe), color="#4477AA", label="QE", linewidth=1.5)

    ax_top.set_ylabel(L"\langle C_\ell^{\hat\phi\hat\phi} - C_\ell^{\phi\phi} \rangle", fontsize=13)
    ax_top.legend(frameon=false, fontsize=11)
    ax_top.tick_params(labelsize=11, labelbottom=false)

    # ── Bottom panel: mode counts histogram ──
    ax_bot.bar(ℓv, mode_counts, width=0.8*Δℓ, color="grey", alpha=0.6, edgecolor="none")
    ax_bot.set_xlabel(L"\ell", fontsize=13)
    ax_bot.set_ylabel(L"N_\mathrm{modes}", fontsize=11)
    ax_bot.tick_params(labelsize=10)

    ax_top.set_xlim(0, maximum(ℓv))

    fig.subplots_adjust(hspace=0.08)
    fig.savefig(savepath, dpi=150)

    plt_em.plotclose("all")

    println("Saved QE noise curve plot to $savepath")
end


function save_error_results_qe(
    results::NamedTuple,
    Δℓ::Int, Nside::Int, θpix::Float64;
    results_file::String = "results/error_analysis_final_qe.jld2",
    plot_path::String = "results/noise_curves_qe.png",
)

    ℓ      = results.ℓ
    N_qe   = results.N_qe
    mean_Δϕ²_qe = results.mean_Δϕ²_qe
    grad_bin_cen = results.grad_bin_cen
    mse_vs_grad_qe = results.mse_vs_grad_qe
    nsims  = results.nsims

    @save results_file ℓ N_qe mean_Δϕ²_qe grad_bin_cen mse_vs_grad_qe nsims
    println("Saved QE-only results to $results_file")

    plot_noise_curves_qe(ℓ, N_qe, Δℓ;
        Nside=Nside, θpix=θpix, savepath=plot_path)
end

