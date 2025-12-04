using CMBLensing
using PythonPlot


Cℓ  = camb(r=0.05, ℓmax=21000)         
Cℓn = noiseCℓs(μKarcminT=1.0)


θpix  = 0.7438046267475303  # pixel size in arcmin
Nside = 512      # number of pixels per side in the map
pol   = :I       # type of data to use (can be :T, :P, or :TP)
T     = Float64  # data type (Float32 is ~2 as fast as Float64);
bandpass_mask = LowPass(6000);

(;f, f̃, ϕ, ds) = load_sim(
    seed = 3,
    Cℓ = Cℓ,
    Cℓn = Cℓn,
    θpix = θpix,
    T = T,
    Nside = Nside,
    beamFWHM = 1.0,
    pol = pol,
    bandpass_mask = bandpass_mask, 
    pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

qe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true)
ϕqe = qe.ϕqe


using CMBLensing: Map, Loess
Base.eval(CMBLensing, :(loess = Loess.loess))


Ωstart = FieldTuple(ϕ = qe.ϕqe)


result = MAP_joint(
    ds,
    Ωstart;
    nsteps = 20,
    αtol = 1e-4,
    nburnin_update_hessian = Inf,
    conjgrad_kwargs=(tol=1e-3, nsteps=300),
    progress=true
)




fJ = result.f
ϕJ = result.ϕ
hist = result.history

using CMBLensing

# Add a module-private global inside CMBLensing to hold ds
@eval CMBLensing begin
    const _DIST_DS = Ref{Any}(nothing)

    set_distributed_dataset(x) = (_DIST_DS[] = x)
    get_distributed_dataset() = _DIST_DS[]
end


# --- MAP_marg call ---
ϕ_marg, hist_marg = MAP_marg(
    ds,
    ϕstart = ϕqe;
    nsteps = 20,
    nsteps_with_meanfield_update = 3,
    conjgrad_kwargs = (tol=1e-6,nsteps=100),
    progress = true,
    pmap = map
)

function get_Cℓ_fft(ϕ1, ϕ2=ϕ1; nbins=300)
    m1 = Map(ϕ1)
    m2 = Map(ϕ2)

    a = m1.arr
    b = m2.arr

    Ny, Nx = size(a)
    proj = m1.proj
    θpix_rad = proj.θpix * (π/180)/60

    F1 = m_rfft(a, (1,2))
    F2 = m_rfft(b, (1,2))

    Pk = real.(F1 .* conj.(F2))
    Pk ./= Ny*Nx*θpix_rad^2

    NyF, NxF = size(F1)
    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓmap = @. sqrt((ℓx')^2 + ℓy^2)

    ℓmin, ℓmax = minimum(ℓmap), maximum(ℓmap)
    edges = range(ℓmin, ℓmax; length=nbins+1)

    Cs  = zeros(nbins)
    ℓc  = zeros(nbins)

    for i in 1:nbins
        mask = (edges[i] .<= ℓmap .< edges[i+1])
        if any(mask)
            Cs[i] = mean(Pk[mask])
            ℓc[i] = mean(ℓmap[mask])
        else
            ℓc[i] = (edges[i] + edges[i+1])/2
            Cs[i] = NaN
        end
    end

    return Cℓs(ℓc, Cs)
end

function bin_spectrum(ells, vals; ΔL=300)
    Lmin, Lmax = minimum(ells), maximum(ells)
    edges = collect(Lmin:ΔL:Lmax)
    centers = @. 0.5*(edges[1:end-1] + edges[2:end])

    binned = similar(centers)
    for i in eachindex(centers)
        mask = (ells .>= edges[i]) .& (ells .< edges[i+1])
        binned[i] = any(mask) ? mean(vals[mask]) : NaN
    end

    return centers, binned
end



using CMBLensing
using FFTW
using Statistics
using StableRNGs: StableRNG
using JLD2: @save, @load

function empirical_WL_maps_loadsim(
    Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
    nsims = 50, nbins = 300,
    checkpoint_file = "WL_checkpoint.jld2"
)

    # storage for per-sim W_L
    R_joint_sims = Vector{Vector{Float64}}()
    R_marg_sims  = Vector{Vector{Float64}}()
    ℓ_template   = nothing
    W_joint_running = nothing
    W_marg_running  = nothing

    # --- resume logic ---
    start_from = 1
    if isfile(checkpoint_file)
        println("Resuming from checkpoint: $checkpoint_file")
        @load checkpoint_file R_joint_sims R_marg_sims ℓ_template W_joint_running W_marg_running

        start_from = length(R_joint_sims) + 1
        println("→ Starting from simulation $start_from")
    end

    for s in start_from:nsims
        println("→ Simulation $s / $nsims")

        # 1. Generate simulation
        (; f, f̃, ϕ, ds) = load_sim(
            seed = 1000 + s,
            Cℓ = Cℓ, Cℓn = Cℓn,
            θpix = θpix, T = T, Nside = Nside,
            beamFWHM = 1.0,
            pol = pol,
            bandpass_mask = bandpass_mask,
            pixel_mask_kwargs = (edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
        )

        ϕ_true = Map(ϕ)

        # 2. QE
        qe = quadratic_estimate(ds; weights=:lensed, wiener_filtered=true)
        ϕqe = qe.ϕqe

        # 3. MAP_marg 

        ϕ_marg = nothing
        try
            ϕ_marg, _ = MAP_marg(
                ds; ϕstart = ϕqe,
                nsteps = 15,
                progress = false,
                conjgrad_kwargs = (tol=1e-3, nsteps=100),
            )
        catch err
            println("MAP_marg failed: $err — skipping sim $s")
            continue
        end

        # 4. MAP_joint 
        Ωstart = FieldTuple(ϕ = ϕqe)
        ϕ_joint = nothing

        try
            result = MAP_joint(
                ds, Ωstart;
                nsteps = 15,
                progress = false,
                conjgrad_kwargs = (tol=1e-3, nsteps=100),
            )
            ϕ_joint = result.ϕ
        catch err
            println("MAP_joint failed: $err — skipping sim $s")
            continue
        end

        # 5. Spectra
        cl_tj = get_Cℓ_fft(ϕ, ϕ_joint; nbins=nbins)
        cl_tm = get_Cℓ_fft(ϕ, ϕ_marg;  nbins=nbins)
        cl_tt = get_Cℓ_fft(ϕ, ϕ_true; nbins=nbins)

        ℓ    = cl_tt.ℓ
        C_tj = cl_tj.Cℓ
        C_tm = cl_tm.Cℓ
        C_tt = cl_tt.Cℓ

        if ℓ_template === nothing
            ℓ_template = ℓ
        end

        push!(R_joint_sims, C_tj ./ C_tt)
        push!(R_marg_sims,  C_tm ./ C_tt)

        # 6. Running averages
        W_joint_running = mean(reduce(hcat, R_joint_sims); dims=2)[:]
        W_marg_running  = mean(reduce(hcat, R_marg_sims);  dims=2)[:]

        # 7. Save everything
        @save checkpoint_file R_joint_sims R_marg_sims ℓ_template W_joint_running W_marg_running
        @save "wl_maps_sim$s.jld2" ϕ_true ϕ_joint ϕ_marg

        println("Saved checkpoint after sim $s")
    end

    return ℓ_template, W_joint_running, W_marg_running
end

ℓ, WL_joint, WL_marg = empirical_WL_maps_loadsim(
    Cℓ, Cℓn, θpix, T, Nside, pol, bandpass_mask;
    nsims = 100,
    nbins = 300
)

using CMBLensing: Map, BaseField, m_irfft

"""
    debias_phi_with_WL(ϕF::BaseField{Fourier}, ℓc, W_L)

Debias a lensing potential φ in *LambertFourier* basis using W_L.

Input:
    ϕF  :: BaseField{Fourier}   (LambertFourier field)
    ℓc  :: Vector{Float64}      (bin centres)
    W_L :: Vector{Float64}      (Wiener filter per bin)

Output:
    Map of debiased φ.
"""
function debias_phi_with_WL(ϕF::BaseField{Fourier}, ℓc::Vector{Float64}, W_L::Vector{Float64})
    proj = ϕF.proj
    φF   = ϕF.arr             # raw Fourier array
    NyF, NxF = size(φF)
    Ny       = proj.Ny

    # --- Build ℓ grid matching rfft layout ---
    # proj.ℓx, proj.ℓy are 1D arrays; keep the parts actually present in F1
    ℓx = proj.ℓx[1:NxF]
    ℓy = proj.ℓy[1:NyF]

    ℓx2D = repeat(ℓx', NyF, 1)
    ℓy2D = repeat(ℓy , 1,  NxF)
    ℓmag = @. sqrt(ℓx2D^2 + ℓy2D^2)


    # --- Bin edges from centres ---
    nbins = length(ℓc)
    edges = similar(ℓc, nbins+1)
    edges[2:nbins] .= 0.5 .* (ℓc[1:nbins-1] .+ ℓc[2:nbins])
    edges[1]  = 0.0
    edges[end] = maximum(ℓmag) * 1.001

    # --- Debias ---
    debiasF = similar(φF)
    for idx in eachindex(φF)
        L = ℓmag[idx]
        b = searchsortedfirst(edges, L) - 1
        if 1 ≤ b ≤ nbins && W_L[b] > 1e-6
            debiasF[idx] = φF[idx] / W_L[b]
        else
            debiasF[idx] = 0
        end
    end

    # --- Back to real space ---
    #φ_arr = m_irfft(debiasF, Ny, (1,2))
    ϕF = BaseField{Fourier}(debiasF, proj)
    return Map(ϕF)      # CMBLensing’s inverse FFT, correct scaling
end


ϕ_marg_deb  = debias_phi_with_WL(ϕ_marg,  ℓ, WL_marg)
ϕ_joint_deb = debias_phi_with_WL(ϕ_joint, ℓ, WL_joint)

@save "debias_results.jld2" ϕ_marg_deb ϕ_joint_deb ℓ WL_marg WL_joint
