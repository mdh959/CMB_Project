#!/usr/bin/env julia
#
# make_ul_comparison_sims.jl
#
# Generates 100 sims with the UL parameters from run_qe_gi_wl12k.jl,
# split into 20 files of 5 sims each (ul_comparison_sims_001_005.npz, etc.).
# 5 sims × 3 maps × 512² × 8 bytes ≈ 30 MB per file (under GitHub's 50 MB limit).
# Each file has flat keys readable in Python via np.load:
#   sim_N_T_obs      — observed T map (512x512, Float64), beam-convolved + noise
#   sim_N_phi_true   — true lensing potential (512x512, Float64)
#   sim_N_phi_qe     — QE estimate (raw, not Wiener-filtered, 512x512, Float64)
#   sim_N_seed       — seed (1-element array)
#
# Metadata (1-element arrays, same in every file):
#   params_theta_pix, params_Nside, params_muKarcminT, params_beamFWHM,
#   params_Lmax, params_lknee, params_seed0, params_n_sims,
#   params_r_tensor, params_Cl_lmax
#
# Python usage:
#   import numpy as np
#   d = np.load("ul_comparison_sims_001_010.npz")
#   T_obs = d["sim_1_T_obs"]        # shape (512, 512), float64
#
# Output: data/ul_comparison_sims_NNN_MMM.npz (20 files)

import Pkg; Pkg.activate(@__DIR__); Pkg.instantiate()

using CMBLensing
using LinearAlgebra
using Statistics: mean
using NPZ
using Printf

include("utils.jl")
using .Utils

const μKarcminT = 0.1
const beamFWHM  = 0.3
const Lmax      = 12000
const θpix      = 0.7438046267475303
const Nside     = 512
const pol       = :I
const seed0     = 1000
const n_sims    = 100

const Cℓ          = camb(r=0.05, ℓmax=35000)
const Cℓn         = noiseCℓs(μKarcminT=μKarcminT, ℓknee=0, ℓmax=Lmax)
# Hard cutoff matching run_qe_gi_wl12k.jl (wide taper caused stripe outliers near Lmax)
const qe_bandpass  = LowPass(Lmax)

const load_kwargs = (
    Cℓ=Cℓ, Cℓn=Cℓn, θpix=θpix, T=Float64, Nside=Nside,
    beamFWHM=beamFWHM, pol=pol, bandpass_mask=qe_bandpass,
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

@eval CMBLensing begin
    function Hessian_logpdf_preconditioner(::Val{:f}, ds::DataSet)
        @unpack Cf̃, B̂, M̂, Cn̂ = ds
        v   = copy(diag(pinv(Cf̃) + B̂'*M̂'*pinv(Cn̂)*M̂*B̂))
        arr = v.arr
        good = isfinite.(arr) .& (arr .> 0)
        fill!(view(arr, .!good), any(good) ? mean(arr[good]) : 1.0)
        arr .= clamp.(arr, 1e-30, 1e30)
        return Diagonal(v)
    end
end

const sims_per_file = 5    # 5 sims × 3 maps × 512² × 8 bytes ≈ 30 MB per file

println("UL params: μKarcmin=$μKarcminT, beamFWHM=$beamFWHM, Lmax=$Lmax, θpix=$θpix, Nside=$Nside")
println("Seeds: $(seed0+1) – $(seed0+n_sims), split into $(cld(n_sims, sims_per_file)) files of $sims_per_file sims")

for file_idx in 1:(n_sims ÷ sims_per_file)
    sim_start = (file_idx - 1) * sims_per_file + 1
    sim_end   = file_idx * sims_per_file
    outfile   = @sprintf "data/ul_comparison_sims_%03d_%03d.npz" sim_start sim_end
    println("\n--- File $file_idx: sims $(sim_start)–$(sim_end) → $outfile ---")

    out = Dict{String, Array}()

    for s in sim_start:sim_end
        seed = seed0 + s
        @printf "  Sim %3d/%d (seed=%d) ... " s n_sims seed
        flush(stdout)

        (; ϕ, ds) = load_sim(; seed=seed, load_kwargs...)

        ϕqe = quadratic_estimate(ds; weights=:unlensed, wiener_filtered=false).ϕqe

        out["sim_$(s)_T_obs"]    = Float64.(Map(ds.d).arr)
        out["sim_$(s)_phi_true"] = Float64.(Map(ϕ).arr)
        out["sim_$(s)_phi_qe"]   = Float64.(Map(ϕqe).arr)
        out["sim_$(s)_seed"]     = [seed]

        println("done")
        flush(stdout)
    end

    out["params_theta_pix"]   = [θpix]
    out["params_Nside"]       = [Nside]
    out["params_muKarcminT"]  = [μKarcminT]
    out["params_beamFWHM"]    = [beamFWHM]
    out["params_Lmax"]        = [Lmax]
    out["params_lknee"]       = [0]
    out["params_seed0"]       = [seed0]
    out["params_n_sims"]      = [n_sims]
    out["params_r_tensor"]    = [0.05]
    out["params_Cl_lmax"]     = [35000]

    npzwrite(outfile, out)
    println("  Saved $outfile")
end

println("\nDone. Saved $n_sims sims across $(cld(n_sims, sims_per_file)) files in data/")
