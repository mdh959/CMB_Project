# run_study.jl
# Runs all three thesis studies and saves results to JLD2.
# Can be re-run safely — already-completed (noise, lmax, seed) combos are skipped.
#
# Usage:
#   julia --project run_study.jl
#
# Results saved to: results/noise_study.jld2
#                   results/bandpass_study.jld2
println("RUN_STUDY STARTED")
using JLD2: @load, @save
using Printf
using PythonCall   # must load before CMBLensing to register Python extension
using CMBLensing
include("study_helpers.jl")
mkpath("results")

# -----------------------------------------------------------------------
# Pre-compute theory Cℓ once — expensive, reuse across all runs
# -----------------------------------------------------------------------
println("Computing theory Cℓ...")
const Cℓ_theory = camb(r=0.05, ℓmax=21000)

const SEEDS = 1:20    

# -----------------------------------------------------------------------
# Study 1: Noise dependence
# Fixed: lmax_cmb=8000, beamFWHM=1.0
# Vary:  μKarcminT
# -----------------------------------------------------------------------
const NOISE_LEVELS = [5.0, 2.0, 1.0, 0.5, 0.1]
const NOISE_LABELS = ["5.0", "2.0", "1.0", "0.5", "0.1"]
const FIXED_LMAX   = 8000

println("\n=== Study 1: Noise dependence ===")

noise_results = Dict()

for (μK, label) in zip(NOISE_LEVELS, NOISE_LABELS)
    noise_results[label] = []
    for seed in SEEDS
        key = "noise_$(label)_seed_$(seed)"

        # skip if already computed
        if isfile("results/noise_study.jld2")
            existing = jldopen("results/noise_study.jld2", "r") do f
                key in keys(f)
            end
            existing && continue
        end

        result = run_one_sim(
            seed        = seed,
            μKarcminT   = μK,
            lmax_cmb    = FIXED_LMAX,
            Cℓ_theory   = Cℓ_theory,
        )

        jldopen("results/noise_study.jld2", "a+") do f
            f[key] = result
        end

        push!(noise_results[label], result)
        GC.gc()
    end
end

println("Noise study complete.")

# -----------------------------------------------------------------------
# Study 2: Bandpass dependence
# Fixed: μKarcminT=0.1, beamFWHM=1.0
# Vary:  lmax_cmb
# -----------------------------------------------------------------------
const BANDPASS_LEVELS  = [4000, 6000, 8000, 12000]
const FIXED_NOISE_MARG = 0.1

println("\n=== Study 2: Bandpass dependence ===")

bandpass_results = Dict()

for lmax in BANDPASS_LEVELS
    bandpass_results[lmax] = []
    for seed in SEEDS
        key = "lmax_$(lmax)_seed_$(seed)"

        if isfile("results/bandpass_study.jld2")
            existing = jldopen("results/bandpass_study.jld2", "r") do f
                key in keys(f)
            end
            existing && continue
        end

        result = run_one_sim(
            seed        = seed,
            μKarcminT   = FIXED_NOISE_MARG,
            lmax_cmb    = lmax,
            Cℓ_theory   = Cℓ_theory,
        )

        jldopen("results/bandpass_study.jld2", "a+") do f
            f[key] = result
        end

        push!(bandpass_results[lmax], result)
        GC.gc()
    end
end

println("Bandpass study complete.")
println("\nAll studies done. Results in results/")
