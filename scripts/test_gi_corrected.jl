#!/usr/bin/env julia
# Diagnostic: run gi_estimate_corrected on one sim and check internals.
# Run from project root:  julia scripts/test_gi_corrected.jl

import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CMBLensing, Statistics, Printf
include("../utils.jl")
using .Utils

const Cℓ = camb(r=0.05, ℓmax=35000)
seed = 1001

# ── S4-like sim (same params as run script) ───────────────────────────────────
println("Loading sim seed=$seed ...")
Cℓn     = noiseCℓs(μKarcminT=1.0, ℓknee=0, ℓmax=12000)
(; ϕ, ds) = load_sim(; seed=seed, Cℓ=Cℓ, Cℓn=Cℓn, θpix=0.7438046267475303,
                        T=Float64, Nside=512, beamFWHM=1.0, pol=:I,
                        bandpass_mask=LowPass(12000),
                        pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0))

# ── Check diagonals ───────────────────────────────────────────────────────────
m    = Map(ds.d)
Ny, Nx = size(m.arr)
NyF, NxF = Ny ÷ 2 + 1, Nx

f_ones  = FlatFourier(ones(ComplexF64, NyF, NxF), Map(ds.d).proj)
B_diag  = real.(ds.B̂.diag.arr)[1:NyF, 1:NxF]   # B̂ is plain Diagonal
Cf_diag = real.((ds.Cf * f_ones).arr)[1:NyF, 1:NxF]
Cn_diag = real.((ds.Cn̂ * f_ones).arr)[1:NyF, 1:NxF]
Cϕ_diag = real.((ds.Cϕ * f_ones).arr)[1:NyF, 1:NxF]

println("\n── Diagonal sanity checks ───────────────────────────────────────")
println("B_diag:  min=$(minimum(B_diag))  max=$(maximum(B_diag))  finite=$(all(isfinite.(B_diag)))")
println("Cf_diag: min=$(minimum(Cf_diag)) max=$(maximum(Cf_diag)) finite=$(all(isfinite.(Cf_diag)))")
println("Cn_diag: min=$(minimum(Cn_diag)) max=$(maximum(Cn_diag)) finite=$(all(isfinite.(Cn_diag)))")
println("Cϕ_diag: n_positive=$(sum(Cϕ_diag .> 0)) / $(length(Cϕ_diag))")

PS_2D_old = @. B_diag^2 * Cf_diag + Cn_diag           # old formula
PS_2D_new = @. Cf_diag + Cn_diag / max(B_diag^2, 1e-30) # new (matches Python)
println("\nPS_2D (old B²Cf+N):  min=$(minimum(PS_2D_old))  max=$(maximum(PS_2D_old))")
println("PS_2D (new Cf+N/B²): min=$(minimum(PS_2D_new))  max=$(maximum(PS_2D_new))")
# At ℓ~6000 (where beam starts cutting), new >> old (noise deconvolved)

# ── Run the estimator ─────────────────────────────────────────────────────────
println("\n── Running gi_estimate_corrected ────────────────────────────────")
ϕgi_c = gi_estimate_corrected(ds; Lgrad=2000, Lmax=12000)

println("Result type: $(typeof(ϕgi_c))")
ϕgi_c_map = Map(ϕgi_c)
arr = ϕgi_c_map.arr
println("Result map:  $(size(arr))  finite=$(all(isfinite.(arr)))")
println("             min=$(minimum(arr))  max=$(maximum(arr))  std=$(std(arr))")

# ── Compare W_L ──────────────────────────────────────────────────────────────
println("\n── W_L (cross/auto ratio) ───────────────────────────────────────")
cl_tt   = get_Cℓ(ϕ; Δℓ=50)
cl_cross = get_Cℓ(ϕ, ϕgi_c; Δℓ=50)
ℓ = cl_tt.ℓ
R = cl_cross.Cℓ ./ cl_tt.Cℓ
println("  ℓ        R = C(true,gi_c)/C(true,true)")
for ℓ_check in [3000, 5000, 7000, 9000, 11000]
    idx = argmin(abs.(ℓ .- ℓ_check))
    @printf "  ℓ≈%5d   R=%8.4f\n" round(Int, ℓ[idx]) R[idx]
end
println("\nDone. If R values are ~0.5–1 the estimator is working correctly.")
println("(R will be calibrated to 1 by W_L in the full pipeline.)")
