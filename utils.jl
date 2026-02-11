module Utils

using CMBLensing, PythonPlot, NPZ, Statistics, LinearAlgebra, JLD2

# ── Spectra & binning ──
export get_Cℓ_fft, bin_spectrum, bin_stat, bin_stat_err

# ── Gradient ──
export grad_fft

# ── Normalization (analytical + empirical W_L) ──
export compute_RL_iterative, compute_WL, compute_WL_analytical
export empirical_WL_maps_loadsim

# ── Debiasing ──
export debias_phi_with_WL

# ── I/O ──
export save_results

# ── Plotting ──
export plot_grad_T_lensed, compare_gradT_phi_errors,
       plot_phi_error_vs_gradT, plot_WL_comparison,
       plot_correlation_coefficient

# Include files (order matters: spectra & gradient before plotting)
include("Functions/spectra.jl")
include("Functions/gradient.jl")
include("Functions/normalization.jl")
include("Functions/debias.jl")
include("Functions/io.jl")
include("Functions/plotting.jl")

end