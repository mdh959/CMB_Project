module Utils

using CMBLensing, PythonPlot, NPZ, Statistics, LinearAlgebra, JLD2

# ── Spectra & binning ──
export get_Cℓ_fft, bin_spectrum, bin_stat, bin_stat_err

# ── Gradient ──
export grad_fft

# ── Gradient Inversion estimator ──
export gi_estimate

# ── Normalization (analytical + empirical W_L) ──
export compute_RL_iterative, compute_WL, compute_WL_analytical
export empirical_WL_all_loadsim

# ── Debiasing ──
export debias_phi_with_WL

# ── Error analysis ──
export run_error_analysis, run_error_analysis_all_debiased, plot_noise_curves, save_error_results
export run_error_analysis_qe, plot_noise_curves_qe, plot_noise_curves_L4, save_error_results_qe

# ── I/O ──
export save_results

# ── Plotting ──
export plot_grad_T_lensed, compare_gradT_phi_errors,
       plot_phi_error_vs_gradT, plot_WL_comparison,
       plot_correlation_coefficient

# Include files (order matters: spectra & gradient before plotting)
include("Functions/spectra.jl")
include("Functions/gradient.jl")
include("Functions/gradient_inversion.jl")
include("Functions/normalization.jl")
include("Functions/debias.jl")
include("Functions/error_mean.jl")
include("Functions/io.jl")
include("Functions/plotting.jl")

end