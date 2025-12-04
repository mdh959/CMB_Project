module Utils

using CMBLensing, PythonPlot, NPZ, Statistics
plt = PythonPlot
export get_Cℓ_fft, bin_spectrum, save_results,
       plot_phi_error_vs_gradT, bin_stat, compare_gradT_phi_errors,
       plot_grad_T_lensed, grad_fft_local

include("Functions/fft.jl")
include("Functions/npz.jl")
include("Functions/plot.jl")

end


