using Statistics
using PythonPlot
using CMBLensing

function plot_grad_T_lensed(f̃)
    dTdx, dTdy, grad_mag = grad_fft_local(f̃)

    lo = quantile(vec(grad_mag), 0.2)
    hi = quantile(vec(grad_mag), 0.8)

    plt.figure(figsize=(12,4))

    plt.subplot(1,3,1)
    plt.imshow(dTdx, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title("∂T/∂x  (µK per pixel)")
    plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(dTdy, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title("∂T/∂y  (µK per pixel)")
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(grad_mag, cmap="viridis", vmin=lo, vmax=hi)
    plt.title("|∇T|  (µK per pixel)")
    plt.colorbar()

    plt.tight_layout()
end

function compare_gradT_phi_errors(f, ϕ, ϕJ, ϕmarg, ϕqe)
    # 1. Compute |∇T|
    _, _, gradT = grad_fft_local(f)
    gradT = gradT ./ maximum(gradT)   # normalise 0–1 for better contrast
    
    # 2. φ residuals in μK
    Δϕ_QE   = Map(ϕ .- ϕqe).arr
    Δϕ_J    = Map(ϕ .- ϕJ).arr
    Δϕ_marg = Map(ϕ .- ϕmarg).arr
    Δϕ_test   = Map(ϕ).arr
    scale = 1e6
    dQE   = scale .* (Δϕ_QE)
    dJ    = scale .* (Δϕ_J)
    dMarg = scale .* (Δϕ_marg)
    dtest = scale .* (Δϕ_test)




    # symmetric φ error scale
    m = maximum(abs.([dQE; dJ; dMarg]))
    lo, hi = -m, m

    plt.figure(figsize=(14,7))

    # |∇T|
    plt.subplot(2,3,1)
    plt.imshow(gradT, cmap="inferno")
    plt.title("|∇T| (normalised)")
    plt.colorbar()

    # φ residuals
    plt.subplot(2,3,2)
    plt.imshow(dQE, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title("φ_true − φ_QE")
    plt.colorbar()

    plt.subplot(2,3,3)
    plt.imshow(dJ, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title("φ_true − φ_joint")
    plt.colorbar()

    plt.subplot(2,3,4)
    plt.imshow(dMarg, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title("φ_true − φ_marg")
    plt.colorbar()

    plt.subplot(2,3,5)
    plt.imshow(dtest, cmap="RdBu_r", vmin=lo, vmax=hi)
    plt.title("φ_true")
    plt.colorbar()

    plt.tight_layout()
end

"""
    bin_stat(x, y; nbins=20)

Given two same-size arrays x,y:
- bins x into `nbins` equal-width bins
- returns (bin_centres, mean(y) in each bin)
"""
function bin_stat(x, y ; nbins=20)
    xvec = vec(x)
    yvec = vec(y)

    lo, hi = minimum(xvec), maximum(xvec)
    edges = range(lo, hi; length=nbins+1)

    means = Float64[]
    centres = Float64[]

    for i in 1:nbins
        binmask = (edges[i] .<= xvec .< edges[i+1])
        push!(centres, 0.5*(edges[i] + edges[i+1]))

        if any(binmask)
            push!(means, mean(yvec[binmask]))
        else
            push!(means, NaN)
        end
    end
    return centres, means
end


"""
    plot_phi_error_vs_gradT(f̃, Δϕ_QE, Δϕ_J, Δϕ_marg)

Plots mean |Δϕ| vs |∇T| per-pixel for each estimator.
"""
function plot_phi_error_vs_gradT(f, Δϕ_QE, Δϕ_J, Δϕ_marg)
    # gradient magnitude per pixel (µK per pixel)
    _, _, grad_mag = grad_fft_local(f)          # |∇T| per rad
    grad_pix = grad_mag 
    
    # absolute φ errors, mod-squared
    e_QE   = abs.(Δϕ_QE).^2
    e_J    = abs.(Δϕ_J).^2
    e_marg = abs.(Δϕ_marg).^2

    # binning
    cen, m_QE   = bin_stat(grad_pix, e_QE)
    _,   m_J    = bin_stat(grad_pix, e_J)
    _,   m_marg = bin_stat(grad_pix, e_marg)

    # plot
    plt.figure(figsize=(7,5))
    plt.plot(cen, m_QE,   label="QE",       marker="o")
    plt.plot(cen, m_J,    label="Joint MAP", marker="s")
    plt.plot(cen, m_marg, label="MAP_marg",  marker="^")
    plt.xlabel("|∇T|  (µK per pixel)")
    plt.ylabel("mean |Δϕ|²")
    plt.title("Mean |Δϕ|² vs |∇T|")
    plt.grid(true, alpha=0.3)
    plt.legend()
    plt.tight_layout()
end

