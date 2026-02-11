
using CMBLensing
using PythonPlot
Cℓ = camb(r=0.05);
Cℓn = noiseCℓs(μKarcminT=1, ℓknee=100);

θpix  = 3        # pixel size in arcmin
Nside = 128      # number of pixels per side in the map
pol   = :P       # type of data to use (can be :T, :P, or :TP)
T     = Float32  # data type (Float32 is ~2 as fast as Float64);

(;f, f̃, ϕ, ds) = load_sim(
    seed = 3,
    Cℓ = Cℓ,
    Cℓn = Cℓn,
    θpix = θpix,
    T = T,
    Nside = Nside,
    pol = pol,
)

(;Cf, Cϕ) = ds;

plot(ϕ, title = raw"true $\phi$");

fJ, ϕJ, history = MAP_joint(ds, nsteps=30, progress=true);

plot(getindex.(history, :logpdf))
xlabel("step")
ylabel("logpdf");

plot(10^6*[ϕ ϕJ], title=["true" "best-fit"] .* raw" $\phi$", vlim=17);

semilogx(get_ρℓ(ϕ, ϕJ))
semilogx(get_ρℓ(quadratic_estimate(ds).ϕqe, ϕ))
xlabel("ℓ")
ylabel("ρℓ")
legend(["Joint MAP", "QE"])
savefig("plot_name.png")
