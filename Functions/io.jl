using NPZ
using CMBLensing

"""
    save_results(filename; ϕ, ϕJ, ϕ_marg, f, fJ, hist, Cℓ, Cℓn, ds, θpix, Nside, T, pol)

Save all reconstruction results and metadata to an `.npz` file.
"""
function save_results(
    filename;
    ϕ, ϕJ, ϕ_marg,
    f, fJ,
    hist,
    Cℓ, Cℓn, ds,
    θpix, Nside, T, pol
)
    getdiag(x) = x isa Diagonal ? x.diag :
                 x isa CMBLensing.ParamDependentOp ? x.op.diag :
                 error("Unknown covariance operator type")

    asbytes(s::String) = Vector{UInt8}(codeunits(s))

    hist_logpdf = [h.logpdf for h in hist]

    mkpath(dirname(filename))

    npzwrite(filename, Dict(
        "phi_true"         => Array(ϕ),
        "phi_map"          => Array(ϕJ),
        "phi_marg"         => Array(ϕ_marg),
        "f_unlensed"       => Array(f),
        "f_map"            => Array(fJ),
        "history"          => hist_logpdf,
        "ell"              => Cℓ.total.ϕϕ.ℓ,
        "Cl_phi"           => Cℓ.total.ϕϕ.Cℓ,
        "Cl_TT"            => Cℓ.total.TT.Cℓ,
        "Cl_noise"         => Cℓn.TT.Cℓ,
        "Cf_diag"          => getdiag(ds.Cf),
        "Cphi_diag"        => getdiag(ds.Cϕ),
        "Cn_diag"          => getdiag(ds.Cn),
        "theta_pix_arcmin" => θpix,
        "Nside"            => Nside,
        "dtype"            => asbytes(string(T)),
        "pol"              => asbytes(string(pol)),
    ))

    println("Saved: $filename")
end