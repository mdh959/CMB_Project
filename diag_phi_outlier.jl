#!/usr/bin/env julia
#
# diag_phi_outlier.jl
#
# Finds the worst QE outlier sim in the UL run (lowest pixel-space correlation
# between ϕ_true and ϕ_qe), then plots phi maps side-by-side for that sim from
# both the UL and S4 runs:
#
#   Row 1 (UL):  ϕ_true | ϕ_qe | ϕ_gi_b | ϕ_mj (if available)
#   Row 2 (S4):  ϕ_true | ϕ_qe | ϕ_gi_b | ϕ_mj (if available)
#
# Also plots difference maps (ϕ̂ - ϕ_true) to highlight reconstruction error.
# Convergence κ = -(ℓ²/2)ϕ is computed and shown for physical intuition.
#
# Run from project root:
#   julia diag_phi_outlier.jl

import Pkg; Pkg.activate(@__DIR__)

using CMBLensing
using JLD2
using Statistics
using Printf
using PythonPlot

include("utils.jl")
using .Utils

const Cℓ    = camb(r=0.05, ℓmax=35000)
const θpix  = 0.7438046267475303
const Nside = 512
const seed0 = 1000

# Shared sim loading metadata (S4 params — both S4 and UL use same θpix/Nside now)
load_kw_s4 = (
    Cℓ=Cℓ, Cℓn=noiseCℓs(μKarcminT=1.0,  ℓknee=0, ℓmax=12000),
    θpix=θpix, T=Float64, Nside=Nside, beamFWHM=1.0,
    pol=:I, bandpass_mask=LowPass(12000),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)
load_kw_ul = (
    Cℓ=Cℓ, Cℓn=noiseCℓs(μKarcminT=0.1,  ℓknee=0, ℓmax=12000),
    θpix=θpix, T=Float64, Nside=Nside, beamFWHM=0.3,
    pol=:I, bandpass_mask=LowPass(12000),
    pixel_mask_kwargs=(edge_padding_deg=0, apodization_deg=0, num_ptsrcs=0),
)

# ── Wrap helper ───────────────────────────────────────────────────────────────
(; ϕ) = load_sim(; seed=seed0+1, load_kw_s4...)
ϕ_ref = Map(ϕ)
wrap(arr) = typeof(ϕ_ref)(Float64.(arr), ϕ_ref.metadata)

# ── κ from ϕ (convergence = -(ℓ²/2)ϕ in Fourier, clipped real part in map) ──
function phi_to_kappa(ϕ_field)
    ϕ_f   = Fourier(ϕ_field)
    ℓmag  = fieldinfo(ϕ_f).ℓmag
    κ_f   = Diagonal(Cℓs(collect(ℓmag[:]), @. -(ℓmag[:]^2 / 2))) * ϕ_f
    return Map(κ_f)
end

# ── Find worst UL outlier ─────────────────────────────────────────────────────
ul_phi_file = "results/phi_maps_qe_gi_12000_ul.jld2"
s4_phi_file = "results/phi_maps_qe_gi_12000.jld2"

if !isfile(ul_phi_file)
    println("UL phi maps file not found: $ul_phi_file")
    println("Run run_qe_gi_wl12k.jl first to generate UL results.")
    exit(1)
end

println("Scanning UL phi maps for worst QE outlier...")
worst_sim = -1
worst_corr = Inf   # we want lowest correlation (most wrong)
n_scanned = 0

jldopen(ul_phi_file, "r") do f
    sims = sort(filter(k -> startswith(k, "sim_"), keys(f)), by=k->parse(Int, k[5:end]))
    for k in sims
        !haskey(f, "$k/ϕ_qe_wf") && continue
        ϕ_true = vec(Float64.(read(f, "$k/ϕ_true")))
        ϕ_qe   = vec(Float64.(read(f, "$k/ϕ_qe_wf")))
        # Pearson correlation as quality metric
        σt = std(ϕ_true); σq = std(ϕ_qe)
        corr = (σt > 0 && σq > 0) ? mean((ϕ_true .- mean(ϕ_true)) .* (ϕ_qe .- mean(ϕ_qe))) / (σt*σq) : 0.0
        if corr < worst_corr
            worst_corr = corr
            worst_sim = parse(Int, k[5:end])
        end
        n_scanned += 1
    end
end

@printf "Scanned %d UL sims. Worst sim: sim_%d (QE-true correlation = %.4f)\n" n_scanned worst_sim worst_corr

# Also report best sim for reference
best_sim = -1; best_corr = -Inf
jldopen(ul_phi_file, "r") do f
    sims = filter(k -> startswith(k, "sim_"), keys(f))
    for k in sims
        !haskey(f, "$k/ϕ_qe_wf") && continue
        ϕ_true = vec(Float64.(read(f, "$k/ϕ_true")))
        ϕ_qe   = vec(Float64.(read(f, "$k/ϕ_qe_wf")))
        σt = std(ϕ_true); σq = std(ϕ_qe)
        corr = (σt > 0 && σq > 0) ? mean((ϕ_true .- mean(ϕ_true)) .* (ϕ_qe .- mean(ϕ_qe))) / (σt*σq) : 0.0
        if corr > best_corr
            best_corr = corr
            best_sim = parse(Int, k[5:end])
        end
    end
end
@printf "Best sim: sim_%d (correlation = %.4f)\n\n" best_sim best_corr

# ── Load maps for worst sim ───────────────────────────────────────────────────
function load_sim_maps(phi_file, map_phi_file, sim_idx)
    maps = Dict{String, Any}()
    jldopen(phi_file, "r") do f
        key = "sim_$sim_idx"
        haskey(f, "$key/ϕ_true")  && (maps["ϕ_true"] = wrap(read(f, "$key/ϕ_true")))
        haskey(f, "$key/ϕ_qe_wf") && (maps["ϕ_qe"]   = wrap(read(f, "$key/ϕ_qe_wf")))
        haskey(f, "$key/ϕ_gi_b")  && (maps["ϕ_gi"]   = wrap(read(f, "$key/ϕ_gi_b")))
    end
    if map_phi_file !== nothing && isfile(map_phi_file)
        jldopen(map_phi_file, "r") do f
            key = "sim_$sim_idx"
            if haskey(f, "$key/ϕ_mj")
                maps["ϕ_mj"] = wrap(read(f, "$key/ϕ_mj"))
            end
        end
    end
    return maps
end

println("Loading maps for worst sim (sim_$worst_sim)...")
maps_ul = load_sim_maps(ul_phi_file,
                        isfile("results/phi_maps_map_12000_ul.jld2") ? "results/phi_maps_map_12000_ul.jld2" : nothing,
                        worst_sim)
maps_s4 = load_sim_maps(s4_phi_file,
                        isfile("results/phi_maps_map_12000.jld2") ? "results/phi_maps_map_12000.jld2" : nothing,
                        worst_sim)

if isempty(maps_ul)
    println("sim_$worst_sim not found in UL phi maps file. May not have been computed yet.")
    exit(1)
end

# ── Load W_L for debiasing ────────────────────────────────────────────────────
function load_wl(wl_file, key)
    isfile(wl_file) || return nothing
    d = JLD2.load(wl_file)
    !haskey(d, key) && return nothing
    ℓ = Float64.(d["ℓ_template"])
    W = Float64.(d[key])
    return Cℓs(ℓ, W)
end

WL_qe_ul = load_wl("results/WL_qe_gi_12000_ul.jld2", "W_qe_raw")
WL_gi_ul = load_wl("results/WL_qe_gi_12000_ul.jld2", "W_gi_b")
WL_qe_s4 = load_wl("results/WL_qe_gi_12000.jld2",    "W_qe_raw")
WL_gi_s4 = load_wl("results/WL_qe_gi_12000.jld2",    "W_gi_b")

function debias_or_raw(ϕ_field, WL; minW=1e-4)
    WL === nothing && return ϕ_field
    debias_phi_with_WL(ϕ_field, WL; minW=minW)
end

# ── Convert to κ for plotting ─────────────────────────────────────────────────
function get_κ(maps, wl_qe, wl_gi)
    out = Dict{String, Any}()
    haskey(maps, "ϕ_true") && (out["κ_true"] = phi_to_kappa(maps["ϕ_true"]))
    haskey(maps, "ϕ_qe")   && (out["κ_qe"]   = phi_to_kappa(debias_or_raw(maps["ϕ_qe"], wl_qe)))
    haskey(maps, "ϕ_gi")   && (out["κ_gi"]   = phi_to_kappa(debias_or_raw(maps["ϕ_gi"], wl_gi)))
    haskey(maps, "ϕ_mj")   && (out["κ_mj"]   = phi_to_kappa(maps["ϕ_mj"]))
    return out
end

println("Converting to κ maps...")
κ_ul = get_κ(maps_ul, WL_qe_ul, WL_gi_ul)
κ_s4 = get_κ(maps_s4, WL_qe_s4, WL_gi_s4)

# ── Plotting ──────────────────────────────────────────────────────────────────
PythonPlot.rc("font", family="serif", size=10)

keys_to_plot = ["κ_true", "κ_qe", "κ_gi", "κ_mj"]
labels_plot  = [L"\kappa_\mathrm{true}", L"\hat\kappa_\mathrm{QE}", L"\hat\kappa_\mathrm{GI}", L"\hat\kappa_\mathrm{MAP}"]
rows = [("UL (0.1 µK)", κ_ul), ("S4 (1 µK)", κ_s4)]

ncols = sum(haskey(κ_ul, k) || haskey(κ_s4, k) for k in keys_to_plot)
nrows = 2 * length(rows)   # κ map + difference map per estimator row

fig, axs = PythonPlot.subplots(nrows, ncols; figsize=(4*ncols, 3.5*nrows),
                                constrained_layout=true)
getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

# Colour scale from true κ of UL (or S4 fallback)
κ_ref_arr = haskey(κ_ul, "κ_true") ? κ_ul["κ_true"].arr : κ_s4["κ_true"].arr
vscale = 2 * std(κ_ref_arr)

fig.suptitle("Worst UL QE outlier: sim_$worst_sim  (QE–true correlation = $(@sprintf("%.3f", worst_corr)))",
             fontsize=12)

for (row_pair, (row_label, κ_dict)) in enumerate(rows)
    row_κ    = 2*(row_pair-1)
    row_diff = 2*(row_pair-1) + 1
    col = 0
    for (ki, k) in enumerate(keys_to_plot)
        (haskey(κ_ul, k) || haskey(κ_s4, k)) || continue
        lbl = labels_plot[ki]

        ax_κ = getax(row_κ,    col)
        ax_d = getax(row_diff, col)

        if haskey(κ_dict, k)
            arr = κ_dict[k].arr
            ax_κ.imshow(arr; cmap="RdBu_r", vmin=-vscale, vmax=vscale, origin="lower")
            ax_κ.set_title("$row_label  $lbl", fontsize=9)
            ax_κ.axis("off")

            # Difference map vs true
            if haskey(κ_dict, "κ_true") && k != "κ_true"
                diff = arr .- κ_dict["κ_true"].arr
                dscale = 2 * std(diff)
                im = ax_d.imshow(diff; cmap="RdBu_r", vmin=-dscale, vmax=dscale, origin="lower")
                ax_d.set_title(L"\hat\kappa - \kappa_\mathrm{true}" * "  RMS=$(@sprintf("%.2e", std(diff)))", fontsize=8)
                ax_d.axis("off")
            else
                ax_d.axis("off")
            end
        else
            ax_κ.axis("off"); ax_d.axis("off")
            ax_κ.set_title("$row_label  $lbl (N/A)", fontsize=9)
        end
        col += 1
    end
end

outpath = "results/diag_phi_outlier_sim$(worst_sim).png"
fig.savefig(outpath; dpi=150)
println("\nSaved: $outpath")
PythonPlot.plotclose("all")

# ── Print pixel-space stats ───────────────────────────────────────────────────
println("\nPixel-space statistics for sim_$worst_sim:")
println("  $(rpad("estimator", 20)) $(rpad("corr(κ̂,κ_true)", 18)) RMS(κ̂-κ_true)")
for (row_label, κ_dict) in rows
    haskey(κ_dict, "κ_true") || continue
    κt = vec(κ_dict["κ_true"].arr)
    for (k, lbl) in [("κ_qe","QE"), ("κ_gi","GI"), ("κ_mj","MAP")]
        haskey(κ_dict, k) || continue
        κh = vec(κ_dict[k].arr)
        σt = std(κt); σh = std(κh)
        corr = σt>0 && σh>0 ? mean((κt.-mean(κt)).*(κh.-mean(κh)))/(σt*σh) : NaN
        rms  = sqrt(mean((κh.-κt).^2))
        @printf "  %-20s %-18.4f %.3e\n" "$row_label $lbl" corr rms
    end
end
