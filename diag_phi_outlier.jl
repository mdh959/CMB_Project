#!/usr/bin/env julia
#
# diag_phi_outlier.jl
#
# Finds the worst QE outlier (lowest pixel-space correlation ϕ_true vs ϕ_qe)
# and the best sim, then plots κ_true | κ_QE | κ_GI | κ_MAP side-by-side for
# both UL and S4, plus difference maps and pixel-space statistics.
# κ = -(ℓ²/2)ϕ, bandpass L=4000-10000.
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

(; ϕ) = load_sim(; seed=seed0+1, load_kw_s4...)
ϕ_ref = Map(ϕ)
wrap(arr) = typeof(ϕ_ref)(Float64.(arr), ϕ_ref.metadata)

# κ = -(ℓ²/2)ϕ, bandpass filtered in Fourier before converting to map
function phi_to_kappa(ϕ_field; Lmin=4000.0, Lmax=10000.0)
    ϕ_f  = Fourier(ϕ_field)
    ℓmag = fieldinfo(ϕ_f).ℓmag
    mask  = @. (ℓmag >= Lmin) & (ℓmag <= Lmax)
    κ_arr = @. -(ℓmag^2 / 2) * ϕ_f.arr * mask
    return Map(typeof(ϕ_f)(κ_arr, ϕ_f.metadata))
end

ul_phi_file = "results/phi_maps_qe_gi_12000_ul.jld2"
s4_phi_file = "results/phi_maps_qe_gi_12000.jld2"

if !isfile(ul_phi_file)
    println("UL phi maps file not found: $ul_phi_file")
    println("Run run_qe_gi_wl12k.jl first to generate UL results.")
    exit(1)
end

function scan_outliers(phi_file, label)
    println("\nScanning $label phi maps for worst QE outliers...")
    sim_corrs = Tuple{Int,Float64}[]
    jldopen(phi_file, "r") do f
        sims = sort(filter(k -> startswith(k, "sim_"), keys(f)), by=k->parse(Int, k[5:end]))
        for k in sims
            !haskey(f, "$k/ϕ_qe_raw") && continue
            ϕ_true = vec(Float64.(read(f, "$k/ϕ_true")))
            ϕ_qe   = vec(Float64.(read(f, "$k/ϕ_qe_raw")))
            σt = std(ϕ_true); σq = std(ϕ_qe)
            corr = (σt > 0 && σq > 0) ? mean((ϕ_true .- mean(ϕ_true)) .* (ϕ_qe .- mean(ϕ_qe))) / (σt*σq) : 0.0
            push!(sim_corrs, (parse(Int, k[5:end]), corr))
        end
    end
    sort!(sim_corrs, by=x->x[2])
    @printf "Scanned %d %s sims.\n" length(sim_corrs) label

    corrs_only = [c for (_,c) in sim_corrs]
    med   = median(corrs_only)
    σ_c   = std(corrs_only)
    threshold = med - 4*σ_c
    println("Correlation distribution: median=$(round(med,digits=4))  std=$(round(σ_c,digits=4))")
    println("Outlier threshold (median - 4σ): $(round(threshold,digits=4))")

    outlier_sims = [(s,c) for (s,c) in sim_corrs if c < threshold]
    println("Sims below threshold ($(length(outlier_sims))):")
    for (i, (s, c)) in enumerate(outlier_sims)
        @printf "  #%d: sim_%d  (corr = %.4f)\n" i s c
    end
    println("Top 5 worst regardless of threshold:")
    for (i, (s, c)) in enumerate(sim_corrs[1:min(5,end)])
        @printf "  #%d: sim_%d  (corr = %.4f)\n" i s c
    end
    exclude_set = sort([s for (s,_) in outlier_sims])
    println("Full below-threshold exclude set ($label):  Set($(exclude_set))")
    return sim_corrs, outlier_sims
end

ul_sim_corrs, ul_outliers = scan_outliers(ul_phi_file, "UL")
s4_sim_corrs, s4_outliers = isfile(s4_phi_file) ? scan_outliers(s4_phi_file, "S4") : (Tuple{Int,Float64}[], Tuple{Int,Float64}[])

# Union of outliers from both scans
all_outlier_sims = sort(collect(Set([s for (s,_) in ul_outliers] ∪ [s for (s,_) in s4_outliers])))
println("\nUnion exclude set (UL ∪ S4, paste into plot script):")
println("  Set($(all_outlier_sims))")

# Use UL results for the map plotting below (worst outliers by UL QE correlation)
sim_corrs   = ul_sim_corrs
outlier_sims = ul_outliers

function load_sim_maps(phi_file, map_phi_file, sim_idx)
    maps = Dict{String, Any}()
    jldopen(phi_file, "r") do f
        key = "sim_$sim_idx"
        haskey(f, "$key/ϕ_true")  && (maps["ϕ_true"] = wrap(read(f, "$key/ϕ_true")))
        haskey(f, "$key/ϕ_qe_raw") && (maps["ϕ_qe"]   = wrap(read(f, "$key/ϕ_qe_raw")))
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




function get_κ(maps)
    out = Dict{String, Any}()
    haskey(maps, "ϕ_true") && (out["κ_true"] = phi_to_kappa(maps["ϕ_true"]))
    haskey(maps, "ϕ_qe")   && (out["κ_qe"]   = phi_to_kappa(maps["ϕ_qe"]))   # raw: A_L-normalised, no W_L div
    haskey(maps, "ϕ_gi")   && (out["κ_gi"]   = phi_to_kappa(maps["ϕ_gi"]))   # raw: amplitude ~W_GI, fine visually
    haskey(maps, "ϕ_mj")   && (out["κ_mj"]   = phi_to_kappa(maps["ϕ_mj"]))
    return out
end

PythonPlot.rc("font", family="serif", size=10)

const keys_to_plot = ["κ_true", "κ_qe", "κ_gi", "κ_mj"]
const labels_plot  = [L"\kappa_\mathrm{true}", L"\hat\kappa_\mathrm{QE}", L"\hat\kappa_\mathrm{GI}", L"\hat\kappa_\mathrm{MAP}"]

map_ul_file = isfile("results/phi_maps_map_12000_ul.jld2") ? "results/phi_maps_map_12000_ul.jld2" : nothing
map_s4_file = isfile("results/phi_maps_map_12000.jld2")    ? "results/phi_maps_map_12000.jld2"    : nothing

# plot worst sims (up to 5, or all below threshold)
n_plot = max(length(outlier_sims), min(5, length(sim_corrs)))
for (rank, (target_sim, target_corr)) in enumerate(sim_corrs[1:n_plot])
    target_label = "worst$(rank)"

    println("\nLoading maps for $target_label sim (sim_$target_sim)...")
    maps_ul = load_sim_maps(ul_phi_file, map_ul_file, target_sim)
    maps_s4 = load_sim_maps(s4_phi_file, map_s4_file, target_sim)

    if isempty(maps_ul)
        println("sim_$target_sim not found in UL phi maps file — skipping")
        continue
    end

    println("Converting to κ maps...")
    κ_ul = get_κ(maps_ul)
    κ_s4 = get_κ(maps_s4)

    rows  = [("UL (0.1 µK)", κ_ul), ("S4 (1 µK)", κ_s4)]
    ncols = sum(haskey(κ_ul, k) || haskey(κ_s4, k) for k in keys_to_plot)
    nrows = 2 * length(rows)

    fig, axs = PythonPlot.subplots(nrows, ncols; figsize=(4*ncols, 3.5*nrows),
                                    constrained_layout=true)
    getax(r, c) = ncols == 1 ? axs[r] : axs[r, c]

    κ_ref_arr = haskey(κ_ul, "κ_true") ? κ_ul["κ_true"].arr : κ_s4["κ_true"].arr
    vscale    = 2 * std(κ_ref_arr)

    fig.suptitle("Outlier #$rank (sim_$target_sim)  QE–true correlation = $(@sprintf("%.3f", target_corr))  [κ bandpass L=4000–10000]",
                 fontsize=11)

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
                if haskey(κ_dict, "κ_true") && k != "κ_true"
                    diff   = arr .- κ_dict["κ_true"].arr
                    dscale = 2 * std(diff)
                    ax_d.imshow(diff; cmap="RdBu_r", vmin=-dscale, vmax=dscale, origin="lower")
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

    outpath = "results/diag_phi_outlier_rank$(rank)_sim$(target_sim).png"
    fig.savefig(outpath; dpi=150)
    println("Saved: $outpath")
    PythonPlot.plotclose("all")

    # pixel-space statistics
    println("\nPixel-space statistics for outlier #$rank (sim_$target_sim):")
    println("  $(rpad("estimator", 20)) $(rpad("corr(κ̂,κ_true)", 18)) RMS(κ̂-κ_true)")
    for (row_label, κ_dict) in rows
        haskey(κ_dict, "κ_true") || continue
        κt = vec(κ_dict["κ_true"].arr)
        for (k, lbl) in [("κ_qe","QE"), ("κ_gi","GI"), ("κ_mj","MAP")]
            haskey(κ_dict, k) || continue
            κh   = vec(κ_dict[k].arr)
            σt   = std(κt); σh = std(κh)
            corr = σt>0 && σh>0 ? mean((κt.-mean(κt)).*(κh.-mean(κh)))/(σt*σh) : NaN
            rms  = sqrt(mean((κh.-κt).^2))
            @printf "  %-20s %-18.4f %.3e\n" "$row_label $lbl" corr rms
        end
    end
end
