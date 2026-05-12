import Pkg; Pkg.activate(@__DIR__)

using JLD2, CMBLensing, Statistics, Printf

phi_file = "results/phi_maps_qe_gi_12000_ul.jld2"
wl_file  = "results/WL_qe_gi_12000_ul.jld2"

Δℓ = 30

# Load W_GI
dwl = JLD2.load(wl_file)
ell_w = Float64.(dwl["ℓ_template"])
W_gi  = Float64.(dwl["W_gi_b"])
WL_gi = Cℓs(ell_w, W_gi)

# Build field wrapper directly — avoids calling camb/load_sim (no Python needed)
wrap(arr) = FlatMap(Float64.(arr); θpix=0.7438046267475303)

raw_sims = Vector{Vector{Float64}}()
sub_sims = Vector{Vector{Float64}}()
n0_sims  = Vector{Vector{Float64}}()
ell_ref  = Ref{Union{Nothing, Vector{Float64}}}(nothing)

jldopen(phi_file, "r") do f
    sims = sort([parse(Int, match(r"^sim_(\d+)$", k).captures[1])
                 for k in keys(f) if match(r"^sim_(\d+)$", k) !== nothing])

    println("Scanning $(length(sims)) sims...")

    for s in sims
        haskey(f, "sim_$s/ϕ_gi_b") || continue
        haskey(f, "sim_$s/N0_gi_fgmc") || continue
        haskey(f, "sim_$s/N0_gi_fgmc_ell") || continue

        ϕg = wrap(read(f, "sim_$s/ϕ_gi_b"))

        cl_raw_obj = get_Cℓ(ϕg; Δℓ=Δℓ)
        ell = Float64.(collect(cl_raw_obj.ℓ))
        cl_raw = Float64.(cl_raw_obj.Cℓ)

        ell_ref[] === nothing && (ell_ref[] = ell)

        n0_ell = Float64.(read(f, "sim_$s/N0_gi_fgmc_ell"))
        n0_raw = Float64.(read(f, "sim_$s/N0_gi_fgmc"))
        itp = Cℓs(n0_ell, n0_raw)
        n0_at = [L < minimum(n0_ell) || L > maximum(n0_ell) ? 0.0 : Float64(itp(L)) for L in ell]

        W_at = Float64.(WL_gi.(ell))
        W2 = max.(W_at.^2, 0.2^2)

        kfac = @. (ell^2 / 2)^2
        raw_k = kfac .* (cl_raw ./ W2)
        n0_k  = kfac .* (n0_at ./ W2)
        sub_k = kfac .* ((cl_raw .- n0_at) ./ W2)

        push!(raw_sims, raw_k)
        push!(n0_sims,  n0_k)
        push!(sub_sims, sub_k)
    end
end

ell_ref[] === nothing && error("No sims with N0_gi_fgmc_ell found in $phi_file")
ell_ref = ell_ref[]

println("Used $(length(raw_sims)) sims with GI fg-MC N0")

raw_mat = reduce(hcat, raw_sims)
n0_mat  = reduce(hcat, n0_sims)
sub_mat = reduce(hcat, sub_sims)

raw_mean = mean(raw_mat; dims=2)[:]
n0_mean  = mean(n0_mat;  dims=2)[:]
sub_mean = mean(sub_mat; dims=2)[:]
raw_std  = std(raw_mat; dims=2)[:]
n0_std   = std(n0_mat;  dims=2)[:]
sub_std  = std(sub_mat; dims=2)[:]

println()
println("UL GI fg-MC N0 size check")
println("L-bin       <raw>        <N0>       <sub>      N0/raw  std_N0/std_raw  std_sub/std_raw")
println("-" ^ 90)

for (lo, hi) in [(4000,6000), (6000,8000), (8000,10000), (10000,12000)]
    idx = findall(L -> lo <= L < hi, ell_ref)
    isempty(idx) && continue

    μraw = mean(raw_mean[idx])
    μn0  = mean(n0_mean[idx])
    μsub = mean(sub_mean[idx])
    σraw = mean(raw_std[idx])
    σn0  = mean(n0_std[idx])
    σsub = mean(sub_std[idx])

    @printf "%5d-%-5d  %.3e  %.3e  %.3e   %.3f     %.3f           %.3f\n" lo hi μraw μn0 μsub μn0/μraw σn0/σraw σsub/σraw
end

println()
println("Interpretation:")
println("  N0/raw: fraction of mean auto power that is noise bias.")
println("  std_N0/std_raw ~ 0: N0 is nearly constant across sims (fixed-gradient property).")
println("    => subtraction removes mean bias only; std_sub/std_raw stays ~ 1.")
println("    => covariance matrix shape unchanged by fg-MC N0 subtraction (physical, not a bug).")
println("  std_N0/std_raw ~ 1: N0 fluctuates as much as raw => subtraction can whiten covariance.")
println("  negative or huge N0/raw: filter/storage/interpolation mismatch.")
