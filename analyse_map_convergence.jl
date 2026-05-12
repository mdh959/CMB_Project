#!/usr/bin/env julia
# analyse_map_convergence.jl

import Pkg
Pkg.activate(@__DIR__)

using JLD2
using Statistics
using Printf

diag_file = "results/diag_map_12000_ul.jld2"

@assert isfile(diag_file) "Missing file: $diag_file"

println("="^100)
println("MAP convergence diagnostics")
println("File: $diag_file")
println("="^100)

rows = []

jldopen(diag_file, "r") do f
    sims = sort([
        parse(Int, match(r"^sim_(\d+)$", k).captures[1])
        for k in keys(f)
        if match(r"^sim_(\d+)$", k) !== nothing
    ])

    println("Found $(length(sims)) diagnostic entries\n")

    for s in sims
        base = "sim_$s"
        haskey(f, "$base/logpdf") || continue

        lp  = Float64.(read(f, "$base/logpdf"))
        ll  = haskey(f, "$base/loglike")  ? Float64.(read(f, "$base/loglike"))  : [NaN, NaN]
        pr  = haskey(f, "$base/logprior") ? Float64.(read(f, "$base/logprior")) : [NaN, NaN]
        ncg = haskey(f, "$base/nCG")      ? Float64.(read(f, "$base/nCG"))      : Float64[]
        rcg = haskey(f, "$base/resCG")    ? Float64.(read(f, "$base/resCG"))    : Float64[]

        finite_lp = all(isfinite, lp)

        Δlp = finite_lp ? lp[end] - lp[1] : NaN
        Δll = length(ll) >= 2 ? ll[end] - ll[1] : NaN
        Δpr = length(pr) >= 2 ? pr[end] - pr[1] : NaN

        last5 = length(lp) >= 6 && finite_lp ? lp[end] - lp[end-4] : NaN
        mean_step = length(lp) >= 2 && finite_lp ? Δlp / (length(lp)-1) : NaN

        last5_frac = isfinite(last5) && isfinite(mean_step) && abs(mean_step) > 0 ?
            abs((last5 / 5) / mean_step) : NaN

        cg_mean = isempty(ncg) ? NaN : mean(filter(isfinite, ncg))
        cg_last = isempty(ncg) ? NaN : ncg[end]
        res_last = isempty(rcg) ? NaN : rcg[end]

        status = if !finite_lp
            "BAD logpdf"
        elseif isfinite(Δll) && Δll < 0
            "likelihood worse"
        elseif isfinite(Δpr) && isfinite(Δll) && Δpr > abs(Δll)
            "prior dominated"
        elseif isfinite(last5_frac) && last5_frac > 0.05
            "still improving"
        else
            "converged-ish"
        end

        push!(rows, (
            sim=s,
            nstep=length(lp),
            Δlp=Δlp,
            Δll=Δll,
            Δpr=Δpr,
            last5=last5,
            last5_frac=last5_frac,
            cg_mean=cg_mean,
            cg_last=cg_last,
            res_last=res_last,
            status=status,
        ))
    end
end

@printf("%-6s %-6s %-12s %-12s %-12s %-12s %-12s %-10s %-10s %-12s %s\n",
    "sim", "steps", "Δlogpdf", "Δlike", "Δprior",
    "last5Δ", "last5frac", "CGmean", "CGlast", "resCG", "status")

println("-"^115)

for r in rows
    @printf("%-6d %-6d %-12.4g %-12.4g %-12.4g %-12.4g %-12.4g %-10.2f %-10.1f %-12.4g %s\n",
        r.sim, r.nstep, r.Δlp, r.Δll, r.Δpr,
        r.last5, r.last5_frac, r.cg_mean, r.cg_last, r.res_last, r.status)
end

println("\nSummary")
println("-"^100)

good = filter(r -> isfinite(r.Δlp), rows)

finite_vals(getter) = [getter(r) for r in good if isfinite(getter(r))]
safe_mean(vals) = isempty(vals) ? NaN : mean(vals)
safe_median(vals) = isempty(vals) ? NaN : median(vals)

if isempty(good)
    println("No finite logpdf histories found.")
else
    @printf("Mean Δlogpdf      = %.4g\n", safe_mean(finite_vals(r -> r.Δlp)))
    @printf("Mean Δloglike     = %.4g\n", safe_mean(finite_vals(r -> r.Δll)))
    @printf("Mean Δlogprior    = %.4g\n", safe_mean(finite_vals(r -> r.Δpr)))
    @printf("Mean last5 Δ      = %.4g\n", safe_mean(finite_vals(r -> r.last5)))
    @printf("Mean last5 frac   = %.4g\n", safe_mean(finite_vals(r -> r.last5_frac)))
    @printf("Mean CG iters     = %.2f\n", safe_mean(finite_vals(r -> r.cg_mean)))
    @printf("Median final res  = %.4g\n", safe_median(finite_vals(r -> r.res_last)))

    println("\nStatus counts:")
    for st in sort(unique(r.status for r in good))
        println("  ", rpad(st, 22), count(r -> r.status == st, good))
    end
end