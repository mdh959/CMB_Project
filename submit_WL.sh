#!/bin/bash
#SBATCH --job-name=WL_sims
#SBATCH --partition=icelake          # CSD3 icelake partition
#SBATCH --account=<YOUR_ACCOUNT>     # e.g. DAMTP-SL3-CPU — ask your supervisor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4            # Julia can use threads for linear algebra
#SBATCH --mem=32G                    # 512x512 maps, 100 sims — 32G should suffice
#SBATCH --time=24:00:00              # wall-time limit (adjust as needed)
#SBATCH --output=logs/WL_sims_%j.out
#SBATCH --error=logs/WL_sims_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<md959@cam.ac.uk>

# ── Load modules ──
module purge
module load rhel8/default-icl
module load julia                    # adjust if version-specific, e.g. julia/1.10

# ── Set threading ──
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ── Run ──
cd $HOME/CMB_Project
julia --project=. run_WL_sims.jl
