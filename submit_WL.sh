#!/bin/bash
#!
#! W_L empirical computation via Monte Carlo simulations
#!

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J WL_sims
#! Which project should be charged:
#SBATCH -A HADZHIYSKA-SL3-CPU
#SBATCH -p icelake
#SBATCH --cluster=csd3
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#! How much wallclock time will be required?
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=md959@cam.ac.uk
##SBATCH --no-requeue

#! sbatch directives end here (put any additional directives above this line)

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')

#! Optionally modify the environment seen by the application
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-icl              # REQUIRED - loads the basic environment

#! Full path to application executable (using local Julia 1.9.4):
application="$HOME/julia-1.9.4/bin/julia --project=. run_WL_sims.jl"

#! Run options for the application:
options=""

#! Work directory (i.e. where the job will run):
workdir="$HOME/CMB_Project"

#! Threading for Julia and linear algebra libraries:
export JULIA_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! The following variables define a sensible pinning strategy for Intel MPI tasks -
#! this should be suitable for both pure MPI and hybrid MPI/OpenMP jobs:
export I_MPI_PIN_DOMAIN=omp:compact # Domains are $OMP_NUM_THREADS cores in size
export I_MPI_PIN_ORDER=scatter # Adjacent domains have minimal sharing of caches/sockets

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
CMD="$application $options"

###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
