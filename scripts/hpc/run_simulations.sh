#!/bin/bash
#!
#! SLURM job script for running 10 identical simulations
#! Modified from Peta4-IceLake template
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#SBATCH -J censim_array
#SBATCH -A HENDERSON-SL3-CPU
#SBATCH -p icelake
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mail-type=NONE
#SBATCH --array=1-10
#SBATCH --output=logs/simulation_%A_%a.out
#SBATCH --error=logs/simulation_%A_%a.err

#!#############################################################
#! Environment setup
#!#############################################################

#! Load basic environment
. /etc/profile.d/modules.sh
module purge
module load rhel8/default-icl

#! Set working directory
workdir="$SLURM_SUBMIT_DIR"
cd $workdir
echo -e "Changed directory to `pwd`.\n"

#! Job information
JOBID=$SLURM_JOB_ID
TASKID=$SLURM_ARRAY_TASK_ID

echo -e "JobID: $JOBID\n======"
echo "Array Task ID: $TASKID"
echo "Time: `date`"
echo "Running on node: `hostname`"
echo "Current directory: `pwd`"
echo -e "\n"

#!#############################################################
#! Singularity container setup
#!#############################################################

#! Path to your Singularity container
CONTAINER="$HOME/censim.sif"

#! Check if container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Singularity container not found at $CONTAINER"
    echo "Please build and transfer the container first:"
    echo "  On your local machine: sudo singularity build censim.sif censim.def"
    echo "  Then transfer: scp censim.sif mab282@login.hpc.cam.ac.uk:~/"
    exit 1
fi

echo "Using Singularity container: $CONTAINER"
echo -e "\n"

#!#############################################################
#! Run the simulation
#!#############################################################

#! Create output directory for this simulation
OUTPUT_DIR="output/simulation_${TASKID}"
mkdir -p $OUTPUT_DIR

#! Run Python simulation inside the Singularity container
echo "Starting simulation $TASKID..."
echo "Output directory: $OUTPUT_DIR"

singularity exec $CONTAINER python \
    ./replicated-assemblies-centromere-study/scripts/full_sim.py \
    --sequence-file ./replicated-assemblies-centromere-study/data/15000copy_cen178.seq \
    --pos-file ./replicated-assemblies-centromere-study/data/15000copy_cen178.178bp.bed.pos \
    --output-dir $OUTPUT_DIR

echo "Simulation $TASKID completed at: `date`"
