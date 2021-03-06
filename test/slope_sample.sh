#!/bin/bash
#SBATCH --mail-user=your_email@domain  # Where to send mail
#SBATCH --mail-type=None               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --job-name=jmorph_sample       # Job name
#SBATCH --partition=comp-astro         # queue for job submission
#SBATCH --account=comp-astro           # queue for job submission
#SBATCH --ntasks=1                     # Number of MPI ranks
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # How many tasks on each node
#SBATCH --time=3:00:00                 # Time limit hrs:min:sec
#SBATCH --output=logs/jmorph_slope_%j.log  # Standard output and error log

pwd; hostname; date

export LC_ALL=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

module purge
module load cuda10.2 hdf5/1.10.6
source activate force

export PROJECT_DIR=$HOME/jades-force-morph
export SLOPE_DIR=/data/groups/comp-astro/jades/DC2/Morphology/slopes/


cd $PROJECT_DIR/test

exp=F090W_488_0023
image_name=$SLOPE_DIR/MorphologyV07_${exp}.smr.fits
incat=$PROJECT_DIR/data/catalogs/truth_initial_catalog.fits
psfs=$PROJECT_DIR/data/stores/psf_jwst_oct21_ng4m0.h5

python single_slope_fit.py --image_name $image_name \
                           --initial_catalog $incat \
                           --psfstore $psfs \
                           --dir output/$exp


date
