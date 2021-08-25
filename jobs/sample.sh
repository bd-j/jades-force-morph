#!/bin/bash
#SBATCH --job-name=jmorph_sample        # Job name
#SBATCH --partition=comp-astro         # queue for job submission
#SBATCH --account=comp-astro           # queue for job submission
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=benjamin.johnson@cfa.harvard.edu  # Where to send mail
#SBATCH --ntasks=1                     # Number of MPI ranks
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --ntasks-per-node=1            # How many tasks on each node
#SBATCH --time=3:00:00                 # Time limit hrs:min:sec
#SBATCH --output=jmorph_sample_%j.log  # Standard output and error log

pwd; hostname; date

export LC_ALL=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

module load cuda10.1 python/3.6.7 hdf5
module load numpy scipy h5py
module load numba pycuda
module load astropy
module load littlemcmc
module load openmpi mpi4py

export PROJECT_DIR=$HOME/jades-force-morph
config=$PROJECT_DIR/pho/morph_config.yml
raw=$PROJECT_DIR/data/catalogs/initial.fits
outbase=$PROJECT_DIR/output/sampling_v1

echo "Running multi patch sampling for $config"
cd $PROJECT_DIR/pho
python sample.py --config_file $config \
                 #--tweak_background tweakbg \
                 --raw_catalog  $raw \
                 --add_barriers 0 \
                 --full_cov 0 \
                 --discard_tuning 0 \
                 --outbase $outbase

date
