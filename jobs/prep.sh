#!/bin/bash
#SBATCH --job-name=fpho_jades_prep   # Job name
#SBATCH --partition=comp-astro       # queue for job submission
#SBATCH --account=comp-astro         # queue for job submission
#SBATCH --ntasks=1                   # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --time=3:00:00               # Time limit hrs:min:sec
#SBATCH --output=morphprep_%j.log      # Standard output and error log

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
cd $PROJECT_DIR/pho

config=$PROJECT_DIR/pho/morph_mosaic_config.yml
#fullsize_ims=$PROJECT_DIR/data/images/mosaics/*final/*[WM].fits
#cutID=jades-morph-mosaic
#max_snr=100

python preprocess.py --config_file $config
date
