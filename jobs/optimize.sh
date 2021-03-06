#!/bin/bash
#SBATCH --job-name=fpho_optimization # Job name
#SBATCH --partition=comp-astro       # queue for job submission
#SBATCH --account=comp-astro         # queue for job submission
#SBATCH --ntasks=1                   # Number of MPI ranks
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=1          # How many tasks on each node
#SBATCH --time=3:00:00               # Time limit hrs:min:sec
#SBATCH --output=jmorph_opt_%j.log   # Standard output and error log

pwd; hostname; date

export LC_ALL=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

module purge
module load cuda10.2 hdf5/1.10.6 openmpi
source activate force

export PROJECT_DIR=$HOME/jades-force-morph
config=$PROJECT_DIR/pho/morph_mosaic_config.yml
catalog=$PROJECT_DIR/data/catalogs/truth_initial_catalog.fits
vers=v2.0
outbase=$PROJECT_DIR/output/optimization_${vers}
outcat=$PROJECT_DIR/data/catalogs/postop_${vers}_catalog.fits

echo "Running optimization test for $config"
cd $PROJECT_DIR/pho
python optimize.py --config_file $config \
                   --raw_catalog $catalog \
                   --n_pix_sw 4 \
                   --minflux 0 --maxfluxfactor 3 \
                   --strict 1 --maxactive_per_patch 12 \
                   --add_barriers 1 --use_gradients 1 --gtol 1e-4 --linear_optimize 1 \
                   --outbase $outbase \
                   --tweak_background tweakbg \

echo "Post-processing output at $outbase"
python postprocess.py --root $outbase --mode postop --catname $outcat
python postprocess.py --root $outbase --mode images

date
