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
#SBATCH --output=jmorph_sample_%j.log  # Standard output and error log

pwd; hostname; date

export LC_ALL=en_US.UTF-8
export LC_TYPE=en_US.UTF-8

module purge
module load cuda10.2 hdf5/1.10.6
source activate force

export PROJECT_DIR=$HOME/jades-force-morph
vers=v2.1
config=$PROJECT_DIR/pho/morph_mosaic_config.yml
incat=$PROJECT_DIR/data/catalogs/postop_v2.1_catalog.fits
outbase=$PROJECT_DIR/output/sampling_$vers
outcat=$outbase/full_chaincat.fits

echo "Running multi patch sampling for $config"
cd $PROJECT_DIR/pho
python sample.py --config_file $config \
                 --raw_catalog  $incat \
                 --add_barriers 0 \
                 --full_cov 0 \
                 --discard_tuning 0 \
                 --outbase $outbase \
                 --tweak_background tweakbg \

echo "Post-processing output at $outbase"
python postprocess.py --root $outbase --mode catalog --catname $outcat
python postprocess.py --root $outbase --mode images


date
