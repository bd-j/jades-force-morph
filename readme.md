Jades-force-morph
======

Applying forcepho to JADES simulated images to test morphology estimates.


Install on lux
--------------------

```sh
export PROJECT_DIR=$HOME/jades-force-morph
```

Install:

```sh
# Get miniconda

cd $HOME
curl -sL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# get forcepho
cd $HOME
module load git
git clone https://github.com/bd-j/forcepho

# can re-run this block to update forcepho
cd $HOME/forcepho
git pull
module purge
module load cuda10.2 hdf5/1.10.6 gcc git slurm
#conda env create -f environment.yml
conda activate force
python -m pip install .


cd $HOME
git clone https://github.com/bd-j/jades-force-morph
```


Organization
------------

* `data/` soft links to the original images, 2048^2 cutouts, and data stores used by forcepho.  Also input and rectified catalogs.

* `pho/` scripts for pre-processing, optimization, sampling, and post-processing, along with utilities

* `jobs/` slurm scripts for running on lux

* `output/` results of fitting runs.
