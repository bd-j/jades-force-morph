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
cd $HOME
module load git
git clone https://github.com/bd-j/forcepho

# can re-run this block to update forcepho
cd $HOME/forcepho
git pull
module purge
module load cuda10.1 python/3.6.7 numpy hdf5 git slurm
python setup.py install --user

pip install pyyaml --user

cd $HOME
git clone https://github.com/bd-j/jades-force-morph
```


Organization
------------

* `data/` soft links to the original images, 2048^2 cutouts, and data stores used by forcepho.  Also input and rectified catalogs.

* `pho/` scripts for pre-processing, optimization, sampling, and post-processing, along with utilities

* `jobs/` slurm scripts for running on lux

* `output/` results of fitting runs.
