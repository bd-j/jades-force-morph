config file
-----------

`morph_config.py` - Basic configuration file.  Note that you must set the environment variable `PROJECT_DIR`

```sh
export PROJECT_DIR=/path/to/jades-force-morph
```

pre-processing scripts
----------------------

* `preprocess.py` - This script makes the pixel and image meta-data storage
                    structures, after reading the config file

* `smallcat.py` - This script reads the input detection catalog and filters out
                  bright sources, plus makes small changes to the catalog data
                  format.

* `background.py` - This script fits a global background tweak to residual
                    images (using sigma clipping). These backgrounds should be
                    subtracted from images before a final run.  The residual
                    images can come from an optimization run, or an agressively
                    masked image, or..


Steps
--------------------------

1. Create PSF mixtures for mosaic (and individual slopes)

2. Pre-Process (`preprocess.py`)

   This creates the HDF5 storage files for pixel and meta-data.
   If slopes are present, make separate stores for mosaic and slope pixels.


3. Create initial catalog (`smallcat.py`)

   Makes some small tweaks to the detection catalog values (e.g. reversing
   sign of PA). Use the result as `raw_catalog` in the config file.

4. Background subtraction & optimization loop

   * Optimize sources in the small catalog (`optimize.py`) based on mosaic.

   * Fit for a residual background (`background.py`) in the mosaic.
     If significant, put resulting tweak values in config file.

   * Look for objects missing in the initial catalog?

   * Re-optimize sources in the small catalog (`optimize.py`) based on mosaic.
     Replace initialization catalog with the optimization results.

5. Sample posterior for source properties (`sample.py`, `sample.sh`)

6. Post-process to create residual images (if available), show chains, etc...