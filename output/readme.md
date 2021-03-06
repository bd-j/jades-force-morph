Working with output
===================

For a given _run_ with a given starting configuration and input catalog,
multiple _patches_ (regions on the sky and sub-catalogs) will have been fit
using MCMC. There are two principal kinds of output, *samples* and *residuals*.
These are specific to each patch.  Therefore, the information is organized in a
directory structure that looks something like:

* `root_path/` - the base output directory for a given _run_
  * `<config_file>.yml` - The input configuration file used for the run.
  * `config.json` - The full configuration dictionary used for the run as a json
    file; This includes any command line overrides of the config file parameters
  * `<outname>.fits` - The full catalog with current parameter values.  Also
    includes information like number of MCMC samples per object.
  * `<outname>_log.json` - A record of which sources (as indices) appeared in
    each patch, and the order in which the patches are run.
  * `patches/`
     * `patch<i>_samples.h5` - information about the _i_ th patch and resulting
       MCMC chain
     * `patch<i>_residuals.h5` - residuals (data - model) and other pixel data
       for the last iteration in the chain for the _i_ th patch

We can read in the meta information as:

```python
import json
from astropy.io import fits

root = "root_path"
outname = "outscene_jades-morph"

with open(f"{root}/{outname}_log.json") as f:
    logs = json.load(f)
    slog = logs["sourcelog"]
    plog = logs["patchlog"]

outcat = fits.getdata(f"{root}/{outname}.fits")
with open(f"{root}/config.json") as f:
    config = json.load(f)
```

Units
-----

The units of the catalogs are generally decimal degrees for `ra` and `dec`,
radians *North of East* (I know...) for `pa`, arcseconds for `rhalf`, and nJy
for `fluxes`.  Note that for `q` the default units are actually sqrt(b/a).

Samples
-------

This output contains info from the MCMC chain, as well as a great deal of info
about the patch itself and the way the fitting happened. We read it in as
something like:

```python
from forcepho.reconstruction import Samples
i = plog[0]
samples = Samples(f"{root}/patches/patch{i}_samples.h5")
```

Now we have an object which has some important attributes.  These include:

* `bands` - A list of strings giving the bands that were fit simultaneously.
  These are also the column names for fluxes in various catalogs.

* `shape_cols` - A list of strings giving the column names for fitted position
  and shape parameters

* `chaincat` - numpy structured array of parameter values in the MCMC chain.
  There is one row per active source, and each field will generally have shape
  `(niter,)` except for the informational ields (`id`, etc.)

* `chain` a plain numpy array of shape `(niter, npar)` that gives the
  MCMC chain.  Generally not very useful, since it's can be hard to figure out which
  column is which parameter, and the positions are all relative to the patch
  center.  But since it's not structured it can be useful for performing various
  math operations.

* `active` - a structured ndarray that is a catalog of the active (i.e. fitted)
  sources in the patch, including the starting parameter positions

* `fixed` - a structured ndarray that is a catalog of the *fixed* i.e.
  (subtracted but _not_ fitted) sources in the patch.

* `final` - like `active`, but containing the final parameter positions at the
  end of sampling (i.e. the state at the last iteration)

* `bounds` - a structured array giving the constraints (lower an upper bounds)
  used for each of the fitted parameters.  It has fields like `active`, but with
  2 elements (lo, hi) for each fitted parameter instead of 1

* `cov` - ndarray of shape `(npar, npar)` that gives the inverse mass matrix
  used in sampling.  However, this matrix is defined in the *unconstrained*
  parameter space, which is related to the constrained parameter space by a
  non-linear transform, and thus is not super useful except as a diagnostic.

There are other attributes (`config`, `ncall`, `wall_time`, `stats`, `ra`, `dec`,
`radius`, `reference_coordinates`, etc.) that may be useful. Then if we
wanted to, say plot the trace for the F606W flux of the third galaxy in the
patch, we would do:

```python
import matplotlib.pyplot as pl
fig, ax = pl.subplots()
ax.plot(samples.chaincat["F606W"][:, 2])
```
Or, to plot the trace for all the parameters of the third active source:
```python
bands = samples.bands
fig, axes = pl.subplots(6 + len(bands))
source_idx = 2
samples.show_chain(source_idx, bands=bands, axes=axes)
```

Residuals
---------

The residuals are stored in a basic HDF5 file.  The structure is

- BAND/expososureID/xpix
- BAND/expososureID/ypix
- BAND/expososureID/data
- BAND/expososureID/ierr
- BAND/expososureID/residual

where each of these is a simple 1-d array. The residual is in the sense data
minus model, where the model is usually the last iteration of the chain for that
patch. But there are convenience methods for displaying these:

```python
from forcepho.reconstruction import Residuals
filename = f"{root}/patches/patch{i}_residuals.h5"
res = Residuals(filename)

# all the exposures
print(res.exposures)

# show the second exposure (e=1) with scale +/- 1 nJy
# this shows data, residual, and model from left to right
# you can also specify the exposure by name, using `exp='blah'` instead of `e=1`
fig, raxes = pl.subplots(1, 3)
data, model, delta = res.show(e=1, axes=raxes, vmin=-1, vmax=1)
# mark fixed sources on the residual:
res.mark_sources(samples.fixed["ra"], samples.fixed["dec"], e=1, axes=[raxes[1]])
# find pixel locations of a particular ra, dec in a particular exposure:
pix = res.sky_to_pix(samples.active["ra"], samples.active["dec"], e=1)
```

There's also a good deal of WCS info, but in highly non-standard format.
However, you can also use the `Residuals` object to place the data or residuals
into a larger image (though without WCS info at the moment.)  Note that for the HLF/UDF
the full images were broken down into 2048 x 2048 tiles, so that's what gets
generated by this method.

```python

subdir = "images"
os.makedirs(f"{root}/{subdir}", exist_ok=True)

deltas = {}  # images will be stored here keyed by exposureID

# we loop over patches in order so each pixel will be represented by the most
# recent patch/model that touched it
for i in plog:
    r = Residuals(f"{root}/patches/patch{i}_residuals.h5")
    # The imshape tells the size of the original exposure
    # use fill_type = "data" or "ierr" to get the original pixel fluxes or inverse error.
    r.fill_images(deltas, fill_type="residual", imshape=(2048, 2048))

# Note this does *not* include WCS info; that capability to be added soon.
for expID, im in deltas.items():
    band, exp = expID.decode("utf-8").split("/")
    fits.writeto(f"{root}/{subdir}/{exp}_delta.fits", im)
```
