Data for jades-force-morph
===============

* `images/` original mosaics

* `cutouts/` 2048^2 cutouts

* `stores/` HDF5 data stores used by forcepho

* `catalogs/`

Image data consists of mosaics (and slopes?) produced by guitarra and the mosaicing pipeline.

The original image units are ?

The images are parsed into 2048 x 2048 sub-images, maintaining the astrometry,
and then converted to an image store. The meta-data for these sub-images (i.e., the
headers and WCS) are put in the the meta-data store.

PSFs
----

The PSF Gaussian approximations are fitted to PSFs made from stars in DC2 mosaics.  The parameters of these PSFs are placed in a HDF5 file in the stores/ directory.


Lux
---

Data locations

```sh
# Raw images and cutouts
export PROJECT_DIR=$HOME/jades-force-morph
cd $PROJECT_DIR/data/images
ln -s /data/groups/comp-astro/jades/<FIXME> images
ln -s /data/groups/comp-astro/jades/<FIXME> cutouts

# Pixel and meta stores
cd $PROJECT_DIR/data/stores
ln -s /data/groups/comp-astro/jades/fpho/stores/pixels_jades-morph.h5 pixels_jades-morph.h5
ln -s /data/groups/comp-astro/jades/fpho/stores/meta_jades-morph.json meta_jades-morph.json
```
