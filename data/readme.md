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

The PSF Gaussian approximations are fitted to PSFs made from stars in DC2 mosaics.  The parameters of these PSFs are placed in a HDF5 file in the `stores/` directory.


Image locations
--------------

Original slopes are on lux at
`/data/groups/comp-astro/jades/DC2/Morphology/slopes/`
and the versions with 'Sandro' format are `*smr.fits`.
Units should be nJy/pixel

The mosaics are on lux at
`/data/groups/comp-astro/jades/DC2/mosaics/morph_v1/<BAND>_final/`
and there are separate images for flux (`<BAND>.fits`), err (`<BAND>_err.fits`), exptime (`<BAND>_exp.fits`), and various convolutions and repixelizations (`<BAND>_conv*.fits`)
Units should be counts/sec but there should be an "ABMAG" keyword.

The mosaics have been tiled into 2048x2048 images and converted to HDF5 storage in forcepho format.  Link to the following directories and files to access them:

```sh
# Raw images and cutouts
export PROJECT_DIR=$HOME/jades-force-morph
cd $PROJECT_DIR/data
mkdir -p /data/groups/comp-astro/jades/fpho/images/jades-morph/cutouts
ln -s /data/groups/comp-astro/jades/fpho/images/jades-morph/cutouts cutouts
mkdir images
ln -s /data/groups/comp-astro/jades/DC2/mosaics/morph_v1/ images/mosaics

# Pixel and meta stores
cd $PROJECT_DIR/data/stores
ln -s /data/groups/comp-astro/jades/fpho/stores/pixels_jades-morph-mosaic.h5 pixels_jades-morph-mosaic.h5
ln -s /data/groups/comp-astro/jades/fpho/stores/meta_jades-morph-mosaic.json meta_jades-morph-mosaic.json
```
