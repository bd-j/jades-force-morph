#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

import matplotlib.pyplot as pl
from astropy.io import fits
from astropy.wcs import WCS

from forcepho.patches import FITSPatch, CPUPatchMixin, GPUPatchMixin
from forcepho.superscene import LinkedSuperScene
from forcepho.utils import write_to_disk, NumpyEncoder
from forcepho.fitting import run_lmc
from forcepho.postprocess import Samples
from forcepho.patches.storage import MetaStore, header_to_id


from test_plot import plot_trace, plot_corner, plot_residual
from test_plot import make_catalog, compare_parameters, compare_apflux

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


__all__ = ["make_image", "fit_image"]


if HASGPU:
    class Patcher(FITSPatch, GPUPatchMixin):
        pass
else:
    class Patcher(FITSPatch, CPUPatchMixin):
        pass

default_tweakbg = {
           "F090W": 0.0125,
           "F115W": 0.0125,
           "F150W": 0.0125,
           "F200W": 0.0125,
           "F277W": 0.0125,
           "F335M": 0.0175,
           "F356W": 0.0125,
           "F410M": 0.0175,
           "F444W": 0.0125
          }


def clean_image(image_name, clean_image_name, bg_tweaks=None, bg_value=0.0):
    with fits.open(image_name) as hdul:
        im = hdul[1].data
        unc = hdul[2].data
        bkg = hdul[3].data
        msk = hdul[4].data
        hdr = hdul[1].header

    band = hdr["FILTER"]
    if type(bg_tweaks) is dict:
        bg_tweak = bg_tweaks.get(band, 0)
    else:
        bg_tweak = bg_value
    print(f"BG tweak for {band} = {bg_tweak}")

    zp = hdr["ABMAG"]
    conv = 1e9 * 10**(0.4 * (8.9 - zp))
    conv = 1.0

    bsub = (im - bkg)*conv - bg_tweak
    err = unc.copy() * conv
    err[msk.astype(bool)] *= -1
    bsub[msk.astype(bool)] = 0

    hdul_out = fits.HDUList([fits.PrimaryHDU(header=hdr),
                            fits.ImageHDU(bsub, hdr),
                            fits.ImageHDU(err, hdr)])

    hdul_out[0].header["BGTWEAK"] = bg_tweak
    hdul_out[0].header["BUNIT"] = "nJy"
    #print(f"writing to {clean_image_name}")
    hdul_out.writeto(clean_image_name, overwrite=True)
    hdul_out.close()


def fit_image(cat, config):

    # build the scene server
    bands = config.bands
    bounds_kwargs = dict(n_pix=2.0,
                         rhalf_range=(0.03, 1.0),
                         sersic_range=(0.8, 6.0))
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               statefile=os.path.join(config.outdir, "final_scene.fits"),
                               roi=cat["rhalf"] * 5,
                               bounds_kwargs=bounds_kwargs,
                               target_niter=config.sampling_draws)

    # load the image data
    patcher = Patcher(fitsfiles=config.clean_image_names,
                      psfstore=config.psfstore,
                      splinedata=config.splinedatafile,
                      sci_ext=1,
                      unc_ext=2,
                      return_residual=True)

    # check out scene & bounds
    region, active, fixed = sceneDB.checkout_region(seed_index=-1)
    bounds, cov = sceneDB.bounds_and_covs(active["source_index"])
    print(bounds.dtype)
    bounds[config.bands[0]][:, 0] = 0
    bounds[config.bands[0]][:, 1] = bounds[config.bands[0]][:, 1] * 5

    # prepare model and data, and sample
    patcher.build_patch(region, None, allbands=bands)
    model, q = patcher.prepare_model(active=active, fixed=fixed,
                                     bounds=bounds, shapes=sceneDB.shape_cols)
    out, step, stats = run_lmc(model, q.copy(),
                               n_draws=config.sampling_draws,
                               warmup=config.warmup,
                               z_cov=cov, full=True,
                               weight=max(10, active["n_iter"].min()),
                               discard_tuned_samples=False,
                               max_treedepth=config.max_treedepth,
                               progressbar=config.progressbar)

    # Check results back in and end and write everything to disk
    final, covs = out.fill(region, active, fixed, model, bounds=bounds,
                           step=step, stats=stats, patchID=0)
    write_to_disk(out, config.outroot, model, config)
    sceneDB.checkin_region(final, fixed, config.sampling_draws,
                           block_covs=covs, taskID=0)
    sceneDB.writeout()


def find_sources(image_name, fullcat, pad=30, origin=1):
    hdr = fits.getheader(image_name, 1)
    wcs = WCS(hdr)
    x, y = wcs.all_world2pix(fullcat["ra"], fullcat["dec"], origin)
    inim = ((x < (hdr["NAXIS1"] - pad)) &
            (x > pad) &
            (y < (hdr["NAXIS2"] - pad)) &
            (y > pad))
    return fullcat[inim]


def find_exposures(metastore, coordinates, bandlist):
    """Return a list of image namesfor all exposures that overlap the
    coordinates. These should be sorted by integer band_id.

    Parameters
    ----------
    region : region.Region instance
        Exposures will be found which overlap this region

    bandlist : list of str
        A list of band names to search for images.

    Returns
    -------
    imnames
    """
    wcs_origin = 1
    imsize = np.zeros(2) + 2048
    bra, bdec = coordinates

    epaths, bands = [], []
    for band in bandlist:
        if band not in metastore.wcs.keys():
            continue
        for expID in metastore.wcs[band].keys():
            #print(expID)
            epath = "{}/{}".format(band, expID)
            wcs = metastore.wcs[band][expID]
            # Check region bounding box has a corner in the exposure.
            # NOTE: If bounding box entirely contains image this might fail
            bx, by = wcs.all_world2pix(bra, bdec, wcs_origin)
            inim = np.any((bx > 0) & (bx < imsize[0]) &
                          (by > 0) & (by < imsize[1]))
            if inim:
                epaths.append(expID)
                bands.append(band)
    return epaths, bands


def make_image_sets(config, bandlist, single=False):
    meta = MetaStore(config.metastorefile)
    band = bandlist[0]

    elist = list(meta.headers[band].keys())
    allexps = []
    expsets, bandsets = [], []
    for exp in elist:
        if exp in allexps:
            continue
        hdr = meta.headers[band][exp]
        coordinates = np.array([hdr["CRVAL1"], hdr["CRVAL2"]])
        exp_set, band_set = find_exposures(meta, coordinates, bandlist)
        expsets.append(exp_set)
        bandsets.append(band_set)
        allexps += exp_set

    # reduce to a single exposure in each band
    if single:
        elists = [one_per_band(eset, bset)
                  for eset, bset in zip(expsets, bandsets)]
        expsets = elists

    return expsets


def one_per_band(expset, bandset):
    covered, explist = [], []
    for e, b in zip(expset, bandset):
        if b not in covered:
            covered.append(b)
            explist.append(e)
    return explist


def make_tag(config):
    return f"{'+'.join(config.bands)}_id{config.id}"


if __name__ == "__main__":

    print(f"HASGPU={HASGPU}")

    # ------------------
    # --- Configure ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--framedir", type=str, default="/data/groups/comp-astro/jades/DC2/Morphology/slopes")
    parser.add_argument("--metastorefile", type=str, default="./meta-morph.json")
    parser.add_argument("--bands", type=str, nargs="*", default=["F200W", "F277W"])
    parser.add_argument("--dir", type=str, default="")
    parser.add_argument("--initial_catalog", type=str, default="")
    # extras for image testing
    parser.add_argument("--set_number", type=int, default=0)
    parser.add_argument("--single_exposure", type=int, default=0)
    parser.add_argument("--tweak_background", type=int, default=1)
    parser.add_argument("--bg_value", type=float, default=0.0)
    # I/O
    parser.add_argument("--psfstore", type=str, default="")
    parser.add_argument("--splinedatafile", type=str, default="../data/stores/sersic_splinedata_large.h5")
    parser.add_argument("--write_residuals", type=int, default=1)
    # sampling
    parser.add_argument("--sampling_draws", type=int, default=2048)
    parser.add_argument("--max_treedepth", type=int, default=8)
    parser.add_argument("--warmup", type=int, nargs="*", default=[256])
    parser.add_argument("--progressbar", type=int, default=0)
    config = parser.parse_args()

    # --- make output directories, copy config ---
    os.makedirs(config.dir, exist_ok=True)
    # copy the config data
    with open(f"{config.dir}/config.json", "w") as cfg:
        json.dump(vars(config), cfg, cls=NumpyEncoder)
    pout = os.path.join(config.dir, os.path.basename(config.psfstore))
    shutil.copy(config.psfstore, pout)

    # --- find all images ---
    expsets = make_image_sets(config, config.bands, single=config.single_exposure)
    explist = expsets[config.set_number]
    config.image_names = [os.path.join(config.framedir, f"{exp}.fits") for exp in explist]
    # write the name of all the images in this set
    with open(f"{config.dir}/explist_{config.set_number}.dat", "w") as elist:
        for e in config.image_names:
            elist.write(f"{e}\n")

    sys.exit()

    # --- find catalog objects in all images ---
    fullcat = np.array(fits.getdata(config.initial_catalog))
    subcat = fullcat.copy()
    for image_name in config.image_names:
        subcat = find_sources(image_name, subcat)
    print(f"found {len(subcat)} sources")
    if len(subcat) < 3:
        sys.exit()
    # copy subcatalog
    fits.writeto(f"{config.dir}/initial_image_catalog_{config.set_number}.fits", subcat, overwrite=True)

    # --- clean the images ---
    if config.tweak_background:
        print("tweaking background using dic")
        bg_tweaks = default_tweakbg
    else:
        print("tweaking background based on bg_val")
        bg_tweaks = None
    config.clean_image_names = []
    config.bands = []
    for image_name in config.image_names:
        clean_image_name = os.path.join(config.dir, os.path.basename(image_name))
        clean_image_name.replace("smr", "cal")
        config.clean_image_names.append(clean_image_name)
        config.bands.append(fits.getheader(image_name, 1)["FILTER"])
        clean_image(image_name, clean_image_name,
                    bg_tweaks=bg_tweaks, bg_value=config.bg_value)

    config.bands = list(np.unique(config.bands))

    #sys.exit()

    # --- fit each object ---
    tags = []
    for row in subcat:

        # make directories and names
        config.id = row["id"]
        config.tag = make_tag(config)
        config.outdir = os.path.join(config.dir, config.tag)
        os.makedirs(config.outdir, exist_ok=True)
        config.outroot = os.path.join(config.outdir, config.tag)

        # --------------------
        # --- Fit the data ---
        try:
            fit_image(np.atleast_1d(row), config)
        except(AssertionError):
            continue

        # --------------------
        # --- make figures ---
        patchname = f"{config.outroot}_samples.h5"
        title = config.tag.replace("_", ", ")

        tfig, ax = plot_trace(patchname)
        tfig.suptitle(title)
        tfig.tight_layout()
        tfig.savefig(f"{config.outroot}_trace.png", dpi=200)
        pl.close(tfig)

        cfig, caxes = plot_corner(patchname)
        cfig.text(0.4, 0.8, title, transform=cfig.transFigure)
        cfig.savefig(f"{config.outroot}_corner.png", dpi=200)
        pl.close(cfig)

        rfig, raxes, rcb, val = plot_residual(patchname)
        rfig.savefig(f"{config.outroot}_residual.png", dpi=200)
        rfig.suptitle(title)
        pl.close(rfig)

        tags.append(config.outroot)

    # Make summary catalog
    scat = make_catalog(tags, bands=config.bands)
    fits.writeto(os.path.join(config.dir, "ensemble_chains.fits"),
                 scat, overwrite=True)

