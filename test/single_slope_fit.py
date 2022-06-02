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

tweakbg = {
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


def clean_image(config, bg_tweak=0):
    with fits.open(config.image_name) as hdul:
        im = hdul[1].data
        unc = hdul[2].data
        bkg = hdul[3].data
        msk = hdul[4].data
        hdr = hdul[1].header

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
    print(f"writing to {config.clean_image_name}")
    hdul_out.writeto(config.clean_image_name, overwrite=True)
    hdul_out.close()


def fit_image(cat, config):

    # build the scene server
    bands = config.bands
    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               statefile=os.path.join(config.outdir, "final_scene.fits"),
                               roi=cat["rhalf"] * 5,
                               bounds_kwargs=dict(n_pix=1.5, rhalf_range=(0.03, 1.0), sersic_range=(0.8, 6.0)),
                               target_niter=config.sampling_draws)

    # load the image data
    patcher = Patcher(fitsfiles=[config.clean_image_name],
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


def find_sources(image_name, fullcat, pad=50, origin=1):
    hdr = fits.getheader(image_name, 1)
    wcs = WCS(hdr)
    x, y = wcs.all_world2pix(fullcat["ra"], fullcat["dec"], origin)
    inim = ((x < (hdr["NAXIS1"] - pad)) &
            (x > pad) &
            (y < (hdr["NAXIS2"] - pad)) &
            (y > pad))
    return fullcat[inim]


def make_tag(config):
    return f"{config.bands[0]}_id{config.id}"


if __name__ == "__main__":

    print(f"HASGPU={HASGPU}")

    # ------------------
    # --- Configure ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, default="")
    parser.add_argument("--initial_catalog", type=str, default="")
    parser.add_argument("--dir", type=str, default="")
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

    # --- find catalog objects in this image ---
    fullcat = np.array(fits.getdata(config.initial_catalog))
    subcat = find_sources(config.image_name, fullcat)
    # copy subcatalog
    fits.writeto(f"{config.dir}/initial_image_catalog.fits", subcat, overwrite=True)

    # --- clean the image ---
    config.bands = [fits.getheader(config.image_name, 1)["FILTER"]]
    config.clean_image_name = os.path.join(config.dir, os.path.basename(config.image_name))
    config.clean_image_name.replace("smr", "cal")
    #sys.exit()
    clean_image(config, bg_tweak=tweakbg[config.bands[0]])

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
        except:
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
        pl.close(rfig)

        tags.append(config.outroot)

    # Make summary catalog
    scat = make_catalog(tags, bands=config.bands)
    fits.writeto(os.path.join(config.dir, "ensemble_chains.fits"),
                 scat, overwrite=True)

