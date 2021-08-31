# -*- coding: utf-8 -*-

"""utils.py - utilities for conducting fits.
"""
import os
import logging
import numpy as np
import h5py

from astropy.coordinates import SkyCoord

from forcepho.superscene import LinkedSuperScene
from forcepho.superscene import isophotal_radius, flux_bounds
from forcepho.fitting import optimize_fluxes, design_matrix
from forcepho.utils import rectify_catalog
from forcepho.patches import JadesPatch


__all__ = ["get_superscene", "get_patcher",
           "adjust_bounds",
           "set_band_backgrounds",
           "optimize_linear", "dump_design_matrices",
           "untouched_scene",
           "dilate_sersic",
           "max_roi", "get_roi"]


def get_superscene(config, logger, **rectify_kwargs):

    # --- Patch Dispatcher ---  (parent)
    logger.info(f"reading catalog from {config.raw_catalog}")
    cat, bands, chdr = rectify_catalog(config.raw_catalog, **rectify_kwargs)
    bands = [b for b in bands if b in config.bandlist]
    try:
        roi = cat["roi"]
    except(IndexError):
        logger.info("using twice rhalf for ROI")
        roi = 2 * cat["rhalf"]

    sceneDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                               maxactive_per_patch=config.maxactive_per_patch,
                               maxradius=config.patch_maxradius,
                               target_niter=config.sampling_draws,
                               statefile=os.path.join(config.outbase, config.scene_catalog),
                               bounds_kwargs=config.bounds_kwargs,
                               strict=config.strict,
                               roi=roi)
    sceneDB = adjust_bounds(sceneDB, bands, config)
    logger.info("Made SceneDB")
    return sceneDB, bands


def get_patcher(config, logger):
    # --- Patch Maker (child; gets reused between patches) ---
    patcher = JadesPatch(metastore=config.metastorefile,
                         psfstore=config.psfstorefile,
                         pixelstore=config.pixelstorefile,
                         splinedata=config.splinedatafile,
                         return_residual=True)
    if config.tweak_background:
        logger.info("Tweaking image backgrounds")
        patcher = set_band_backgrounds(patcher, config)

    return patcher


def get_bigscene(ra, dec, radius, bigDB):

    cat, bands, chdr = rectify_catalog(config.big_catalog)
    bands = [b for b in bands if b in config.bandlist]
    roi = r_iso
    bigDB = LinkedSuperScene(sourcecat=cat, bands=bands,
                             maxactive_per_patch=50,
                             maxradius=100,
                             roi=roi)

    return bigDB

    center = bigDB.sky_to_scene(ra, dec)
    overlaps = bigDB.find_overlaps(center, radius)
    return bigDB.sourcecat[overlaps]


def dilate_sersic(in_file, out_file, dilation=10.0):
    """Dumb code to expand the seric Gaussian mixtures to approximate larger radii
    """
    paths = ["amplitudes", "nsersic", "radii", "rh"]
    with h5py.File(out_file, "w") as fdest:
        with h5py.File(in_file, "r") as fsrc:
            for path in paths:
                fsrc.copy(fsrc[path], fdest, path)
        fdest["radii"][...] = fdest["radii"][:] * dilation
        fdest["rh"][...] = fdest["rh"][:] * dilation


def set_band_backgrounds(patcher, config):
    """Add keywords to the header data that specify the residual background to
    be subtracting when buiolding a patch.
    """
    tweak_field = getattr(config, "tweak_background")
    if tweak_field == "":
        print("No background tweak field name supplied")
        return patcher

    bgs = getattr(config, tweak_field)
    for band in patcher.metastore.headers.keys():
        if band not in bgs:
            continue
        for expID in patcher.metastore.headers[band].keys():
            patcher.metastore.headers[band][expID][tweak_field] = bgs[band]

    return patcher


def untouched_scene(active, fixed):
    new_inds = active["n_iter"] == 0
    old = active[~new_inds]
    if fixed is not None:
        old = np.concatenate([fixed, old])
    if len(old) == 0:
        old = None
    new = active[new_inds]
    return new, old, new_inds


def adjust_bounds(sceneDB, bands, config):
    # --- Adjust initial bounds ---
    if config.minflux is not None:
        # set lower bound for the flux that is <= minflux
        for b in bands:
            lower = sceneDB.bounds_catalog[b][:, 0]
            sceneDB.bounds_catalog[b][:, 0] = np.minimum(lower, config.minflux)
    if config.maxfluxfactor > 0:
        for b in bands:
            upper = sceneDB.bounds_catalog[b][:, 1]
            new_upper = np.maximum(upper, sceneDB.sourcecat[b] * config.maxfluxfactor)
            sceneDB.bounds_catalog[b][:, 1] = new_upper
    return sceneDB


def optimize_linear(patcher, active, bounds, fixed=None,
                    factor=3.0,
                    shape_cols=[], return_all=True, rank=1):
    """Find the ML flux values for any sources with no HMC sampling yet. Update
    the starting flux and the bounds to be appropriate for these ML fluxes. Note
    this changes the bounds, and so makes any previous covariance matrix
    estimates in the unconstrained space invalid.  It should therefore only be
    used *before* any sampling for a particular source, or the covariance matrix
    should be reset to the identity.
    """

    logger = logging.getLogger(f'dispatcher-child-{rank}')

    new, old, new_inds = untouched_scene(active, fixed)
    if len(new) == 0:
        return active, bounds
    out = optimize_fluxes(patcher, new, old, shape_cols=shape_cols,
                          return_all=return_all)
    fluxes = out[0]

    # now update the initial fluxes for the active sources,
    # and update the bounds
    new_bounds = bounds.copy()
    for i, b in enumerate(patcher.bandlist):
        f = np.atleast_1d(fluxes[i])
        if np.any(~np.isfinite(f)):
            logger.info(f"Invalid flux in band {b} with fluxes {f}")
            logger.info(f"Current active IDs {new[new_inds]['id']}")
            raise ValueError
            #continue

        lo, hi = flux_bounds(f, factor)
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(lo)):
            logger.info(f"could not create valid bounds for band {b} with fluxes {f}")
            continue

        if np.any(lo >= f):
            logger.info(f"low bound > flux for band {b} with\nfluxes: {f}\nlower: {lo}")
            continue

        if np.any(hi <= f):
            logger.info(f"upper bound <= flux for band {b} with\nfluxes: {f}\nupper: {hi}")
            continue

        active[b][new_inds] = f
        new_bounds[b][new_inds, 0] = lo
        new_bounds[b][new_inds, 1] = hi

    return active, new_bounds


def dump_design_matrices(fn, patcher, active, fixed, shape_cols=None):
    new, old, _ = untouched_scene(active, fixed)
    Xes, fixedX = design_matrix(patcher, new, fixed=old, shape_cols=shape_cols)
    ws = patcher.split_band("ierr")
    ys = patcher.split_band("data")
    xpix = patcher.split_band("xpix")
    ypix = patcher.split_band("ypix")

    pixel_lists = dict(X=Xes, fixedX=fixedX, w=ws, y=ys,
                       xpix=xpix, ypix=ypix)

    with h5py.File(fn, "a") as h5:
        h5.create_dataset("active", data=active)
        h5.create_dataset("fixed", data=fixed)
        for i, b in enumerate(patcher.bandlist):
            g = h5.create_group(b)
            for k, v in pixel_lists.items():
                if v is not None:
                    g.create_dataset(k, data=v[i])


def max_roi(raw, bands, threshold=0.1, pixel_scale=0.06, flux_radius=None):

    iso = threshold / pixel_scale**2
    roi = raw["rhalf"]
    for band in bands:
        this = get_roi(raw, isophote=(band, iso), flux_radius=flux_radius)
        roi = np.maximum(roi, this)

    # add a pixel
    roi += pixel_scale

    return roi


def get_roi(raw, isophote=("F160W", 0.1/0.06**2), flux_radius=None):

    band, iso = isophote
    roi = isophotal_radius(iso, raw[band], raw["rhalf"], sersic=raw["sersic"], flux_radius=flux_radius)
    # negative fluxes
    roi[np.isnan(roi)] = raw["rhalf"][np.isnan(roi)]
    # if surface brightness at Rhalf < target, use Rhalf
    roi = np.maximum(roi, raw["rhalf"])
    # account for elongation
    # not that in the prepped catalogs "q" = sqrt(b/a)
    roi /= raw["q"]

    return roi


def in_isophotes(cat, bright, bands, threshold=0.1, pixel_scale=0.06):
    wcs = dummy_wcs(cat)
    iso = max_roi(bright, bands, threshold=threshold, pixel_scale=pixel_scale)
    regions = make_regions(bright, roi=iso, ellipse=True)
    inbright = np.zeros(len(cat), dtype=bool)
    coords = SkyCoord(ra=cat["ra"], dec=cat["dec"], unit="deg", frame="fk5")
    for r in regions:
        inbright |= r.contains(coords, wcs)

    return inbright


def dummy_wcs(cat):
    from astropy.wcs import WCS
    wcs = WCS(naxis=2)

    # Set up an TAN projection
    # Vector properties may be set with Python lists, or Numpy arrays
    args = dict(naxis=2,
                crpix1=0,
                crpix2=0,
                cdelt1=-0.002777778,
                cdelt2=0.00277778,
                crval1=cat[0]["ra"],
                crval2=cat[0]["dec"],
                ctype1="RA---TAN",
                ctype2="DEC--TAN")
    wcs = WCS(args)
    return wcs


def make_regions(cat, roi=None, ellipse=False):
    from astropy import units as u
    from regions import EllipseSkyRegion, Regions

    if roi is None:
        roi = cat["rhalf"]

    regs = []
    for i, row in enumerate(cat):
        center_sky = SkyCoord(row["ra"], row["dec"], unit='deg', frame='fk5')

        if ellipse:
            sqrtq = row["q"]
            pa = np.rad2deg(-row["pa"])
            #pa = np.rad2deg(row["pa"])
        else:
            sqrtq = 1
            pa = 0.0
        a = roi[i] / sqrtq
        b = sqrtq * roi[i]
        reg = EllipseSkyRegion(center=center_sky, height=b * u.arcsec,
                               width=a * u.arcsec, angle=pa * u.deg)
        regs.append(reg)

    return Regions(regs)


def cat_to_reg(cat, slist="", showid=False, default_color="green",
               valid=None, roi=None, ellipse=False):

    if type(cat) is str:
        from astropy.io import fits
        cat = fits.getdata(cat)

    regions = make_regions(cat, roi=roi, ellipse=ellipse)

    if valid is not None:
        for i, r in enumerate(regions):
            if valid[i]:
                r.visual["color"] = "green"
            else:
                r.visual["color"] = "red"

    if showid:
        for i, r in enumerate(regions):
            r.meta["text"] = f"{cat[i]['id']}"

    if slist:
        regions.write(slist, format="ds9", overwrite=True)

    return regions
