#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import h5py
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

from forcepho.patches.storage import MetaStore
from forcepho.utils import read_config, sourcecat_dtype
from forcepho.superscene import LinkedSuperScene, isophotal_radius

try:
    from starmask import mask_psf, mask_radius, get_cat_star
except(ImportError):
    pass
from utils import max_roi, cat_to_reg, make_regions, in_isophotes


def mask_stars(cat, band="f105w", cam="wfc3-60", threshold_njy=0.1,
               HDR_STORE="../data/images/hlf2_full_headerDB.json"):
    """Mask sources in pixels above some threshold stellar flux based on the
    PSF. Uses pre-computed star x, y catalogs generated by K Whitaker from 60mas
    pixel images and including a magnitude measurement.

    Parameters
    ----------
    cat : structured ndarray
        Catalog of sources with the fields 'ra' and 'dec'

    Returns
    -------
    mask : boolean array
        True for sources that do not fall in a pixel with stellar flux above
        threshold.
    """

    galaxies = SkyCoord(cat["ra"] * u.degree, cat["dec"] * u.degree)

    psf = fits.getdata(f"../data/psfs/{band}_psf.fits")
    hstore = MetaStore(HDR_STORE)
    ref_image = f'hlsp_hlf_hst_{cam}mas_goodss_{band}_v2.0_sci'
    wcs = hstore.wcs[band.upper()][ref_image]

    stars, scat = get_cat_star(band)
    #good = scat["flag"] == "O"
    good = slice(None)
    stars, scat = stars[good], scat[good]

    sflux = 3631e9 * 10**(-0.4 * scat["mag"])
    threshold = np.array(threshold_njy / sflux)
    pmask = mask_psf(galaxies, stars, psf, wcs=wcs, threshold=threshold)
    return pmask == 0


def distance(ra, dec, mra, mdec):

    center = SkyCoord(mra, mdec, unit="deg")
    scene_frame = center.skyoffset_frame()
    c = SkyCoord(ra, dec, unit="deg")
    xy = c.transform_to(scene_frame)
    x, y = xy.lon.arcsec, xy.lat.arcsec
    d = np.hypot(x, y)
    return d


def mask_big(cat, roi, bands, ng_max=14, r_max=2.0):
    cat["roi"] = roi
    sceneDB = LinkedSuperScene(bands=bands, roi=roi)
    sceneDB.ingest(cat)
    gid = sceneDB.make_group_catalog()
    groups, ng = np.unique(gid, return_counts=True)
    inmulti = np.isin(gid, groups[ng > ng_max])
    big = roi > r_max
    bigg = np.unique(gid[big])
    inbig = np.isin(gid, bigg)
    valid = (~inmulti) & (~inbig)
    return valid

def jades_pipe_to():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input
    parser.add_argument("--config_file", type=str, default="./morph_mosaic_config.yml")
    parser.add_argument("--detection_catalog", type=str, default="../data/catalogs/hd_all.cat.fits")
    parser.add_argument("--output_catalog", type=str, default="initial_catalog.fits")
    parser.add_argument("--flux_type", type=str, default="KRON",
                        choices=["KRON", "CIRC1", "CIRC2", "CIRC3", "CIRC4"])
    # Basic selection
    parser.add_argument("--snr_threshold", type=float, default=0)
    parser.add_argument("--center", type=float, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--n_source", type=int, default=0)
    # Advanced Masking
    parser.add_argument("--bright_catalog", type=str, default=None)
    parser.add_argument("--mask_stars", type=int, default=0)
    parser.add_argument("--mask_big", type=int, default=0,
                        help="Whether to mask large objects and objects part of large groups")
    # Other
    parser.add_argument("--pixel_scale", type=float, default=0.03)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Flux per pixel defining the isophote (catalog units, usually nJy)")

    args = parser.parse_args()
    config = read_config(args.config_file, args)

    bands = (config.bandlist).tolist()
    det = fits.getdata(config.detection_catalog)
    dhdr = fits.getheader(config.detection_catalog)
    try:
        config.pixel_scale = dhdr["DPIXSCAL"]
    except(KeyError):
        pass


    # --- Create catalog and fill columns ---
    # ---------------------
    cat_dtype = sourcecat_dtype(bands=bands)
    cat = np.zeros(len(det), dtype=cat_dtype)
    cat["id"] = np.arange(len(cat))
    cat["ra"] = det["RA"]
    cat["dec"] = det["DEC"]
    cat["q"] = np.sqrt(det["B"] / det["A"])
    cat["pa"] = det["PA"]
    cat["sersic"] = 2.0
    cat["rhalf"] = det["A"] * config.pixel_scale
    ind = 0
    snr = {}

    # rotate by +90 degrees, but keep in -90, +90 interval
    #p = cat["pa"] > 0
    #cat["pa"] += np.pi / 2. - p * np.pi
    # reverse: SEP measures (-pi/2, pi/2) CCW from +x, fpho needs (-pi/2, pi/2) North of East
    cat["pa"] *= -1

    if config.flux_type == "KRON":
        ftype = "KRON"
        config.aperture_radius = det["R_KRON"] * config.pixel_scale
    else:
        ftype = config.flux_type
        config.aperture_radius = dhdr[f"F200WRC{ftype[-1]}"]

    sel = np.ones(len(cat), dtype=bool)

    for b in bands:
        conv = 1.0  # fluxes already in nJy
        fcol = f"{b.upper()}_{ftype}"
        cat[b] = det[fcol] * conv
        snr[b] = det[fcol] / det[f"{fcol}_e"]
        sel = sel & np.isfinite(snr[b])

    # --- Get ROI ---
    # ---------------
    config.sersic = 1
    roi = max_roi(cat, bands, threshold=config.threshold,
                  pixel_scale=config.pixel_scale, flux_radius=config.aperture_radius)

    # --- Select on S/N ---
    # ---------------------
    if config.snr_threshold:
        snr_array = np.array([snr[b] > config.snr_threshold for b in bands])
        sel = sel & np.any(snr_array, axis=0)

    # --- Select on location ---
    # --------------------------
    dmax = np.inf
    if config.n_source > 0:
        if config.seed <= 0:
            default = [np.median(cat["ra"][sel]), np.median(cat["dec"][sel])]
            center = getattr(config, "center", default)
        else:
            center = [cat["ra"][config.seed], cat["dec"][config.seed]]
        dist = distance(cat["ra"], cat["dec"], center[0], center[1])
        oo = np.argsort(dist[sel])
        dmax = dist[sel][oo][config.n_source]
        sel = sel & (dist <= dmax)
    print(f"Selected {sel.sum()} sources within {dmax:0.2f} arcsec")

    cat = cat[sel]
    roi = roi[sel]
    valid = np.ones(len(cat), dtype=bool)

    # --- Mask stars, bright, and big ---
    # --------------------------
    if config.mask_stars:
        valid = mask_stars(cat)
        print(f"removed {(~valid).sum()} stars")

    if config.mask_big:
        notbig = mask_big(cat, roi, bands)
        valid = valid & notbig
        print(f"removed {(~notbig).sum()} big sources")

    if getattr(config, "bright_catalog", None):
        # --- HACK ---
        # These are things affected by the bright objects
        big = roi > 2.0
        nroi = 0.4 * roi[big]
        roi[big] = np.clip(nroi, 0, 0.1)
        # ------------
        bright = fits.getdata(config.bright_catalog)
        inbright = in_isophotes(cat, bright, bands, threshold=config.threshold,
                                pixel_scale=config.pixel_scale, )
        valid = valid & ~inbright
        print(f"removed {(inbright).sum()} sources within the bright objects")

        bcat = np.zeros(len(bright), dtype=cat_dtype)
        for c in bcat.dtype.names:
            try:
                bcat[c] = bright[c]
            except:
                print(f"didn't copy {c} column from bright catalog")

        bcat["id"] = -np.arange(len(bcat))

        if config.n_source > 0:
            bdist = distance(bcat["ra"], bcat["dec"], center[0], center[1])
            dsel = (bdist <= dmax)
            bcat = bcat[dsel]

        broi = max_roi(bcat, bands, threshold=config.threshold,
                       pixel_scale=config.pixel_scale)
        valid = np.concatenate([valid, np.ones(len(bcat), dtype=bool)])
        cat = np.concatenate([cat, bcat])
        roi = np.concatenate([roi, broi])

    cat["roi"] = roi
    regname = args.output_catalog.replace(".fits", "_roi.reg")
    regions = cat_to_reg(cat, slist=regname, roi=roi, valid=valid, ellipse=True, showid=True)

    ocat = cat[valid]

    table = fits.BinTableHDU.from_columns(ocat)
    hdr = fits.Header()
    hdr["FILTERS"] = ",".join(bands)
    hdr["DET_CAT"] = config.detection_catalog
    hdr["STARMSK"] = config.mask_stars
    hdr["BIGMSK"] = config.mask_big
    hdr["SEEDID"] = config.seed
    full = fits.HDUList([fits.PrimaryHDU(header=hdr), table])
    full.writeto(config.output_catalog, overwrite=True)
