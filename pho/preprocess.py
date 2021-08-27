#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, sys
import time

import numpy as np
import argparse

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.nddata import Cutout2D

from forcepho.utils import read_config
from forcepho.patches.storage import ImageNameSet, ImageSet, header_to_id
from forcepho.patches.storage import PixelStore, MetaStore


def prep_cutouts(original_names, cutID="jades-morph", path_out=None,
                 cutout_kwargs={}):
    for n in original_names:
        im, hdr = fits.getdata(n), fits.getheader(n)
        cutout, tiles, nt = make_cutouts(im, hdr, **cutout_kwargs)

        # construct the filenames
        path_in = os.path.dirname(n)
        name_in = os.path.basename(n)
        if path_out is None:
            path_out = path_in
        name_out = "{}/{}_{}".format(path_out, cutID, name_in)
        write_cutout(name_out, cutout, hdr)

        for x in range(nt[0]):
            for y in range(nt[1]):
                i = x * nt[1] + y
                name_out = "{}/{}-{:02.0f}-{:02.0f}_{}".format(path_out, cutID, x, y, name_in)
                write_cutout(name_out, tiles[i], hdr)


def make_cutouts(im, hdr, ra=53.162958332, dec=-27.7901389, sidearcs=300,
                 big_pixel_scales=None):
    """
    Parameters
    ----------
    sidearcs : float or ndarray of floats
        The length of the sides of the rectangle of interest in arcseconds

    big_pixel_scales : optional
        If given, ndarray of pixel scales used to define the basic tiling scheme.
        The actual tiles will be based on an integer multiple of this basic tiling scheme,
        where the integer is given by the ratio of the actual pixel scales to these pixel scales
    """
    # TODO: may need to invert shape somewhere if images aren't square
    wcs = WCS(hdr)
    pixel_scales = 3600 * proj_plane_pixel_scales(wcs)
    if big_pixel_scales is None:
        big_pixel_scales = pixel_scales
    if sidearcs is not None:
        shape = sidearcs / big_pixel_scales
    else:
        # note inverted order
        shape = np.array([hdr["NAXIS2"], hdr["NAXIS1"]])
    ntile = (shape // 2048 + (shape % 2048 > 0).astype(int)).astype(int)
    ntile *= np.round(big_pixel_scales/pixel_scales).astype(int)
    shape = 2048 * ntile

    pos = wcs.all_world2pix(ra, dec, 0)
    pos = (int(pos[0]), int(pos[1]))
    cutout = Cutout2D(im, pos, shape.astype(int), wcs=wcs)

    tiles = []
    for x in range(ntile[0]):
        for y in range(ntile[1]):
            pos = (x*2048 + 1024, y*2048 + 1024)
            tile = Cutout2D(cutout.data, pos, (2048, 2048), wcs=cutout.wcs)
            tiles.append(tile)

    return cutout, tiles, ntile


def write_cutout(name, cutout, hdr=None):
    hdu = fits.PrimaryHDU(cutout.data)
    if hdr:
        hdu.header.update(hdr)
    hdu.header.update(cutout.wcs.to_header())
    hdu.writeto(name)


def find_images(loc="/data/groups/comp-astro/jades/hlf/v2.0/",
                pattern="jades-morph-??-??_*_sci.fits"):
    search = os.path.join(os.path.expandvars(loc), pattern)
    import glob
    files = glob.glob(search)
    names = [ImageNameSet(im=f, err=f.replace(".fits", "_err.fits"), mask=None, bkg=None)
             for f in files]
    return names


def nameset_to_imset(nameset, zeropoints={}, max_snr=None):
    # Read the header and set identifiers
    hdr = fits.getheader(nameset.im)
    band, expID = header_to_id(hdr, nameset.im)
    # Add the zeropoint
    try:
        hdr.update(ABMAG=zeropoints[band])
    except(KeyError):
        pass

    # Read data and perform basic operations
    # NOTE: we transpose to get a more familiar order where the x-axis
    # (NAXIS1) is the first dimension and y is the second dimension.
    im = np.array(fits.getdata(nameset.im)).T
    # err images are units of uncertainty (sigma)
    ierr = 1 / np.array(fits.getdata(nameset.err)).T
    ierr[~np.isfinite(ierr)] = 0.0
    # cap S/N ?
    if max_snr:
        to_cap = (ierr > 0) & (im * ierr > max_snr)
        ierr[to_cap] = max_snr / im[to_cap]

    imset = ImageSet(hdr=hdr, band=band, expID=expID, names=nameset,
                     im=im, ierr=ierr, mask=None, bkg=None)
    return imset


def nameset_to_imset_slopes(nameset, zeropoints={}):
    # Read the header and set identifiers
    hdr = fits.getheader(nameset)
    band, expID = header_to_id(hdr, nameset)
    # Add the zeropoint
    try:
        hdr.update(ABMAG=zeropoints[band])
    except(KeyError):
        pass

    # Read data and perform basic operations
    # NOTE: we transpose to get a more familiar order where the x-axis
    # (NAXIS1) is the first dimension and y is the second dimension.
    im = np.array(fits.getdata(nameset, 1)).T
    ierr = 1 / np.array(fits.getdata(nameset, 2)).T
    mask = fits.getdata(nameset, 4).T
    bkg = fits.getdata(nameset, 3).T
    imset = ImageSet(hdr=hdr, band=band, expID=expID, names=nameset,
                     im=im, ierr=ierr, mask=mask, bkg=bkg)
    return imset


def mosaic_box(lw_imagename):
    hdr = fits.getheader(lw_imagename)
    wcs = WCS(hdr)
    cx, cy = hdr["NAXIS1"] / 2, hdr["NAXIS2"] / 2
    pos = wcs.all_pix2world(cx, cy, origin=0)
    


if __name__ == "__main__":

    t = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="config.yml",
                        help="Location of the configuration YAML file.")
    parser.add_argument("--original_images", type=str, default="../data/images/*final/*[WM].fits",
                        help=("If making cutouts, read the original full size "
                              "images (and weights) from this search pattern."))
    parser.add_argument("--cutID", type=str, default="",
                        help=("If supplied, generate cutout images and 2048x2048 "
                              "pixel tiles with this prefix in the "
                              "`config.frames_directory` location"))
    parser.add_argument("--max_snr", type=float, default=0,
                        help="Force max S/N to this value by altering the noise values")
    parser.add_argument("--snr_thresh", type=float, default=0,
                        help="Adjust S/N when larger than this.")
    parser.add_argument("--stop_at", type=int, default=-1,
                        help="Stop preprocessing at this step.")
    parser.add_argument("--store_directory", type=str, default=None,
                        help="Location for the stores")
    parser.add_argument("--pixelstorefile", type=str, default=None,
                        help="Location for the stores")
    parser.add_argument("--metastorefile", type=str, default=None,
                        help="Location for the stores")
    args = parser.parse_args()

    # -------------------
    # --- read config ---
    config = read_config(args.config_file, args)
    if args.pixelstorefile is None:
        raise ValueError

    if config.stop_at == 1:
        sys.exit()

    # --------------------
    # --- make cutouts ---
    if getattr(config, "cutID", ""):
        os.makedirs(config.frames_directory, exist_ok=True)
        original = glob.glob(config.original_images)
        original += [o.replace(".fits", "_err.fits") for o in original]
        original = np.unique(original)
        cutout_kwargs = dict(big_pixel_scales=np.array([0.06, 0.06]))
        prep_cutouts(original, cutID=config.cutID, path_out=config.frames_directory,
                     cutout_kwargs=cutout_kwargs)

    if config.stop_at == 2:
        sys.exit()

    # -------------------
    # --- Find Images ---
    names = find_images(config.frames_directory,
                        config.frame_search_pattern)

    if config.stop_at == 3:
        sys.exit()

    # ---------------------------
    # -- Make and fill stores ---
    [os.makedirs(os.path.dirname(a), exist_ok=True)
     for a in (config.pixelstorefile, config.metastorefile)]

    # Make the (empty) PixelStore
    pixelstore = PixelStore(config.pixelstorefile,
                            nside_full=config.nside_full,
                            super_pixel_size=config.super_pixel_size,
                            pix_dtype=config.pix_dtype)
    # Make the (empty) metastore
    metastore = MetaStore()

    # Fill pixel and metastores
    for n in names:
        im = nameset_to_imset(n, zeropoints=config.zeropoints, max_snr=config.max_snr)
        pixelstore.add_exposure(im, bitmask=config.bitmask,
                                do_fluxcal=config.do_fluxcal)
        metastore.add_exposure(im)

    # Write the filled metastore
    metastore.write_to_file(config.metastorefile)

    print("done in {}s".format(time.time() - t))