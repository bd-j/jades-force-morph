#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits


def initial_convert():
    with open("truth_table.txt", "r") as f:
        hdr = f.readline()[1:].split()

    dtype = np.dtype([(c, int) for c in hdr[:2]] + [(c, float) for c in hdr[2:]])
    cat = np.genfromtxt("truth_table.txt", skip_header=1, dtype=dtype)
    fits.writeto("truth_table.fits", cat)


def to_forcepho_format():
    from forcepho.superscene import sourcecat_dtype

    tcat = fits.getdata("truth_table.fits")
    bands = [b.replace("NIRCAM_", "") for b in tcat.dtype.names if "NIRCAM" in b]

    dtype = sourcecat_dtype(bands=bands)
    cat = np.zeros(len(tcat), dtype=dtype)

    cat["id"] = tcat["JAGUAR_ID"]
    cat["ra"] = tcat["ra"]
    cat["dec"] = tcat["dec"]
    cat["sersic"] = tcat["sersic_n"]
    cat["q"] = np.sqrt(tcat["semi_b"] / tcat["semi_a"])
    cat["rhalf"] = tcat["semi_a"] * cat["q"]
    cat["roi"] = 5 * cat["rhalf"]

    # add 90 degrees, reverse and to radians
    pa = np.deg2rad(-(tcat["pa"] + 90))
    # restrict to +/- pi/2
    pa[pa < (-np.pi * 3 / 2)] += 2 * np.pi
    pa[pa < (-np.pi / 2)] += np.pi
    cat["pa"] = pa

    # convert to nJy
    for b in bands:
        mag = np.clip(tcat[f"NIRCAM_{b}"], 0, 40)
        cat[b] = 10**(-0.4 * mag) * 3631e9

    hdulist = fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(cat)])
    for h in hdulist:
        h.header["FILTERS"] = ",".join(bands)
    hdulist.writeto("truth_initial_catalog.fits", overwrite=True)


if __name__ == "__main__":

    to_forcepho_format()
    icat = fits.getdata("truth_initial_catalog.fits")
    ihdr = fits.getheader("truth_initial_catalog.fits")