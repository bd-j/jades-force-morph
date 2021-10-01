#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, glob
import numpy as np
from scipy.optimize import curve_fit
from astropy.io import fits


ZP = dict(F225W=24.04,
          F275W=24.13,
          F336W=24.67,
          F435W=25.68,
          F606W=26.51,
          F775W=25.69,
          F814W=25.94,
          F850LP=24.87,
          F098M=25.68,
          F105W=26.27,
          F125W=26.23,
          F140W=26.45,
          F160W=25.94)


def gauss(x, amp, mu, sig):
    return amp * np.exp(-(x-mu)**2/(2.0*sig**2))


def fit_constant_bg(imnames, niter=4, tweakfile="bg_tweaks.dat"):
    """Fit a constant background to given images.  Write values (in units of
nJy/pix) for each image
    """
    if tweakfile:
        out = open(tweakfile, "w")
        out.write("band  bg(e/s/pix) bg(nJy/pix)  unc_bg  sigma_bg\n")
        ofmt = "{}  {:.3e} {:.3e} {:.3e} {:.3e}\n"

    result, valid = [], []
    for n in imnames:
        h = fits.getheader(n)
        b = h["FILTER"]
        try:
            zp = h["ABMAG"]
        except(KeyError):
            zp = ZP.get(b.upper(), None)

        if zp is not None:
            conv = 1e9 * 10**(0.4 * (8.9 - zp))
        else:
            conv = np.nan
        im = fits.getdata(n).flatten()
        g = np.isfinite(im)
        # HACK: what are these - zero to single precision?
        g = g & (np.abs(im) > 1e-6)

        for k in range(niter):
            mu = np.median(im[g])
            sig = np.std(im[g])
            nbin = int(3 * sig / 1e-2)
            bins = np.linspace(mu - 1.5*sig, mu + 1.5*sig, nbin)
            keep = np.abs(im - mu) < 2 * sig
            g = g & keep

        nbin = int(3 * sig / 1e-3)
        bins = np.linspace(mu - 1.5*sig, mu + 1.5*sig, nbin)
        y, x = np.histogram(im[g], bins=bins)
        xx = (x[:-1] + x[1:]) / 2
        err = np.sqrt(y)
        amp = y.max()

        try:
            p_opt, p_cov = curve_fit(gauss, xx, y, sigma=err)
        except(RuntimeError):
            #p_opt, p_cov = np.ones(4) * np.nan, np.eye(2) * np.nan
            continue
        if np.abs(p_opt[1] - mu) > (30 * sig):
            #throw out crazy fits
            continue
        print(b, g.sum())
        print(mu, sig)
        print(p_opt[1], np.sqrt(p_cov[1, 1]), p_opt[2])
        values = b, p_opt[1]/conv, p_opt[1], np.sqrt(p_cov[1, 1]), p_opt[2]
        result.append(values)
        valid.append(n)
        if tweakfile:
            out.write(ofmt.format(*values))
    if tweakfile:
        out.close()
    return result, valid


if __name__ == "__main__":

    #root = "../output/opt_all_v0.5_linopt_nobig_nodupe3"
    #root = "../output/opt_all_v0.5_bgsub"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="constant")
    parser.add_argument("--root", type=str, default="../output/runname")
    parser.add_argument("--search_pattern", type=str, default="*0?-0?_*F*_delta.fits")
    parser.add_argument("--tweakfile", type=str, default="bg_tweaks.dat")
    parser.add_argument("--niter", type=int, default=4)
    args = parser.parse_args()

    # fit constant background to residual images
    if args.mode == "constant":
        imfmt = f"{args.root}/image/{args.search_pattern}"
        imnames = glob.glob(imfmt)
        result, valid = fit_constant_bg(imnames, niter=args.niter,
                                        tweakfile=args.tweakfile)

    bb = np.array([(float(r[0][1:-1]), r[2], r[3]) for r in result])

    import matplotlib.pyplot as pl
    fig, ax = pl.subplots()
    ax.errorbar(bb[:, 0]/100 + np.random.uniform(-0.05,0.05, len(bb)), bb[:,1], bb[:,2], linestyle="", marker=".")
    ax.set_ylim(-1e-2, 0.5e-1)
    ax.set_xlabel("Filter wavelength (micron)")
    ax.set_ylabel("Residual background (nJy/pix)")
