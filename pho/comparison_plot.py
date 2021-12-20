#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as pl

from astropy.io import fits

units = dict(q="b/a",
             pa="radians N of E",
             rhalf="arcsec",
             sersic="",
             flux="nJy")


def compare(x, ys, field, ax, c=None, **scatter_kwargs):

    y = ys[f]

    if "F" in field:
        unit = units["flux"]
    else:
        unit = units[field]
    if ("F" in field) or (field == "rhalf"):
        ax.set_xscale("log")
        ax.set_yscale("log")
    if f == "q":
        x = x**2
        y = y**2

    try:
        lnp = ys["lnp"]
    except:
        lnp = None

    ymed, yerr, ymax = rectify(y, lnp=lnp)

    cb = ax.scatter(x, ymax, c=c, **scatter_kwargs)
    if yerr is not None:
        ax.errorbar(x, ymed, yerr=yerr, linestyle="", color="gray")
    xx = np.linspace(x.min(), x.max())
    ax.plot(xx, xx, "k:")
    ax.set_xlabel(f"{f} True ({unit})")
    ax.set_ylabel(f"{f} Forcepho ({unit})")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(x.min(), x.max())

    return ax, cb


def rectify(y, lnp=None):
    if y.ndim > 1:
        yp = np.percentile(y, [16, 50, 84], axis=-1)
        ymed = yp[1]
        delta = np.diff(yp, axis=0)
        if lnp is not None:
            ind_ml = np.argmax(lnp, axis=-1)
            ymax = y[np.arange(len(y)), ind_ml]
        else:
            ymax = None
    else:
        ymed = y
        delta = None
        ymax = y

    return ymed, delta, ymax


if __name__ == "__main__":

    pl.ion()

    root = "../output/optimization_v2.1"
    bands, pix_scale, sigma_pix, module = ["F200W"], 0.03, 0.06, "sw"
    fields = bands + ["rhalf", "sersic", "q", "pa"]
    snr_thresh = 30

    # --- truth catalog ---
    truths = fits.getdata("../data/catalogs/truth_initial_catalog.fits")
    valid = fits.getdata("../data/catalogs/truth_in_image.fits")
    mag = -2.5 * np.log10(truths[bands[0]] / 3631e9)
    snr = truths[bands[0]] / 2 / (np.sqrt(np.pi) * truths["rhalf"] / pix_scale * sigma_pix)
    # restrict to forcepho priors
    sel = ((truths["sersic"] > 1.0) & (truths["sersic"] < 5) &
           (truths["rhalf"] < 0.25) & (truths["q"] > 0.4) &
           (valid[f"in_{module.lower()}"] > 0) & (snr > snr_thresh))

    # -- thing to compare to truth ---
    #out = fits.getdata(f"{root}/outscene.fits")
    result = "../output/sampling_v2.1.3/full_chaincat.fits"
    out = fits.getdata(result)
    #out = fits.getdata("../data/catalogs/postop_v2.1_catalog.fits")
    #unc = fits.getdata("../data/catalogs/postop_v2.1_catalog.fits", 1)
    sel = sel & (out["id"] > 0)
    assert np.all(out[sel]["id"] == truths[sel]["id"])

    fig, axes = pl.subplots(2, 3, figsize=(12, 7))
    for i, f in enumerate(fields):
        ax = axes.flat[i]
        x = truths[f]
        ax, cb = compare(x[sel], out[sel], f, ax,
                         c=np.log10(snr[sel]), marker="o", alpha=0.8)

    fig.colorbar(cb, label=fr"$\log(SNR_{{{bands[0]}}})$")
    fig.suptitle(f"Single band optimization: {bands[0]}")
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{result.replace('.fits', '')}_comparison.png", dpi=300)
