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

if __name__ == "__main__":

    pl.ion()
    root = "optimization_v2.1"
    bands = ["F200W"]

    out = fits.getdata(f"{root}/outscene.fits")
    inc = fits.getdata("../data/catalogs/truth_initial_catalog.fits")
    mag = -2.5*np.log10(inc[bands[0]]/3631e9)

    sel = ((inc["sersic"] > 1.0) & (inc["sersic"] < 5) &
           (inc["rhalf"] < 0.25) & (inc["q"] > 0.4))

    fields = bands + ["rhalf", "sersic", "q", "pa"]
    fig, axes = pl.subplots(2, 3, figsize=(12, 7))

    for i, f in enumerate(fields):
        ax = axes.flat[i]
        if "F" in f:
            unit = units["flux"]
        else:
            unit = units[f]
        if ("F" in f) or (f == "rhalf"):
            ax.set_xscale("log")
            ax.set_yscale("log")

        x = inc[f]
        y = out[f]
        if f =="q":
            x = x**2
            y = y**2
        cb = ax.scatter(x[sel], y[sel], c=mag[sel], marker="o", alpha=0.8)
        xx = np.linspace(x.min(), x.max())
        ax.plot(xx, xx, "k:")
        ax.set_xlabel(f"{f} True ({unit})")
        ax.set_ylabel(f"{f} Forcepho ({unit})")
        ax.set_xlim(x[sel].min(), x[sel].max())
        ax.set_ylim(x[sel].min(), x[sel].max())

    fig.colorbar(cb, label=fr"$m_{{{bands[0]}}}$")
    fig.suptitle(f"Single band optimization: {bands[0]}")
    axes.flat[-1].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{root}_comparison.png", dpi=300)
