#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys, glob, shutil
import argparse, logging
import numpy as np
import json, yaml

import matplotlib.pyplot as pl
from astropy.io import fits


from multi_slope_fit import make_tag
from test_plot import plot_trace, plot_corner, plot_residual
from test_plot import make_catalog, compare_parameters, compare_apflux, allcorner

try:
    import pycuda
    import pycuda.autoinit
    HASGPU = True
except:
    print("NO PYCUDA")
    HASGPU = False


def make_all_tags(subcat, config):
    tags = []
    for row in subcat:

        # make directories and names
        config.id = row["id"]
        config.tag = make_tag(config)
        config.outdir = os.path.join(config.dir, config.tag)
        config.outroot = os.path.join(config.outdir, config.tag)
        tags.append(config.outroot)

    return tags


if __name__ == "__main__":

    print(f"HASGPU={HASGPU}")

    # ------------------
    # --- Configure ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./output/gpu/F200W+F277W/")
    parser.add_argument("--bands", type=str, nargs="*", default=["F200W", "F277W"])
    config = parser.parse_args()
    tcat_name = os.path.join(config.dir, "initial_image_catalog*.fits")
    #config.banddir = f"{config.dir}/{config.bands[0].lower()}"
    thisband = config.bands[0]

    # Make summary plots
    tcat = np.concatenate([fits.getdata(cn) for cn in glob.glob(tcat_name)])
    tags = make_all_tags(tcat, config)
    scat = make_catalog(tags, bands=config.bands)

    color_name = f"[{config.bands[0]}] - [{config.bands[1]}]"

    tcolor = -2.5 * np.log10(tcat[config.bands[0]] / tcat[config.bands[1]])
    scolor = -2.5 * np.log10(scat[config.bands[0]] / scat[config.bands[1]])
    pcolor = np.percentile(scolor, [16, 50, 84], axis=-1)
    ecolor = np.diff(pcolor, axis=0)
    flux = tcat[config.bands[0]]
    dcolor = pcolor[1] - tcolor

    pl.ion()
    fig, axes = pl.subplots(2, 1, gridspec_kw=dict(hspace=0.25))
    ax = axes[0]
    cb = ax.scatter(tcolor, pcolor[1], c=np.log10(flux))
    ax.errorbar(tcolor, pcolor[1], yerr=ecolor, color="grey", alpha=0.7, linestyle="")
    x = np.linspace(np.nanmin(pcolor), np.nanmax(pcolor), 100)
    ax.plot(x, x, linestyle="--", color="k")
    ax.plot(x, x-0.1, linestyle=":", color="k")
    ax.plot(x, x+0.1, linestyle=":", color="k")
    ax.set_ylabel(f"{color_name} (forcepho)")
    ax.set_xlabel(f"{color_name} (Input)")
    ax.set_xlim(tcolor.min()-0.1, min(tcolor.max()+0.1, 4))

    ax = axes[1]
    ax.scatter(flux, dcolor, c=np.log10(flux))
    ax.errorbar(flux, dcolor, yerr=ecolor, color="grey", alpha=0.7, linestyle="")
    ax.axhline(0.0, color="k", linestyle="--")
    ax.axhline(-0.1, color="k", linestyle=":")
    ax.axhline(0.1, color="k", linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel(f"{config.bands[0]} (Input)")
    ax.set_ylabel(f"forcepho-Input color")
    ax.set_ylim(-1, 1)

    fig.colorbar(cb, label=f"log({config.bands[0]})", ax=axes)
    fig.savefig(os.path.join(config.dir, f"{'+'.join(config.bands)}_color_comparison.png"), dpi=200)
    pl.close(fig)

    comp = [("rhalf", config.bands[0]), ("sersic", thisband), ("q", config.bands[0])]
    for show, by in comp:
        fig, axes = compare_parameters(scat, tcat, show, colorby=by, splitby=None)
        fig.savefig(os.path.join(config.dir, f"{'+'.join(config.bands)}_{show}_comparison.png"))
        pl.close(fig)

    fig, axes = compare_apflux(scat, tcat, band=config.bands, colorby="rhalf", xpar=config.bands[0])
    axes.set_ylim(0.1, 1.5)
    axes.set_xlim(1, 1e3)
    fig.savefig(os.path.join(config.dir, f"{'+'.join(config.bands)}_flux_comparison.png"), dpi=200)
    pl.close(fig)
