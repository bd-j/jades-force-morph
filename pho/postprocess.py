#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np

from astropy.io import fits

from forcepho.utils import read_config
from forcepho.postprocess import Residuals, Samples
from forcepho import postprocess as fpost


def find_multipatch(root):
    config, plog, slog, final = fpost.run_metadata(root)
    samples = [Samples(f"{root}/patches/patch{p}_samples.h5") for p in plog]
    chains, sources, patches = [], [], []
    for sid, pids in slog.items():
        if len(pids) == 1:
            continue
        c = []
        for pid in pids:
            ai = samples[int(pid)].active["source_index"].tolist().index(int(sid))
            c.append(samples[int(pid)].chaincat[ai:ai+1])
        chains.append(np.concatenate(c))
        sources.append(sid)
        patches.append(pids)

    return sources, patches, chains


def check_multipatch(root, n_sample=256):
    sid, pids, chains = find_multipatch(root)
    stats = {}
    for s, p, c in zip(sid, pids, chains):
        stats[s] = {}
        for col in vcols:
            sm = c[col][:, -n_sample:]
            stats[s][col] = sm.mean(axis=-1), sm.std(axis=-1)
    return stats


def make_errorbars(samplecat, percentiles=[16, 50, 84]):
    """Make percentile based assymetric errorbars.

    Example shows how to plot asymmetric 1-sigma-ish errorbars:

    >>> scat = "path/to/postsample/catalog.fits"
    >>> ecat, hdr = make_errorbars(scat, percentiles=[16, 50, 84])
    >>> ecols = hdr["SMPLCOLS"].split(",")
    >>> colname = "rhalf"
    >>> y = ecat[colname][:,1]
    >>> e = np.diff(ecat[colname], axis=-1).T
    >>> ax.errorbar(y, y, e, linestyle="", marker="o", ecolor="k")

    Parameters
    ----------
    samplecat : string or structured ndarray
        Name of fits file contining result of
        :py:func:`forcepho.postprocess.postsample_catalog()`

    percentiles : list of floats in the interval (0, 100)
        The percentiles to compute

    Returns
    -------
    errorcat : structured ndarray
        Catalog of percentile values for each parameter.  These are given
        in the same order as the list in the `percentiles` keyword.

    hdr : dictionary or FITSHeader
        information about the errorbars.
    """
    if type(samplecat) is str:
        cat = np.array(fits.getdata(samplecat))
        hdr = fits.getheader(samplecat)
        hdr["PCTS"] = ",".join([str(p) for p in percentiles])
        bands = hdr["FILTERS"].split(",")
    else:
        cat = samplecat
        hdr = dict(PCTS=",".join([str(p) for p in percentiles]))

    desc, scol = [], []
    for d in cat.dtype.descr:
        if len(d) == 3:
            scol.append(d[0])
            desc.append((d[0], d[1], len(percentiles)))
        else:
            desc.append(d)
    ecat = np.zeros(len(cat), dtype=np.dtype(desc))
    for col in ecat.dtype.names:
        if col in scol:
            pct = np.percentile(cat[col], q=percentiles, axis=-1)
            ecat[col] = pct.T
        else:
            ecat[col] = cat[col]

    hdr["SMPLCOLS"] = ",".join(scol)

    return ecat, hdr


if __name__ == "__main__":

    modes = ["images", "patches", "catalog", "chains", "postop"]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="images",
                        choices=modes)
    parser.add_argument("--root", type=str, default="../output/opt_all_v0.5_linopt_debug_v4")
    parser.add_argument("--metafile", type=str, default="")
    parser.add_argument("--exp", type=int, default=14)
    parser.add_argument("--catname", type=str, default=None)
    args = parser.parse_args()

    #write_sourcereg(args.root, slist="sources.reg", showid=True)
    #write_patchreg(args.root, plist="patches.reg")

    if args.mode == 'images':
        if not args.metafile:
            config, _, _, _ = fpost.run_metadata(args.root)
            args.metafile = config["metastorefile"]

        fpost.write_images(args.root, metafile=args.metafile, show_model=True)
        fpost.write_patchreg(args.root)
        fpost.write_sourcereg(args.root, showid=True, isophote=("F200W", 0.1/0.06**2))

    elif args.mode == 'patches':
        fpost.residual_pdf(root=args.root, e=args.exp)

    elif args.mode == "chains":
        patches = glob.glob(f"{args.root}/patches/*samples*h5")
        for p in patches:
            fpost.chain_pdf(Samples(p), p.replace("samples.h5", "chain.pdf"))

    elif args.mode == "postop":
        print(f"writing to {args.catname}")
        fpost.postop_catalog(args.root, catname=args.catname)

    elif args.mode == "catalog":
        print(f"writing to {args.catname}")
        cat = fpost.postsample_catalog(args.root, catname=args.catname)

    else:
        print(f"{args.mode} not a valid mode.  choose one of {modes}")
