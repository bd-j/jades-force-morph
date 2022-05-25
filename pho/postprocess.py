#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import FormatStrFormatter

from astropy.io import fits

from forcepho.utils import read_config
from forcepho.postprocess import Residuals, Samples
from forcepho import postprocess as fpost
from forcepho.utils import frac_sersic


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
            args.metafile = config.get("metastorefile", None)

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
