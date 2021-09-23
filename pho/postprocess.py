#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np

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


if __name__ == "__main__":

    modes = ["images", "patches", "catalog", "chains", "postop"]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="images",
                        choices=modes)
    parser.add_argument("--root", type=str, default="../output/opt_all_v0.5_linopt_debug_v4")
    parser.add_argument("--metafile", type=str, default="../data/stores/meta_hlf2_udf.json")
    parser.add_argument("--exp", type=int, default=14)
    parser.add_argument("--catname", type=str, default=None)
    args = parser.parse_args()

    #write_sourcereg(args.root, slist="sources.reg", showid=True)
    #write_patchreg(args.root, plist="patches.reg")

    if args.mode == 'images':
        fpost.write_images(args.root, metafile=args.metafile, show_model=True)
        fpost.write_patchreg(args.root)
        fpost.write_sourcereg(args.root, showid=True)

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
