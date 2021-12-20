#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import glob, argparse
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import FormatStrFormatter

from astropy.io import fits

from forcepho.utils import read_config
from forcepho.postprocess import Residuals, Samples
from forcepho import postprocess as fpost
from forcepho.utils import frac_sersic

from prospect.plotting.corner import allcorner, scatter


def aperture_flux(scat, truths):
    band = truths["band"]
    rhalf = truths["rhalf"]
    fr = frac_sersic(rhalf[:, None], sersic=scat["sersic"], rhalf=scat["rhalf"])
    total_flux = np.array([scat[i][b] for i, b in enumerate(band)])
    aperture_flux = total_flux * fr

    return aperture_flux, total_flux


def get_map(s, xx=None):
    lnp = s.stats["model_logp"]
    ind_ml = np.argmax(lnp)
    #row_map = s.get_sample_cat(ind_ml)[0]
    #ymap = np.atleast_2d([row_map[c] for c in s.bands + s.shape_cols])
    if xx is not None:
        ymap = np.atleast_2d(xx[:, ind_ml])
    else:
        ymap = np.atleast_2d(s.chain[ind_ml, :])

    return ymap


def plot_residual(patchname, sid=-1):
    s = Samples(patchname)
    r = Residuals(patchname.replace("samples", "residuals"))
    delta, lo, hi = r.make_exp(value="residual")
    data, _, _ = r.make_exp(value="data")
    ierr, _, _ = r.make_exp(value="ierr")
    chi = (delta * ierr).T

    rfig, raxes = pl.subplots(1, 2, sharex=True, sharey=True)
    rax = raxes[0]
    vmin, vmax = np.nanmin(chi), np.nanmax(chi)
    cb = raxes[0].imshow(chi, extent=(lo[0], hi[0], lo[1], hi[1]),
                         origin="lower", vmin=vmin, vmax=vmax)
    _ = raxes[1].imshow((data * ierr).T, extent=(lo[0], hi[0], lo[1], hi[1]),
                          origin="lower", vmin=vmin, vmax=vmax)
    rfig.colorbar(cb, label=r"$\chi=$ (Data - Model) / Unc", ax=raxes, orientation="horizontal")

    raxes[0].text(0.1, 0.8, "Residual", color="magenta", transform=raxes[0].transAxes)
    raxes[1].text(0.1, 0.8, "Data", color="magenta", transform=raxes[1].transAxes)

    if sid > 0:
        ind = np.where(s.active["source_index"] == sid)[0]
    else:
        ind = 0

    val = s.get_sample_cat(-1)[ind]
    vdict = {c: val[c] for c in val.dtype.names}
    x, y = r.sky_to_pix([val["ra"]], [val["dec"]])[0]
    vdict["x"] = x
    vdict["y"] = y

    return rfig, rax, vdict


def plot_corner(patchname, sid=-1, truths=None, smooth=0.05, hkwargs=dict(alpha=0.65),
                dkwargs=dict(color="red", marker="."), fsize=(8, 8)):
    samples = Samples(patchname)
    labels = samples.chaincat.dtype.names[1:]
    #truths["ra"] -= samples.reference_coordinates[0]
    #truths["dec"] -= samples.reference_coordinates[1]
    truth = np.atleast_2d([truths[l] for l in labels])

    if sid > 0:
        ind = np.where(samples.active["source_index"] == sid)[0]
    else:
        ind = 0

    xx = np.squeeze(np.array([samples.chaincat[ind][l] for l in labels]))

    fig, axes = pl.subplots(7, 7, figsize=fsize)
    axes = allcorner(xx, labels, axes,
                     color="royalblue",  # qcolor="black",
                     psamples=truth.T,
                     smooth=smooth, hist_kwargs=hkwargs,
                     samples_kwargs=dkwargs)
    for i, ax in enumerate(np.diag(axes)):
        ax.axvline(truth[0, i], color="red")
        if (labels[i] == "ra") | (labels[i] == "dec"):
            axes[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
            axes[-1, i].xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    ymap = get_map(samples, xx)
    scatter(ymap.T, axes, zorder=20, color="k", marker=".")
    for ax, val in zip(np.diag(axes), ymap[0]):
        ax.axvline(val, linestyle=":", color="k")

    # this doesn't do anything
    #[ax.set_xlabel(ax.get_xlabel(), labelpad=200) for ax in axes[-1,:]]
    #[ax.set_ylabel(ax.get_ylabel(), labelpad=30) for ax in axes[:, 0]]

    return fig, axes


if __name__ == "__main__":
    args = argparse.Namespace()
    args.root = "../output/sampling_v2.1.3"
    args.snr_thresh = 30

    config, plog, slog, final = fpost.run_metadata(args.root)
    bands = ["F200W"]
    module, pix_scale, sigma_pix = "sw", 0.03, 0.06

    title_fmt = "JAGUAR_ID={id:.0f}\nnsersic={sersic:.1f}, rhalf={rhalf:.2f}, q={q:.2f}"
    truths = np.array(fits.getdata("../data/catalogs/truth_initial_catalog.fits"))
    valid = fits.getdata("../data/catalogs/truth_in_image.fits")
    snr = truths[bands[0]] / 2 / (np.sqrt(np.pi) * truths["rhalf"] / pix_scale * sigma_pix)
    # restrict to forcepho priors
    sel = ((truths["sersic"] > 1.0) & (truths["sersic"] < 5) &
           (truths["rhalf"] < 0.25) & (truths["q"] > 0.4) &
           (valid[f"in_{module.lower()}"] > 0) & (snr > args.snr_thresh))

    # loop over patches
    for pid in plog:
        slist = [s for s in slog.keys() if str(pid) in slog[s]]
        for s in slist:
            sid = int(s)

            if not sel[sid]:
                continue

            patchname = f"{args.root}/patches/patch{pid}_samples.h5"
            truth = truths[sid]
            truth_dict = {c: truths[sid][c] for c in truths.dtype.names}
            title = title_fmt.format(**truth_dict)
            jid = truth["id"]
            print(pid, sid, jid)

            cfig, caxes = plot_corner(patchname, sid=sid, truths=truth.copy())
            cfig.text(0.4, 0.8, title, transform=cfig.transFigure)
            cfig.savefig(f"figures/corner/JAGUARID={jid}_corner.png", dpi=200)
            pl.close(cfig)

            rfig, raxes, val = plot_residual(patchname)
            vdict = deepcopy(truth_dict)
            vdict.update(val)
            vtitle = "Last iteration:" + title_fmt.format(**vdict) + f"\nx={val['x']:.1f}, y={val['y']:.1f}"
            rfig.suptitle(vtitle)
            rfig.savefig(f"figures/residuals/JAGUARID={jid}_residual.png", dpi=200)
            pl.close(rfig)
