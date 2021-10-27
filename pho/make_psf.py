#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, sys
import argparse

import numpy as np
from astropy.io import fits
import h5py

from forcepho.mixtures.utils_hmc import Image, infer, display, radial_plot, draw_ellipses
from forcepho.mixtures.psf_mix_hmc import psf_model, psf_prediction


__all__ = ["read_image", "fit_image",
           "convert_psf_data", "package_output", "make_psfstore"]


def read_image(filename, pad=10, ext=0):
    """Read a FITS image into flattenend 1-d array

    From the PSF directory README:
        Each file contains 4 image extensions:

        OVERSAMP : PSF oversampled by 4x relative to detector sampling.
        DET_SAMP : PSF binned to detector sampling. This will not be Nyquist sampled in all cases.
        OVERDIST : Oversampled PSF, modified based on expected geometric distortion at center of that instrument’s FOV
        DET_DIST : Detector sampled PSF, similarly modified based on the expected geometric distortion.

        Notes
        1. More information about the different parameters used in these simulations can be found inside the fits file headers.
        2. For the source spectrum, these calculations use a G0V stellar spectral type from the Castelli & Kurucz model libraries.

    Also note that the source is centered *between* the pixels for even size
    arrays, and at the center of a pixel for odd-size arrays. In the inference
    code the convention is that the center of a pixel is at the integer value of
    that pixel.

    Parameters
    ----------
    pad : int
        remove this many rows or columns from each edge of the image before fitting.

    """
    # NOTE: Image is transposed here! so x is the leading dimension
    data = (fits.getdata(filename, ext=ext).T).astype(np.float)
    slx, sly = slice(pad, data.shape[0] - pad), slice(pad, data.shape[1] - pad)
    data = data[slx, sly]

    nx, ny = data.shape
    cx, cy = (nx - 1) / 2., (ny-1) / 2.
    ypix, xpix = np.meshgrid(np.arange(ny), np.arange(nx))
    xpix, ypix = xpix.flatten(), ypix.flatten()
    data = data.flatten()
    #unc = data.max() / snr
    im = Image(xpix, ypix, data, 0, nx, ny, cx, cy)

    hdr = fits.getheader(filename, ext=ext)

    return im, hdr


def fit_image(image, args, a=None, oversample=1, **kwargs):
    """This wrapper on infer() that computes some required
    input parameters based on image and arguments
    """
    if (not a):
        a = image.data.sum()
    unc = a / (args.snr * np.sqrt(image.nx * image.ny))
    #unc = np.hypot(unc/10, image.data / args.snr)
    if args.fix_amplitude:
        kwargs["afix"] = a
    else:
        kwargs["amax"] = a * 2
    if args.fit_bg:
        kwargs["maxbg"] = np.abs(np.median(image.data))

    # let some big ole gaussians in
    kwargs["smax"] = np.linspace(5, 20, args.ngauss) * oversample
    # keep teeny-tiny gaussians out
    kwargs["smin"] = oversample * 0.5

    best, samples, mcmc = infer(psf_model, image=image.data,
                                xpix=image.xpix, ypix=image.ypix,
                                ngauss=args.ngauss, unc=unc,
                                dense_mass=True,
                                num_warmup=args.num_warmup,
                                num_samples=args.num_samples,
                                **kwargs)

    if args.fix_amplitude:
        best["a"] = a

    return best, samples, mcmc, unc


def convert_psf_data(best, cx, cy, nloc=1, nradii=9, scale_factor=1.):
    """
    Parameters
    ----------
    best : dict
        Dictionary of psf Gaussian parameters, which are vectors of length
        `ngauss` when appropriate.

    cx : float
        Center of PSF, in PSF image pixels

    cy : float
        Center of PSF, in PSF image pixels

    scale_factor : float
        Ratio of the plate scale in the PSF image (e.g. in mas) to the plate
        scale of the science image.  So if the PSF image if oversampled by a
        factor of 2, then this number is 0.5

    nloc : int
        Make `nloc` copies of the PSF parameter arrays - this will be the first
        dimesion of the output.

    nradii : int
        Make `nradii` copies of the PSF parameter arrays for each location; the
        "sersic_bin" field will be set to an integer corresponding to the index
        of the copy.  This will be the second dimension of the output.

    Returns
    -------
    pars : structured ndarray of shape (nloc, nradii, ngauss)
    """
    ngauss = len(best["x"])

    # This order is important
    cols = ["amp", "xcr", "ycr", "Cxx", "Cyy", "Cxy"]
    pdt = np.dtype([(c, np.float32) for c in cols] +
                   [("sersic_bin", np.int32)])
    pars = np.zeros([nloc, nradii, ngauss], dtype=pdt)

    o = np.argsort(np.array(best["weight"]))[::-1]
    scale = best.get("a", 1.0)
    pars["amp"] = best["weight"][o] * scale
    pars["xcr"] = (best["x"][o] - cx) * scale_factor
    pars["ycr"] = (best["y"][o] - cy) * scale_factor
    sx = (best["sx"][o] * scale_factor)
    sy = (best["sy"][o] * scale_factor)
    pars["Cxx"] = sx**2
    pars["Cyy"] = sy**2
    cxy = sx * sy * best["rho"][o]
    pars["Cxy"] = cxy
    pars["sersic_bin"] = np.arange(nradii)[None, :, None]

    return pars, np.float(scale)


def revert_psf_data(pars, cx=0, cy=0, scale_factor=1):
    best = {}
    best["weight"] = pars["amp"]
    best["x"] = pars["xcr"] / scale_factor + cx
    best["y"] = pars["ycr"] / scale_factor + cy
    sx = np.sqrt(pars["Cxx"])
    sy = np.sqrt(pars["Cyy"])
    rho = pars["Cxy"] / sx / sy
    best["sx"] = sx / scale_factor
    best["sy"] = sy / scale_factor
    best["rho"] = rho

    return best


def package_output(h5file, band, pars, scale, image, model=None):
    """
    """
    with h5py.File(h5file, "a") as h5:
        bg = h5.create_group(band)
        bg.create_dataset("parameters", data=pars.reshape(pars.shape[0], -1))
        im = image.data.reshape(image.nx, image.ny)
        bg.create_dataset("truth", data=image.data)
        bg.create_dataset("xpix", data=image.xpix)
        bg.create_dataset("ypix", data=image.ypix)
        bg.attrs["n_psf_per_source"] = pars.shape[1] * pars.shape[2]
        bg.attrs["nx"] = image.nx
        bg.attrs["ny"] = image.ny
        bg.attrs["cx"] = image.cx
        bg.attrs["cy"] = image.cy
        bg.attrs["flux_scale"] = scale
        if model is not None:
            m = model.reshape(image.nx, image.ny)
            bg.create_dataset("model", data=model)


def fitsify_output(fn, pars, image, model=None, oversample=1, band="F090W", **hdr_kwargs):

    n = 1
    im = image.data.reshape(image.nx, image.ny)
    pri = fits.PrimaryHDU()
    pri.header["FILTER"] = band.upper()
    hdulist = fits.HDUList(hdus=[pri, fits.ImageHDU(im.T)])
    pri.header[f"EXT{n}"] = "True PSF"
    n += 1
    if model is not None:
        m = model.reshape(image.nx, image.ny)
        hdulist += [fits.ImageHDU(m.T)]
        pri.header[f"EXT{n}"] = "Model PSF"
        n += 1

    hdulist += [fits.BinTableHDU(pars)]
    pri.header[f"EXT{n}"] = "Table of Gaussian Parameters"

    pri.header["SOURCEX"] = image.cx
    pri.header["SOURCEY"] = image.cy
    pri.header["DET_SAMP"] = oversample
    for k, v in hdr_kwargs.items():
        pri.header[k] = v

    hdulist.writeto(fn, overwrite=True)
    return hdulist


if __name__ == "__main__":

    swbands = ["f070w", "f090w", "f115w", "f150w", "f200w"]
    lwbands = ["f277w", "f335m", "f356w", "f410m", "f444w"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--bands", type=str, nargs="*",
                        default=swbands+lwbands)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--imdir", type=str, default="")
    # model
    parser.add_argument("--ngauss", type=int, default=3)
    parser.add_argument("--ngauss_neg", type=int, default=0)
    parser.add_argument("--fix_amplitude", type=int, default=0)
    parser.add_argument("--fit_bg", type=int, default=0)
    parser.add_argument("--snr", type=int, default=100)
    # fitting
    parser.add_argument("--num_warmup", type=int, default=1024)
    parser.add_argument("--num_samples", type=int, default=1024)
    # display
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--savefig", action="store_true")
    args = parser.parse_args()

    odir = os.path.dirname(args.output)
    if odir:
        os.makedirs(odir, exist_ok=True)

    ext = 0

    # TODO: use multiprocess for this loop
    for i, b in enumerate(args.bands):
        filename = glob.glob(os.path.join(args.imdir, f"*{b.upper()}*.fits"))[0]
        print("------\nworking on {}:".format(filename))
        pad = int(500 / (1 + 3 * int(b in lwbands)))
        print(pad)
        image, hdr = read_image(filename, pad=pad, ext=ext)
        oversample = hdr["DET_SAMP"]
        best, samples, mcmc, unc = fit_image(image, args, a=1.0,
                                             oversample=oversample, dcen=oversample / 2,
                                             max_tree_depth=11, ngauss_neg=args.ngauss_neg)
        best_m = {}
        for p in ["a", "sx", "sy", "rho", "x", "y", "weight"]:
            try:
                best_m[p] = best.pop(f"{p}_m")
            except(KeyError):
                pass

        model = psf_prediction(image.xpix, image.ypix, **best)
        if len(best_m):
            model_m = psf_prediction(image.xpix, image.ypix, **best_m)
            model -= model_m
        chi = (model - image.data) / image.data.max()

        if args.output:
            pars, scale = convert_psf_data(best, image.cx, image.cy, scale_factor=1/oversample)
            outname = f"{args.output}_ng{args.ngauss}m{args.ngauss_neg}.h5"
            package_output(outname, b, pars, scale, image, model=model)
            ff = fitsify_output(outname.replace(".h5", f"_{b}.fits"), pars[0, 0], image, model=model,
                                oversample=oversample, flxscl=scale, psfim=os.path.basename(filename))

        if args.show:
            import matplotlib.pyplot as pl
            pl.ion()

        if args.show or args.savefig:
            fig, axes = display(model, image)
            title = r"error as % of peak:"
            title += r"min={:.2f}%, max={:.2f}%".format(chi.min()*100, chi.max()*100)
            title += "\nSummed difference: "
            title += r"{:.2f}%".format(100 * (model - image.data).sum()/image.data.sum())
            fig.suptitle(title)

            import arviz as az
            data = az.from_numpyro(mcmc)
            azax = az.plot_trace(data, compact=True)
            azfig = azax.ravel()[0].figure

            rfig, raxes = radial_plot(image, model)
            if unc.shape == ():
                raxes[0].axhline(unc)
            import matplotlib.pyplot as pl
            efig, eax = pl.subplots(figsize=(5, 5))
            eax = draw_ellipses(best, eax)
            eax.set_xlim(0, image.nx)
            eax.set_ylim(0, image.ny)
            eax.grid()

            #mcmc.print_summary()
            print(title)
            print(f"Divergences: {mcmc.get_extra_fields('diverging')['diverging'].sum()}")

        if args.savefig:
            outname = f"{args.output}_ng{args.ngauss}m{args.ngauss_neg}_{b}.pdf"
            from matplotlib.backends.backend_pdf import PdfPages
            with PdfPages(outname) as pdf:
                pdf.savefig(fig)
                pdf.savefig(rfig)
                pdf.savefig(efig)
                pdf.savefig(azfig)
            pl.close('all')