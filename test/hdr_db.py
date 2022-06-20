#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, os
from argparse import ArgumentParser, Namespace
import numpy as np
from astropy.io import fits

from forcepho.patches.storage import MetaStore, header_to_id


def find_exposures(metastore, coordinates, bandlist):
    """Return a list of image namesfor all exposures that overlap the
    coordinates. These should be sorted by integer band_id.

    Parameters
    ----------
    region : region.Region instance
        Exposures will be found which overlap this region

    bandlist : list of str
        A list of band names to search for images.

    Returns
    -------
    imnames
    """
    wcs_origin = 1
    imsize = np.zeros(2) + 2048
    bra, bdec = coordinates

    epaths, bands = [], []
    for band in bandlist:
        if band not in metastore.wcs.keys():
            continue
        for expID in metastore.wcs[band].keys():
            #print(expID)
            epath = "{}/{}".format(band, expID)
            wcs = metastore.wcs[band][expID]
            # Check region bounding box has a corner in the exposure.
            # NOTE: If bounding box entirely contains image this might fail
            bx, by = wcs.all_world2pix(bra, bdec, wcs_origin)
            inim = np.any((bx > 0) & (bx < imsize[0]) &
                          (by > 0) & (by < imsize[1]))
            if inim:
                epaths.append(expID)
                bands.append(band)
    return epaths, bands


def make_metastore(config):

    imnames = glob.glob(os.path.join(config.framedir, "*smr*fits"))

    meta = MetaStore()
    for imname in imnames:
        hdr = fits.getheader(imname)
        band, expID = header_to_id(hdr, imname)
        imset = Namespace(hdr=hdr, band=band, expID=expID)
        meta.add_exposure(imset)

    meta.write_to_file(config.metastorefile)


def make_image_sets(config, bandlist):
    meta = MetaStore(config.metastorefile)
    band = bandlist[0]

    elist = list(meta.headers[band].keys())
    allexps = []
    expsets = []
    for exp in elist:
        if exp in allexps:
            continue
        hdr = meta.headers[band][exp]
        coordinates = np.array([hdr["CRVAL1"], hdr["CRVAL2"]])
        exp_set, bandset = find_exposures(meta, coordinates, bandlist)
        expsets.append(exp_set)
        allexps += exp_set

    return expsets


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--framedir", type=str,
                        default="/data/groups/comp-astro/jades/DC2/Morphology/slopes")
    parser.add_argument("--metastorefile", type=str,
                        default="./meta-morph.json")
    parser.add_argument("--bands", type=str, nargs="*", default=["F200W", "F277W"])
    config = parser.parse_args()

    if False:
        make_metastore(config)

    if True:
        bands = config.bands
        expsets = make_image_sets(config, bands)
        for i, expset in enumerate(expsets):
            with open(f"explists/explist_{'+'.join(bands)}_{i}.txt", "w") as out:
                for e in expset:
                    out.write(f"{e}\n")
