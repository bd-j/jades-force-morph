#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob, os
from argparse import ArgumentParser, Namespace
import numpy as np
from astropy.io import fits

from forcepho.patches.storage import MetaStore, header_to_id


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--framedir", type=str,
                        default="/data/groups/comp-astro/jades/DC2/Morphology/slopes")
    parser.add_argument("--metastorefile", type=str,
                        default="meta.json")
    config = parser.parse_args()

    imnames = glob.glob(os.path.join(config.framedir, "*fits"))

    meta = MetaStore()
    for imname in imnames:
        hdr = fits.getheader(imname)
        band, expID = header_to_id
        imset = Namespace(hdr=hdr, band=band, expID=expID)
        meta.add_exposure(imset)

    metastore.write_to_file(config.metastorefile)