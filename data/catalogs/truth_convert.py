#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from astropy.io import fits


if __name__ == "__main__":

    with open("truth_table.txt", "r") as f:
        hdr = f.readline()[1:].split()

    dtype = np.dtype([(c, int) for c in hdr[:2]] + [(c, float) for c in hdr[2:]])
    cat = np.genfromtxt("truth_table.txt", skip_header=1, dtype=dtype)
    fits.writeto("truth_table.fits", cat)