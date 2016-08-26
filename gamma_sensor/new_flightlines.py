#!/usr/bin/env python3

""" Investigating the Gamma sensor for the GA-cover project
    Disclaimer: prototype code.
    Here we assume a constant altitude for a stationary (sensor position
    invariant) sensor footprint shape.
"""

import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
from scipy import sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
from bisect import bisect_left
import shapefile  # package pyshp
import os
import numpy as np
import logging

from prototype_gamma_sensor import get_flightlines, LocalFrame, data_root
log = logging.getLogger(name=__name__)


mass_attenuation_air = 0.09  # assume a bulk property...
density_air = 1.22
mu_air = mass_attenuation_air * density_air



def main():
    """Main demo."""

    # Load survey data
    llh, data = get_flightlines()

    decimate = 5
    llh = llh[::decimate]
    data = data[::decimate]


    # Select by height intervals  (57% of the data)
    hrange = [95., 105.]
    keep = np.bitwise_and(llh[:, 2] > hrange[0], llh[:,2] < hrange[1])
    llh = llh[keep]
    data = data[keep]

    # Write out the reduced llh, data
    sf = shapefile.Writer(shapefile.POINT)
    outname = data_root + 'new_flightlines'
    log.info('Writing shapefile')
    sf.field("K")
    sf.field("Th")
    sf.field("U")
    for ll, dat in tqdm(zip(llh, data)):
        sf.point(ll[0], ll[1], ll[2])
        sf.record(K=dat[0], Th=dat[1], U=dat[2])
    sf.save(outname)
    log.info('Done!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    np.set_printoptions(formatter={'float': (lambda f: '%.2f' % f)})
    main()
    cache.close()
