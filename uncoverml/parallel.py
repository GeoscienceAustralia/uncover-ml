from __future__ import division

import logging
import ipyparallel as ipp

log = logging.getLogger(__name__)


def direct_view(profile, n_chunks=None):
    client = ipp.Client(profile=profile) if profile is not None \
        else ipp.Client()
    c = client[:]  # direct view
    nclients = len(c)
    if n_chunks is None:
        n_chunks = nclients
    assert n_chunks <= nclients
    c = client[:n_chunks]

    # Initialise the cluster
    c.block = True
    # Ensure this module's requirments are imported externally
    c.execute('import numpy as np')
    c.execute('from uncoverml import geoio')
    c.execute('from uncoverml import patch')
    c.execute('from uncoverml import parallel')
    c.execute('from uncoverml import stats')

    log.info("dividing work between {} engines".format(n_chunks))
    for i in range(n_chunks):
        cmd = "chunk_index = {}".format(i)
        c.execute(cmd, targets=i)

    return c


def apply_and_write(f, x, feature_name,
                    outputdir, shape, bbox):

    log.info("Filtering out nodes with no data")
    has_data = x is not None
    if has_data:
        log.info("Applying final transform and writing output files")
        f_x = f(x)
        outfile = geoio.output_filename(featurename, chunk_index,
                                        n_chunks, outputdir)
        write_ok = geoio.output_features(f_x, outfile, shape=shape, bbox=bbox)
    else:
        write_ok = geoio.output_blank(chunk_index, n_chunks)


