from __future__ import division

import ipyparallel as ipp
import logging

log = logging.getLogger(__name__)


def direct_view(profile):
    client = ipp.Client(profile=profile) if profile is not None \
        else ipp.Client()
    c = client[:]  # direct view
    nchunks = len(c)

    # Initialise the cluster
    c.block = True
    # Ensure this module's requirments are imported externally
    c.execute('import numpy as np')
    c.execute('from uncoverml import geoio')
    c.execute('from uncoverml import patch')
    c.execute('from uncoverml import parallel')
    c.execute('from uncoverml import stats')

    log.info("dividing work between {} engines".format(nchunks))
    for i in range(nchunks):
        cmd = "chunk_index = {}".format(i)
        c.execute(cmd, targets=i)

    return c


def apply_and_write(cluster, f, data_var_name, feature_name,
                    outputdir, shape, bbox):
    log.info("Applying transform across nodes")
    # Apply the transformation function
    cluster.push({"f": f, "featurename": feature_name, "outputdir": outputdir,
                  "shape": shape, "bbox": bbox})
    log.info("Applying final transform and writing output files")
    cluster.execute("f_x = f({})".format(data_var_name))
    cluster.execute("outfile = geoio.output_filename(featurename, "
                    "chunk_index, outputdir)")
    cluster.execute("geoio.output_features(f_x, outfile, "
                    "shape=shape, bbox=bbox)")
