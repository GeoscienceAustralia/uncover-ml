from __future__ import division

import ipyparallel as ipp
import logging

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


def apply_and_write(cluster, f, data_var_name, feature_name,
                    outputdir, shape, bbox):

    log.info("Filtering out nodes with no data")

    cluster.execute("has_data = x is not None")
    nodes_with_data = cluster['has_data']
    indices_with_data = [i for i, j in enumerate(nodes_with_data) if j]

    # Filter out no-data nodes
    for new_index, old_index in enumerate(indices_with_data):
        cluster.client[old_index].execute("chunk_index = {}".format(new_index))

    new_cluster = cluster.client[indices_with_data]
    new_cluster.execute("n_chunks = {}".format(len(new_cluster)))

    log.info("Applying transform across nodes")
    # Apply the transformation function

    new_cluster.push({"f": f, "featurename": feature_name, "outputdir":
                      outputdir, "shape": shape, "bbox": bbox})
    log.info("Applying final transform and writing output files")
    new_cluster.execute("f_x = f({})".format(data_var_name))
    new_cluster.execute("outfile = geoio.output_filename(featurename, "
                        "chunk_index, n_chunks, outputdir)")
    new_cluster.execute("geoio.output_features(f_x, outfile, " "shape=shape, "
                        "bbox=bbox)")
