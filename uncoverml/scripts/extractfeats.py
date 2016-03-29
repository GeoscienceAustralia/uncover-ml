"""
Extract patch features from a single geotiff.

.. program-output:: extractfeats --help
"""
import logging
from functools import partial
import os
import click as cl
import uncoverml.feature as feat
from uncoverml import geoio
import uncoverml.defaults as df
from uncoverml import parallel

log = logging.getLogger(__name__)


def myfunc(x, otherarg):
    return x * 2 * otherarg

@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--patchsize', type=int,
           default=df.feature_patch_size, help="window size of patches")
@cl.option('--chunks', type=int, default=df.work_chunks, help="Number of "
           "chunks in which to split the computation and output")
@cl.option('--standalone', is_flag=True, default=df.standalone)
@cl.option('--targets', type=cl.Path(exists=True), help="Optional hdf5 file "
           "for providing target points at which to evaluate feature. See "
           "maketargets for creating an appropriate target files.")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.option('--ipyprofile', type=str, help="ipyparallel profile to use", 
           default=None)
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
@cl.argument('name', type=str, required=True)
def main(geotiff, name, targets, standalone, chunks, patchsize, 
         quiet, outputdir, ipyprofile):
    """ TODO
    """
    
    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # build full filename for geotiff
    full_filename = os.path.abspath(geotiff)
    log.debug("Input file full path: {}".format(full_filename))

    # build the images
    image_chunks = [geoio.Image(full_filename, i, chunks) for i in range(chunks)]

    #Build the function to call
    f = partial(feat.features_from_image, name=name, transform=feat.transform, 
                patchsize=patchsize, output_dir=outputdir, targets=targets)

    cluster = parallel.task_view(ipyprofile) if not standalone else None
    parallel.map(f, image_chunks, cluster)
    
