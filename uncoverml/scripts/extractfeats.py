import logging
import os
import click as cl
import uncoverml.feature as feat
from uncoverml import geoio
import uncoverml.defaults as df
from uncoverml import celerybase
log = logging.getLogger(__name__)


@cl.command()
@cl.option('--quiet', is_flag=True, help="Log verbose output",
           default=df.quiet_logging)
@cl.option('--patchsize', type=int,
           default=df.feature_patch_size, help="window size of patches")
@cl.option('--chunks', type=int, default=df.work_chunks, help="Number of "
           "chunks in which to split the computation and output")
@cl.option('--redisdb', type=int, default=df.redis_db)
@cl.option('--redishost', type=str, default=df.redis_address)
@cl.option('--redisport', type=int, default=df.redis_port)
@cl.option('--standalone', is_flag=True, default=df.standalone)
@cl.option('--targets', type=cl.Path(exists=True), help="Optional shapefile "
           "for providing target points at which to evaluate feature")
@cl.option('--outputdir', type=cl.Path(exists=True), default=os.getcwd())
@cl.argument('geotiff', type=cl.Path(exists=True), required=True)
@cl.argument('name', type=str, required=True)
def main(geotiff, name, targets, redisdb, redishost, redisport, standalone,
         chunks, patchsize, quiet, outputdir):
    """ TODO
    """

    # setup logging
    if quiet is True:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # initialise celery
    celerybase.configure(redishost, redisport, redisdb, standalone)

    # build the images
    image_chunks = [geoio.Image(geotiff, i, chunks) for i in range(chunks)]

    # Define the transform function to build the features
    transform = feat.transform

    celerybase.map_over(feat.features_from_image, image_chunks, standalone,
                        name=name, transform=transform, patchsize=patchsize,
                        output_dir=outputdir, targets=targets)
