"""
Run cross-validation metrics on a model prediction

.. program-output:: validatemodel --help
"""
import logging
import sys
import json
import os.path
import click as cl
import click_log as cl_log
import matplotlib.pyplot as pl

from uncoverml import mpiops
from uncoverml import geoio
from uncoverml import pipeline
# from uncoverml.validation import input_cvindex, input_targets
from uncoverml.models import apply_multiple_masked

log = logging.getLogger(__name__)


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to save output too")
@cl.option('--plotyy', is_flag=True, help="Show plot of the target vs."
           "prediction, otherwise just save")
@cl.argument('cvindex', type=int)
@cl.argument('targetsfile', type=cl.Path(exists=True))
@cl.argument('prediction_files', type=cl.Path(exists=True), nargs=-1)
def main(cvindex, targetsfile, prediction_files, plotyy, outfile):
    """
    Run cross-validation metrics on a model prediction.

    The following metrics are evaluated:

    - R-square

    - Explained variance

    - Standardised Mean Squared Error

    - Lin's concordance correlation coefficient

    - Mean Gaussian negative log likelihood (for probabilistic predictions)

    - Standardised mean Gaussian negative log likelihood (for probabilistic
      predictions)

    """

    # This runs on the root node only
    if mpiops.chunk_index != 0:
        return

    # build full filenames
    full_filenames = [os.path.abspath(f) for f in prediction_files]
    log.debug("Input files: {}".format(full_filenames))

    # verify the files are all present
    files_ok = geoio.file_indices_okay(full_filenames)
    if not files_ok:
        log.fatal("Input file indices invalid!")
        sys.exit(-1)

    # Load all prediction files
    filename_dict = geoio.files_by_chunk(full_filenames)
    data_vectors = [geoio.load_and_cat(filename_dict[i])
                    for i in range(len(filename_dict))]

    targets = geoio.load_targets(targetsfile)
    scores, Ys, EYs = pipeline.validate(targets, data_vectors, cvindex)

    if outfile is not None:
        with open(outfile + ".json", 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)

    # Make figure
    if plotyy and (outfile is not None):
        fig = pl.figure()
        maxy = max(Ys.max(), get_first_dim(EYs).max())
        miny = min(Ys.min(), get_first_dim(EYs).min())
        apply_multiple_masked(pl.plot, (Ys, get_first_dim(EYs)), ('k.',))
        pl.plot([miny, maxy], [miny, maxy], 'r')
        pl.grid(True)
        pl.xlabel('True targets')
        pl.ylabel('Predicted targets')
        pl.title('True vs. predicted target values.')
        if outfile is not None:
            fig.savefig(outfile + ".png")
        if plotyy:
            pl.show()
