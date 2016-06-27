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
import numpy as np
import matplotlib.pyplot as pl
from mpi4py import MPI

from sklearn.metrics import r2_score, explained_variance_score
from revrand.metrics import smse, mll, msll, lins_ccc

from uncoverml import geoio
# from uncoverml.validation import input_cvindex, input_targets
from uncoverml.models import apply_multiple_masked

log = logging.getLogger(__name__)


def get_first_dim(Y):

    return Y[:, 0] if Y.ndim > 1 else Y


# Decorator to deal with probabilistic output for non-probabilistic scores
def score_first_dim(func):

    def newscore(y_true, y_pred, *args, **kwargs):

        return func(y_true.flatten(), get_first_dim(y_pred), *args, **kwargs)

    return newscore

metrics = {'r2_score': r2_score,
           'expvar': explained_variance_score,
           'smse': smse,
           'lins_ccc': lins_ccc,
           'mll': mll,
           'msll': msll
           }

probscores = ['msll', 'mll']


@cl.command()
@cl_log.simple_verbosity_option()
@cl_log.init(__name__)
@cl.option('--outfile', type=cl.Path(exists=False), default=None,
           help="File name (minus extension) to save output too")
@cl.option('--plotyy', is_flag=True, help="Show plot of the target vs."
           "prediction, otherwise just save")
@cl.argument('cvindex', type=int)
@cl.argument('targets', type=cl.Path(exists=True))
@cl.argument('prediction_files', type=cl.Path(exists=True), nargs=-1)
def main(cvindex, targets, prediction_files, plotyy, outfile):
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

    # MPI globals
    comm = MPI.COMM_WORLD
    chunk_index = comm.Get_rank()
    # This runs on the root node only
    if chunk_index != 0:
        return

    # Read cv index and targets
    ydict = geoio.points_from_hdf(targets, ['targets_sorted',
                                            'FoldIndices_sorted'])
    Y = ydict['targets_sorted']
    cvind = ydict['FoldIndices_sorted']
    # cvind = input_cvindex(cvindex[0])
    # _, Y = input_targets(targets)

    s_ind = np.where(cvind == cvindex)[0]
    t_ind = np.where(cvind != cvindex)[0]

    Yt = Y[t_ind]
    Ys = Y[s_ind]
    Ns = len(Ys)

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
    EYs = np.ma.concatenate(data_vectors, axis=0)

    # See if this data is already subset for xval
    if len(EYs) > Ns:
        EYs = EYs[s_ind]

    scores = {}
    for m in metrics:

        if m not in probscores:
            score = apply_multiple_masked(score_first_dim(metrics[m]),
                                          (Ys, EYs))
        elif EYs.ndim == 2:
            if m == 'mll' and EYs.shape[1] > 1:
                score = apply_multiple_masked(mll, (Ys, EYs[:, 0], EYs[:, 1]))
            elif m == 'msll' and EYs.shape[1] > 1:
                score = apply_multiple_masked(msll, (Ys, EYs[:, 0], EYs[:, 1]),
                                              (Yt,))
            else:
                continue
        else:
            continue

        scores[m] = score
        log.info("{} score = {}".format(m, score))

    if outfile is not None:
        with open(outfile + ".json", 'w') as f:
            json.dump(scores, f, sort_keys=True, indent=4)

    # Make figure
    if plotyy or (outfile is not None):
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
