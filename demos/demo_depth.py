#! /usr/bin/env python
""" GLM Demo with compound likelihood"""

import matplotlib.pyplot as pl
import numpy as np
import logging

from revrand import glm
from revrand.basis_functions import RandomRBF
from revrand.btypes import Parameter, Positive
from revrand.utils.datasets import gen_gausprocess_se

from uncoverml.likelihoods import Beta3

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


#
# Settings
#

# Plotting properties
vis_likelihood = True

# Algorithmic properties
nbases = 100
lenscale = 1  # For all basis functions that take lengthscales
rho = 0.9
epsilon = 1e-5
passes = 40
batch_size = 10
kappa = 6
a = 1
b = 1
use_sgd = True

N = 500
Ns = 250

# Dataset properties
lenscale_true = 0.7  # For the gpdraw dataset
a_true = 5
b_true = 1


#
# Make Data
#

Xtrain, ftrain, Xtest, ftest = \
    gen_gausprocess_se(N, Ns, lenscale=lenscale_true, noise=0.0)


# Setup likelihood
like = Beta3(a_init=Parameter(a, Positive()),
             b_init=Parameter(b, Positive()))

fmin = min(ftest.min(), ftrain.min())
offset = 0.1
ftest = ftest - fmin + offset
ftrain = ftrain - fmin + offset
ytrain = like.rvs(ftrain, a_true, b_true)
# ytrain = np.maximum(ytrain, 0)


#
# Visualise likelihood
#

if vis_likelihood:
    y = np.linspace(1e-3, 10 - 1e-3, 1000)
    p = like.pdf(y, 10, a_true, b_true)
    P = like.cdf(y, 10, a_true, b_true)

    pl.plot(y, p, 'b', linewidth=2, label='PDF')
    pl.plot(y, P, 'g', linewidth=2, label='CDF')
    pl.grid(True)
    pl.xlabel('$y$')
    pl.ylabel('$p$')
    pl.legend()
    pl.title('Asym. Laplace, $p(y| f={}, a={}, b={})$'
             .format(10, a_true, b_true))
    pl.show()


#
# Inference
#

basis = RandomRBF(nbases, Xtrain.shape[1],
                  lenscale_init=Parameter(lenscale, Positive()))

params = glm.learn(Xtrain, ytrain, like, basis, use_sgd=use_sgd, rho=rho,
                   epsilon=epsilon, batch_size=batch_size, maxit=passes)

Ey, Vy, Eyn, Eyx = glm.predict_moments(Xtest, like, basis, *params)

m, C = params[0:2]
hypers = params[3]
fs = np.array(list(glm.sample_func(Xtest, basis, m, C, hypers))).T
fmean = fs.mean(axis=1)

y95n, y95x = glm.predict_interval(0.8, Xtest, like, basis, *params,
                                  multiproc=False)


#
# Plot
#

Xpl_t = Xtrain.flatten()
Xpl_s = Xtest.flatten()

# Regressor
pl.plot(Xpl_s, Ey, 'b-', label='GLM mean.')
pl.plot(Xpl_s, fs, 'r:', alpha=0.3)
pl.plot(Xpl_s, fmean, 'r-', label='$\mathbf{f}$ mean')
pl.fill_between(Xpl_s, y95n, y95x, facecolor='none', edgecolor='b', label=None,
                linestyle='--')

# Training/Truth
pl.plot(Xpl_t, ytrain, 'k.', label='Training')
pl.plot(Xpl_s, ftest, 'k-', label='Truth')

pl.gca().invert_yaxis()
pl.legend()
pl.grid(True)
pl.title('Regression demo')
pl.ylabel('depth')
pl.xlabel('x')

pl.show()
