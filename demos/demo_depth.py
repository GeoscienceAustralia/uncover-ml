#! /usr/bin/env python
""" GLM Demo with compound likelihood """

import matplotlib.pyplot as pl
import numpy as np
import logging

from revrand import glm
from revrand.basis_functions import RandomRBF
from revrand.btypes import Parameter, Positive
from revrand.utils.datasets import gen_gausprocess_se
from revrand.mathfun.special import softplus
# from revrand.likelihoods import Gaussian

from uncoverml.likelihoods import Switching, UnifGauss

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


#
# Settings
#

# Plotting properties
vis_likelihood = False

# Dataset properties
N = 100
Ns = 250
offset = 20
lenscale_true = 0.7  # For the gpdraw dataset
noise_true = 0.5
hit_proportion = 0.0

# Algorithmic properties
nbases = 100
lenscale = 1  # For all basis functions that take lengthscales
rho = 0.9
epsilon = 1e-5
passes = 5 * 40
batch_size = 10
kappa = 6
use_sgd = True
regulariser = 1.
noise = 1.0

#
# Make Data
#

hit = np.random.binomial(n=1, p=hit_proportion, size=N).astype(bool)
not_hit = ~ hit
Xtrain, ftrain, Xtest, ftest = \
    gen_gausprocess_se(N, Ns, lenscale=lenscale_true, noise=0.0)

gtrain = softplus(ftrain + offset)
gtest = softplus(ftest + offset)

ytrain = np.empty(N)
ytrain[hit] = gtrain[hit] + np.random.randn(hit.sum()) * noise_true
ytrain[not_hit] = np.random.rand(not_hit.sum()) * gtrain[not_hit]


#
# Transform Inputs for learning
#

ymean = ytrain.mean()
ytrain -= ymean

# Setup likelihood
var = Parameter(noise**2, Positive(1.))
# like = UnifGauss(y0=-ymean)
like = Switching(y0=-ymean, var_init=var)
# like = Gaussian(var)

#
# Visualise likelihood
#

if vis_likelihood:
    print(ymean)

    f = ytrain.mean()
    y = np.linspace(-ymean, f + 3, 1000)
    p = like.pdf(y, f)
    P = like.cdf(y, f)
    Ey = like.Ey(f)

    pl.plot(y, p, 'b', linewidth=2, label='PDF')
    pl.plot(y, P, 'g', linewidth=2, label='CDF')
    pl.plot([Ey, Ey], [0., 1.], 'r-', linewidth=2, label='Ey')
    pl.grid(True)
    pl.xlabel('$y$')
    pl.ylabel('$p$')
    pl.legend()
    pl.title('Asym. Laplace, $p(y| f={})$'
             .format(f))
    pl.show()

#
# Inference
#

basis = RandomRBF(nbases, Xtrain.shape[1],
                       lenscale_init=Parameter(lenscale, Positive()))
regulariser = Parameter(regulariser, Positive())

params = glm.learn(Xtrain, ytrain, like, basis, regulariser=regulariser,
                   likelihood_args=(hit,), use_sgd=use_sgd, rho=rho,
                   # likelihood_args=(), use_sgd=use_sgd, rho=rho,
                   epsilon=epsilon, batch_size=batch_size, maxit=passes)

hit_predict = np.ones(Ns, dtype=bool)
Ey, Vy, Eyn, Eyx = glm.predict_moments(Xtest, like, basis, *params,
                                       likelihood_args=(hit_predict,))
                                       # likelihood_args=())

m, C = params[0:2]
hypers = params[3]
fs = np.array(list(glm.sample_func(Xtest, basis, m, C, hypers))).T
gmean = softplus(fs).mean(axis=1)

y95n, y95x = glm.predict_interval(0.8, Xtest, like, basis, *params,
                                  multiproc=False,
                                  # likelihood_args=())
                                  likelihood_args=(hit_predict,))

#
# Untransform targets
#

Ey += ymean
y95n += ymean
y95x += ymean
ytrain += ymean


#
# Plot
#

Xpl_t = Xtrain.flatten()
Xpl_s = Xtest.flatten()

# Regressor
pl.plot(Xpl_s, Ey, 'b-', label='GLM mean.')
pl.plot(Xpl_s, gmean, 'r-', label='$\mathbf{g}(\mathbf{f})$ mean')
# pl.fill_between(Xpl_s, y95n, y95x, facecolor='none', edgecolor='b', label=None,
                # linestyle='--')

# # Training/Truth
pl.plot(Xpl_t[hit], ytrain[hit], 'k.', label='Training (hit)')
pl.plot(Xpl_t[not_hit], ytrain[not_hit], 'kx', label='Training (not hit)')
pl.plot(Xpl_s, gtest, 'k-', label='Truth')

pl.gca().invert_yaxis()
pl.legend()
pl.grid(True)
pl.title('Regression demo')
pl.ylabel('depth')
pl.xlabel('x')

pl.show()
