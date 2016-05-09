Uncover-ML project report
=========================

This page contains a brief report on some of the aspects of the uncover ml GA
project not covered by the documentation.

.. contents::

Revrand - large-scale approximate Gaussian processes
----------------------------------------------------

The main algorithm used from revrand_ is the standard linear model for
regression. The aim is to learn a function that maps input covariate values,
:math:`\mathbf{x}_n \in \mathbb{R}^d`, to target values, :math:`y_n \in
\mathbb{R}`. That is, learn :math:`f` such that :math:`y_n = f(\mathbf{x}_n) +
\epsilon`, where :math:`\epsilon` is random (Gaussian) noise. The standard
linear model represents :math:`f` as a linear combination of (non-linear)
*basis* functions, :math:`f(\mathbf{x}_n) = \phi(\mathbf{x}_n, \theta)^\top
\mathbf{w}`, where :math:`\phi: d \to D`, with weights :math:`\mathbf{w} \in
\mathbb{R}^D`. The exact form of the model implemented in revrand is,

.. math::
    
    \text{Likelihood:}& \quad
    \mathbf{y} \sim \prod^N_{n=1} \mathcal{N}(\phi(\mathbf{x}_n)^\top 
        \mathbf{w}, \sigma^2),

    \text{prior:}& \quad
    \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \lambda \mathbf{I}_D),

We then maximise the *log marginal likelihood* of the model to learn the
parameters :math:`\sigma^2, \lambda` and :math:`\theta` from the data (without
over fitting). We can then solve for the posterior over the weights,

.. math::

    \mathbf{w} | \mathbf{y}, \mathbf{X} \sim& \mathcal{N}(\mathbf{m},
        \mathbf{C}),

    \mathbf{C} =& [\lambda^{-1}\mathbf{I}_D + \sigma^{-2}\phi(\mathbf{X})^\top
        \phi(\mathbf{X})]^{-1},

    \mathbf{m} =& \frac{1}{\sigma^2} \mathbf{C} \phi(\mathbf{X})^\top
        \mathbf{y}.

The predictive distribution given a query point, :math:`\mathbf{x}^*` is,

.. math::
    
    p(y^*|\mathbf{x}^*, \mathbf{y}, \mathbf{X}) =& \int
        \mathcal{N}(y^* | \phi(\mathbf{x}^*)^\top \mathbf{w}, \sigma^2)
        \mathcal{N}(\mathbf{w} | \mathbf{m}, \mathbf{C}) d\mathbf{w}

        =& \mathcal{N}\!\left(y^* | \phi(\mathbf{x}^*)^\top \mathbf{m},
            \sigma^2 + \phi(\mathbf{x}^*)^\top \mathbf{C} \phi(\mathbf{x}^*)
            \right).

This model can learn from large datasets, unlike a Gaussian process, which
needs to invert a matrix the dimension of the learning dataset. A Gaussian
process is in general more flexible than the above model, as it specifies a
prior directly over functions as opposed to weights (:math:`\mathbf{f} \sim
\mathcal{N}(\mathbf{0}, \mathbf{K})`, where :math:`\mathbf{K}` is a kernel
matrix). However, the trick implemented in revrand is that by choosing special
types of basis functions (:math:`\phi(\cdot)`) we can approximate the behaviour
of Gaussian processes.  See [1]_ and [2]_ for more information.


Heterogeneous drill observations
--------------------------------

While we did not have time to implement an algorithm to use heterogeneous drill
holes types, i.e. those that do and do not hit the basement, we did establish a
model for incorporating these observations. The basis for this model is a
conditional likelihood model,

.. math::
    
    \text{Likelihood:}& \quad
    \mathbf{y} | \mathbf{z} \sim \prod^N_{n=1}
            \mathcal{N}(\phi(\mathbf{x}_n)^\top \mathbf{w}, \sigma^2)^{z_n}
            \mathcal{B}(\phi(\mathbf{x}_n)^\top \mathbf{w}, \alpha, \beta)
            ^{1 - z_n},

    \text{prior:}& \quad
    \mathbf{w} \sim \mathcal{N}(\mathbf{0}, \lambda \mathbf{I}_D).

Here :math:`z_n` is an indicator variable that is 1 if an observation has hit
the basement, and so uses a Gaussian measurement error model, or 0 if the
basement was not hit, and so uses a *three parameter* Beta_ measurement model,

.. math::

    \mathcal{B}(y | f, \alpha, \beta) = \frac{1}{f^{\alpha + \beta - 1}
        B(\alpha, \beta)} y^{\alpha - 1} (f - y)^{\beta - 1},

where :math:`B(\cdot)` is a Beta function. This is a distribution between
:math:`(0, f)`, with the special case of :math:`\alpha = \beta = 1` being a
uniform distribution. This essentially models the case where our measurement of
depth has to occur between the basement depth :math:`f`, and the surface, 0
with some non-zero probability. There is zero probability of measurement
outside of these bounds.

Inference in this model is more difficult than in the standard linear model,
however there is an implementation of a *generalised* linear model in revrand,
that can be easily extended to use this compound likelihood.


References
----------

.. _revrand: http://github.com/NICTA/revrand
.. _Beta: https://en.wikipedia.org/wiki/Beta_distribution#Four_parameters_2

.. [1] Yang, Z., Smola, A. J., Song, L., & Wilson, A. G. "A la Carte --
   Learning Fast Kernels". Proceedings of the Eighteenth International
   Conference on Artificial Intelligence and Statistics, pp. 1098-1106,
   2015.
.. [2] Rasmussen, C. E., & Williams, C. K. I. "Gaussian Processes for Machine
   Learning", MIT Press, 2006.
.. [3] Rahimi, A., & Recht, B. "Random features for large-scale kernel
   machines." Advances in neural information processing systems. 2007. 
.. [4] Gershman, S., Hoffman, M., & Blei, D. "Nonparametric variational
   inference". arXiv preprint arXiv:1206.4665 (2012).
