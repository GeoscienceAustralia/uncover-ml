Models Module
=============

This module makes many of the models in `scikit learn
<http://scikit-learn.org/>`_ and `revrand <https://github.com/NICTA/revrand>`_
available to our pipeline, as well as augmenting their functionality with, for
examples, target transformations.

This table is a quick breakdown of the advantages and disadvantages of the
various algorithms we can use in this pipeline.

==========================  ====================  ==================  ================
Algorithm                   Learning Scalability  Modelling Capacity  Prediction Speed
==========================  ====================  ==================  ================
Bayesian linear regression  \+ \+ \+              \+                  \+ \+ \+ \+
Approx. Gaussian process    \+ \+                 \+ \+ \+ \+         \+ \+ \+ \+
SGD linear regression       \+ \+ \+ \+           \+                  \+ \+ \+
SGD Gaussian process        \+ \+ \+ \+           \+ \+ \+ \+         \+ \+ \+
==========================  ====================  ==================  ================

.. automodule:: uncoverml.models
   :members:
