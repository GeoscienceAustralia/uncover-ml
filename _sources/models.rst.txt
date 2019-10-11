Models Module
=============

This module makes many of the models in `scikit learn
<http://scikit-learn.org/>`_ and `revrand <https://github.com/NICTA/revrand>`_
available to our pipeline, as well as augmenting their functionality with, for
examples, target transformations.

This table is a quick breakdown of the advantages and disadvantages of the
various algorithms we can use in this pipeline.

==========================  ====================  ==================  ================  =============
Algorithm                   Learning Scalability  Modelling Capacity  Prediction Speed  Probabilistic
==========================  ====================  ==================  ================  ============= 
Bayesian linear regression  \+ \+ \+              \+                  \+ \+ \+ \+       Yes
Approx. Gaussian process    \+ \+                 \+ \+ \+ \+         \+ \+ \+ \+       Yes
SGD linear regression       \+ \+ \+ \+           \+                  \+ \+ \+          Yes
SGD Gaussian process        \+ \+ \+ \+           \+ \+ \+ \+         \+ \+ \+          Yes
Support Vector Regression   \+                    \+ \+ \+ \+         \+                No
Random Forest Regression    \+ \+ \+              \+ \+ \+ \+         \+ \+             Pseudo
Cubist Regression           \+ \+ \+              \+ \+ \+ \+         \+ \+             Pseudo
ARD Regression              \+ \+                 \+ \+               \+ \+ \+          No
Extremely Randomized Reg.   \+ \+ \+              \+ \+ \+ \+         \+ \+             No
Decision Tree Regression    \+ \+ \+              \+ \+ \+            \+ \+ \+ \+       No
==========================  ====================  ==================  ================  =============

.. automodule:: uncoverml.models
   :members:
