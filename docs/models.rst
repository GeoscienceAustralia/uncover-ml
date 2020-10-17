.. _models-page:

Models
======

UncoverML has a variety of algorithms to choose from.

*Config* shows the ``algorithm`` parameter to add to the ``learning`` block of the
config to use the model. 

The majority of models are subclassed from models available in scikit-learn
or the NICTA Revrand package. Follow the links under *Documentation and parameters* 
to view the specific model documentation and available parameters from these resources.

*Additional parameters* covers parameters specific to 
UncoverML that can be added to the ``arguments`` section of the ``learning``
block.

Regressors
----------

Random Forest
~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: randomforest

**Documentation and parameters:**

- `Scikit Learn <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_
- :class:`uncoverml.models.RandomForestTransformed`

**Additional parameters:**

- ``target_transform``

Multi Random Forest
~~~~~~~~~~~~~~~~~~~

An ensemble of Random Forest predictors.

**Config:**

.. code:: yaml

    algorithm: multirandomforest

**Documentation and parameters:**

- See :ref:`Random Forest`
- :class:`uncoverml.models.MultiRandomForestTransformed`

**Additional parameters:**

- ``target_transform``
- ``forests``: number of Random Forest submodels
- ``parallel``: boolean, whether to train this model in parallel

  - If ``parallel`` is True, this model can be trained using multiple processors.
    See :ref:`Multiprocessing and Partitioning`.

Bayes Regression/Standard Linear Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: bayesreg

**Documentation and parameters:**

- `NICTA Revrand <http://nicta.github.io/revrand/slm.html>`_
- :class:`uncoverml.models.LinearReg`

**Additional parameters:**

- ``target_transform``

Stochastic Gradient Descent Bayes Regression/Standard Linear Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: sgdbayesreg

**Documentation and parameters:**

- `NICTA Revrand <http://nicta.github.io/revrand/glm.html>`_
- :class:`uncoverml.models.SGDLinearReg`

**Additional parameters:**

- ``target_transform``

Approximate Gaussian Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of :ref:`Bayes Regression/Standard Linear Model`

**Config:**

.. code:: yaml

    algorithm: approxgp

**Documentation and parameters:**

- `NICTA Revrand <http://nicta.github.io/revrand/slm.html>`_
- :class:`uncoverml.models.ApproxGP`

**Additional parameters:**

- ``target_transform``

Stochastic Gradient Descent Approximate Gaussian Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of :ref:`Stochastic Gradient Descent Bayes Regression/Standard Linear Model`

**Config:**

.. code:: yaml

    algorithm: sgdapproxgp

**Documentation and parameters:**

- `NICTA Revrand <http://nicta.github.io/revrand/glm.html>`_
- :class:`uncoverml.models.SGDApproxGP`

**Additional parameters:**

- ``target_transform``

Support Vector Regression
~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: svr

**Documentation and parameters:**

- `Scitkit-Learn <https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVR.html>`_
- :class:`uncoverml.models.SVRTransformed`

**Additional parameters:**

- ``target_transform``

Automatic Relevance Determination Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: ardregression

**Documentation and parameters:**

- `Scitkit-Learn <https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.ARDRegression.html>`_
- :class:`uncoverml.models.ARDRegressionTransformed`

**Additional parameters:**

- ``target_transform``

Decision Tree
~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: decisiontree

**Documentation and parameters:**

- `Scitkit-Learn <https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeRegressor.html>`_
- :class:`uncoverml.models.DecisionTreeTransformed`

**Additional parameters:**

- ``target_transform``

Extra Tree
~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: extratree

**Documentation and parameters:**

- `Scitkit-Learn <http://scikit-learn.org/dev/modules/generated/sklearn.tree.ExtraTreeRegressor.html>`_
- :class:`uncoverml.models.ExtraTreeTransformed`

**Additional parameters:**

- ``target_transform``

Cubist
~~~~~~

**Config:**

.. code:: yaml

    algorithm: cubist

**Documentation and parameters:**

- `Rule-Quest <https://www.rulequest.com/cubist-info.html>`_
- :class:`uncoverml.cubist.Cubist`

**Additional parameters:**

- ``target_transform``

Multi Cubist
~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: multicubist

**Documentation and parameters:**

- `Rule-Quest <https://www.rulequest.com/cubist-info.html>`_
- :class:`uncoverml.cubist.MultiCubist`

**Additional parameters:**

- ``target_transform``
- ``trees``: number of Cubist submodels to train
- ``parallel``: boolean, whether to train this model in parallel

  - If ``parallel`` is True, this model can be trained using multiple processors.
    See :ref:`Multiprocessing and Partitioning`.

K Nearest Neighbour
~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: nnr

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`_
- :class:`uncoverml.models.CustomKNeighborsRegressor`

**Additional parameters:**

- ``target_transform``

Bootstrapped SVR
~~~~~~~~~~~~~~~~

Allows probabilistic predictions for SVR by taking statistics from an
ensemble of SVR models predicting on bootstrapped (resampled) data.

**Config:**

.. code:: yaml

    algorithm: bootstrapsvr

**Documentation and parameters:**

- See :ref:`Support Vector Regression`
- :class:`uncoverml.models.BootstrappedSVR`

**Additional parameters:**

- ``target_transform``
- ``n_models``: int, number of models to train (i.e. number of times to resample data)
- ``parallel``: boolean, whether to train this model in parallel

  - If ``parallel`` is True, this model can be trained using multiple processors.
    See :ref:`Multiprocessing and Partitioning`.

.. _optimisable-models:

Random Forest (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a subclass of :ref:`Random Forest` structured to be compatible 
with optimisation.

**Config:**

.. code:: yaml

    algorithm: transformedrandomforest

**Documentation and parameters:**

- See :ref:`Random Forest`
- :class:`uncoverml.optimise.models.TransformedForestRegressor`

**Additional parameters:**

- ``target_transform``

Gradient Boost (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: gradientboost

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_
- :class:`uncoverml.optimise.models.TransformedGradientBoost`

**Additional parameters:**

- ``target_transform``

Gaussian Process Regressor (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: transformedgp

**Documentation and parameters:**

- `Scitkit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_
- :class:`uncoverml.optimise.models.TransformedGPRegressor`

**Additional parameters:**

- ``target_transform``

Stochastic Gradient Descent Regressor (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: sgdregressor

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html>`_
- :class:`uncoverml.optimise.models.TransformedSGDRegressor`

**Additional parameters:**

- ``target_transform``

Support Vector Regression (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Duplicate of :ref:`Support Vector Regression` structured to be 
compatible with optimisation.

**Config:**

.. code:: yaml

    algorithm: transformedsvr

**Documentation and parameters:**

- See :ref:`Support Vector Regression`
- :class:`uncoverml.optimise.models.TransformedSVR`

**Additional parameters:**

- ``target_transform``

Ordinary Least Squares Regression (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of base scikit-learn `LinearRegression <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_.

**Config:**

.. code:: yaml

    algorithm: ols

**Documentation and parameters:**

- :class:`uncoverml.optimise.models.TransformedOLS`

**Additional parameters:**

- ``target_transform``

Elastic Net (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: elasticnet

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html>`_
- :class:`uncoverml.optimise.models.TransformedElasticNet`

**Additional parameters:**

- ``target_transform``

Huber (Optimisable)
~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: huber

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html>`_
- :class:`uncoverml.optimise.models.Huber`

**Additional parameters:**

- ``target_transform``

XGBoost (Optimisable)
~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: xgboost

**Documentation and parameters:**

- `XGBoost <https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn>`_
- :class:`uncoverml.optimise.models.XGBoost`

**Additional parameters:**

- ``target_transform``

Interpolators
-------------

Linear ND Interpolator
~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: linear

**Documentation and parameters:**

- `Scipy <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.LinearNDInterpolator.html>`_
- :class:`uncoverml.interpolate.SKLearnLinearNDInterpolator`

Nearest ND Interpolator
~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: nn
    
**Documentation and parameters:**

- `Scipy <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.NearestNDInterpolator.html>`_
- :class:`uncoverml.interpolate.SKLearnNearestNDInterpolator`

RBF Interpolator
~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: rbf

**Documentation and paramters:**

- `Scipy <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.Rbf.html>`_
- :class:`uncoverml.interpolate.SKLearnRbf`

CT Interpolator
~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: cubic2d

**Documentation and parameters:**

- `Scipy <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.CloughTocher2DInterpolator.html>`_
- :class:`uncoverml.interpolate.SKLearnCT`

Classifiers
-----------

Label encoding is performed implictly on UncoverML classifiers.

Logistic Classifier
~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: logistic

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
- :class:`uncoverml.models.LogisticClassifier`

Logistic RBF Classifier 
~~~~~~~~~~~~~~~~~~~~~~~

Kernelized version of :ref:`LogisticClassifier`

**Config:**

.. code:: yaml

    algorithm: logisticrbf

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
- :class:`uncoverml.models.LogisticRBF`
- :meth:`uncoverml.models.kernelize`

Random Forest Classifier
~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: forestclassifier

**Documentation and parameters:**

- `Scitkit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
- :class:`uncoverml.models.RandomForestClassifier`

Suport Vector Classifier
~~~~~~~~~~~~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: svc

**Documentation and parameters:**

- `Scitkit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
- :class:`uncoverml.models.SupportVectorClassifier`

Boosted Trees
~~~~~~~~~~~~~

**Config:**

.. code:: yaml

    algorithm: boostedtrees

**Documentation and parameters:**

- `Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_
- :class:`uncoverml.models.GradBoostedTrees`

