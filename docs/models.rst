Models
======

UncoverML has a variety of algorithms to choose from. The YAML block
shows the ``algorithm`` parameter to add to the ``learning`` block of the
config to use the model. 

Follow the links to view specific model documentation and available 
parameters.

*Additional parameters* covers parameters specific to 
UncoverML that can be added to the ``arguments`` section of the ``learning``
block.

Regressors
----------

Random Forest
~~~~~~~~~~~~~

.. code:: yaml

    algorithm: randomforest

Documentation and parameters:
`Scikit Learn: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_

Additional parameters:

- ``target_transform``

Multi Random Forest
~~~~~~~~~~~~~~~~~~~

An ensemble of Random Forest predictors.

.. code:: yaml

    algorithm: multirandomforest

Documentation and parameters:
See :ref:`Random Forest`

Additional parameters:

- ``target_transform``
- ``forests``: number of Random Forest submodels
- ``parallel``: boolean, whether to train this model in parallel

  - If ``parallel`` is True, this model can be trained using multiple processors.
    See :ref:`Multiprocessing and Partitioning`.

Bayes Regression/Standard Linear Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: bayesreg

Documentation and Parameters:
`NICTA Revrand <http://nicta.github.io/revrand/slm.html>`_
:class:`uncoverml.models.LinearReg`

Additional parameters:

- ``target_transform``

Stochastic Gradient Descent Bayes Regression/Standard Linear Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: sgdbayesreg

Documentation and parameters:
`NICA Revrand <http://nicta.github.io/revrand/glm.html>`_
:class:`uncoverml.models.SGDLinearReg`

Additional parameters:

- ``target_transform``

Approximate Gaussian Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of :ref:`Bayes Regression/Standard Linear Model`

.. code:: yaml

    algorithm: approxgp

Documentation and parameters:
`NICTA Revrand <http://nicta.github.io/revrand/slm.html>`_
:class:`uncoverml.models.ApproxGP`

Additional parameters:

- ``target_transform``

Stochastic Gradient Descent Approximate Gaussian Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subclass of :ref:`Stochastic Gradient Descent Bayes Regression/Standard Linear Model`

.. code:: yaml

    algorithm: sgdapproxgp

Documentation and parameters:
`NICTA Revrand <http://nicta.github.io/revrand/glm.html>`_
:class:`uncoverml.models.SGDApproxGP`

Additional parameters:

- ``target_transform``

Support Vector Regression
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: svr

Documentation and parameters:
`Scitkit-Learn <https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVR.html>`_

Additional parameters:

- ``target_transform``

Automatic Relevance Determination Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: ardregression

Documentation and parameters:
`Scitkit-Learn <https://scikit-learn.org/dev/modules/generated/sklearn.linear_model.ARDRegression.html>`_

Additional parameters:

- ``target_transform``

Decision Tree
~~~~~~~~~~~~~

.. code:: yaml

    algorithm: decisiontree

Documentation and parameters:
`Scitkit-Learn <https://scikit-learn.org/dev/modules/generated/sklearn.tree.DecisionTreeRegressor.html>`_

Additional parameters:

- ``target_transform``

Extra Tree
~~~~~~~~~~

.. code:: yaml

    algorithm: extratree

Documentation and parameters:
`Scitkit-Learn <http://scikit-learn.org/dev/modules/generated/sklearn.tree.ExtraTreeRegressor.html>`_

Additional parameters:

- ``target_transform``

Cubist
~~~~~~

.. code:: yaml

    algorithm: cubist

Documentation and parameters:
`Rule-Quest <https://www.rulequest.com/cubist-info.html>`_
:class:`uncoverml.cubst.Cubist`

Additional parameters:

- ``target_transform``

Multi Cubist
~~~~~~~~~~~~

.. code:: yaml

    algorithm: multicubist

Documentation and parameters:
`Rule-Quest <https://www.rulequest.com/cubist-info.html>`_
:class:`uncoverml.cubst.MultiCubist`

Additional parameters:

- ``target_transform``
- ``trees``: number of Cubist submodels to train
- ``parallel``: boolean, whether to train this model in parallel

  - If ``parallel`` is True, this model can be trained using multiple processors.
    See :ref:`Multiprocessing and Partitioning`.

K Nearest Neighbour
~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: nnr

Documentation and parameters:
`Scikit-Learn <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html`>_
:class:`uncoverml.models.CustomNeighborsRegressor`

Additional parameters:

- ``target_transform``

Random Forest (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: transformedrandomforest

Gradient Boost (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: gradientboost

Gaussian Process Regressor (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: transformedgp

Stochastic Gradient Descent Regressor (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: sgdregressor

Support Vector Regression (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: transformedsvr

Ordinary Least Squares Regression (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: ols

Elastic Net (Optimisable)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: elasticnet

Huber (Optimisable)
~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: huber

XGBoost (Optimisable)
~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: xgboost

Interpolators
-------------

Linear ND Interpolator
~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: linear

Nearest ND Interpolator
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: nn

RBF Interpolator
~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: rbf

CTI Interpolator
~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: cubic2d

Classifiers
-----------

Logistic Classififer
~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: logistic

Logistic RBF Classifier 
~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: logisticrbf

Random Forest Classifier
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: forestclassifier

Suport Vector Classifier
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: yaml

    algorithm: svc

Boosted Trees
~~~~~~~~~~~~~

.. code:: yaml

    algorithm: boostedtrees
