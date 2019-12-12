Configuration
------------

UncoverML workflows are controlled by a YAML configuration file.
This section provides some examples and explanations of different 
workflows and possible parameters.

For a reference to all possible config parameters, view the module
documentation: :mod:`uncoverml.config`

Learning
~~~~~~~~

.. code:: yaml
 
  learning:
    algorithm: multirandomforest
    arguments:
      n_estimators: 10
      target_transform: log
      forests: 20

The ``learning`` block specifies the algorithm or model to train. ``algorithm``
is the name of the algorithm. ``arguments`` specifies a dictionary of
keyword arguments specific to that model. For reference about what
arguments are applicable, refer the documentation for the specific model.

Features
~~~~~~~~

.. code:: yaml

  features:
    - type: ordinal
      files:
        - directory: $UNCOVERML_SRC/tests/test_data/sirsam/covariates/
      transforms:
        - centre
        - standardise
      imputation: mean

The ``features`` block contains the features (AKA covariates) to be used for training and prediction.
UncoverML supports ordinal and categorical data, but they must be provided in separate
files and the ``type`` must be provided (if no type is provided, UncoverML will assume the
data is ordinal). To provide a separate feature set for ``categorical data``, you can
add another entry under the ``features`` block:

.. code:: yaml

  features:
    - type: ordinal
      # ...files, transforms, imputation
    - type: categorical
      files: 
        - directory: path/to/categorical/data
      # ...transforms, imputation

The ``files`` field is where paths to covariate data is specified. In this case, a ``directory``
is provided, so every file in that directory will be treated as part of the feature set.
Individual paths can be set or a text file with multiple paths separated by commas:

.. code:: yaml

    features:
      - type: ordinal
        files:
          - path: path/to/covariate1.tif
          - path: path/to/covariate2.tif
          - list: path/to/list_of_covariates.txt
  

Covariate files must be in geotiff format, and each file must have the same dimensions and 
projection.

``transforms`` specifies scaling that will be applied to that set of features. Multiple can be 
provided. Available transforms are:

- ``centre``
- ``log``
- ``sqrt``
- ``standardise``
- ``whiten``
- ``onehot``
- ``randomhot``

``imputation`` is the imputation (filling of no data values) method. Only one can be provided for
each feature set. Available methods are:

- ``none``
- ``mean``
- ``gaus``
- ``nn`` (nearest neighbour)

Targets
~~~~~~~

.. code:: yaml

  targets:
    file: $UNCOVERML_SRC/tests/test_data/sirsam/targets/geochem_sites_log.shp
    property: Na_log

The ``targets`` block contains details for the training data. ``file`` is the path to the shapefile
containing the targets. ``property`` is the name of the field in the shapefile to train on. UncoverML
works by intersecting patches of the covariate data with corresponding target locations.

Validation
~~~~~~~~~~

.. code:: yaml

  validation:
    feature_rank: True
    k-fold:
      parallel: True
      folds: 5
      random_seed: 1

The ``validation`` block is optional and contains parameters for performing k-fold cross validation,
feature ranking and permutation importance. In this config file, ``feature_ranking`` has been 
enabled and ``k-fold`` has also been enabled. ``k-fold`` cross validation has some parameters to set.
``parellel`` will allow the cross validation to take advantage of multiprocessing: if you are running
UncoverML with MPI and more than one processor, setting this to ``True`` will accelerate the 
validation. ``folds`` is the number of folds to split the training data into. ``random_seed`` is the 
seed provided to numpy for getting random permutations of data to split into folds. The permutation
is pseudorandom, i.e. using the same seed will provide deterministic results.

Prediction
~~~~~~~~~~

.. code:: yaml

  prediction:
    quantiles: 0.95
    outbands: 4

The ``prediction`` block configures the prediction output. ``quantiles`` refers to the prediction 
interval, e.g. '0.95' means that predicted values will fall within the lower and upper quantiles
95% of the time. ``outbands`` specifies the bands to output. Each band will be written as a separate
geotiff file. For classification, the available outbands is equivalent to the available classes.
For regression, the first outband is prediction and if the model provides them, the next are
variance, lower quantile and upper quantile. Some specific models provide further options - refer
the documentation for the specific model you are using. The ``outbands`` number is used as the RHS
of a slice, so providing '1' for a regression will output prediction (0) and variance (1). 

.. todo::
  
  'outbands' is currently a bit broken. It gets used a slice for the output bands, so giving
  some arbitrarily high number will you give you all bands. This will change in future and the
  user will provide explicit labels for the bands they want.
  
Output
~~~~~~

.. code:: yaml

  output:
    directory: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out
    model: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/sirsam_Na_randomforest.model
    plot_feature_ranks: True
    plot_intersection: True
    plot_real_vs_pred: True
    plot_correlation: True
    plot_target_scaling: True

The ``output`` block controls where outputs will be stored. ``directory`` is where all outputs from
learning, prediction and other commands will be stored. ``model`` is a special case, and specifies
where the '.model' file created from the learn step will be stored and also what model will be
used in the prediction step. If you want to predict based on a previously learned model, you
need to change the ``model`` field to the path of the model you are using.

There are also various flags for generating plots. If these are set to ``True``, then a plot will
be created. Some plots will only be created if certain steps have been run, e.g. ``plot_feature_ranks``
will only generate a plot if feature ranking is performed as part of validation. For more details,
view the section on diagnostics: :ref:`diagnostics`.

For a comprehensive list of the outputs each step of UncoverML generates, see the section on
outputs: :ref:`outputs`.

Pickling
~~~~~~~~

.. code:: yaml

  pickling:
    covariates: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/features.pk
    targets: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/targets.pk

The final block is for ``pickling``. During the learn step, covariates and targets are scaled and
intersected. Depending on the machine being used and the size of the data, this may take a 
non-trivial amount of time. In situations where you are tweaking parameters and re-running the 
learn step, pickling the intersected covariate and target data may save time. The ``covariates`` field
is the path to where the pickle file will be saved to and then read, and the ``targets`` file is the
same but for target data. If these are provied but do not exist, coviarates and targerts will be
scaled and intersected as normal then pickled to these files for future use. If provided and they
exist, intersection will be skipped and data will be loaded from these files instead.

