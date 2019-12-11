Configuring UncoverML
=====================

UncoverML workflows are controlled by a YAML configuration file.
This section provides some examples and explanations of different 
workflows and possible parameters.

For a reference to all possible config parameters, view the module
documentation: :mod:`uncoverml.config`

Example - Random Forest
-----------------------

The first example is a configuration file for a Random Forest model.
This file can be found in the repository under `tests/test_data/sirsam/random_forest/sirsam_Na_random_forest.yaml`.

.. code:: yaml
 
  learning:
    algorithm: multirandomforest
    arguments:
      n_estimators: 10
      target_transform: log
      forests: 20

The 'learning' block specifies the algorithm or model to train. 'algorithm'
is the name of the algorithm. 'arguments' specifies a dictionary of
keyword arguments specific to that model. For reference about what
arguments are applicable, refer the documentation for the specific model.

.. code:: yaml

  features:
    - type: ordinal
      files:
        - directory: $UNCOVERML_SRC/tests/test_data/sirsam/covariates/
      transforms:
        - centre
        - standardise
      imputation: mean

The 'features' block contains the features (AKA covariates) to be used for training and prediction.
UncoverML supports ordinal and categorical data, but they must be provided in separate
files and the 'type' must be provided (if no type is provided, UncoverML will assume the
data is ordinal). To provide a separate feature set for 'categorical data', you can
add another entry under the 'features' block:

.. code:: yaml

  features:
    - type: ordinal
      # ...files, transforms, imputation
    - type: categorical
      files: 
        - directory: path/to/categorical/data
      # ...transforms, imputation

The 'files' field is where paths to covariate data is specified. In this case, a directory
is provided, so every file in that directory will be treated as part of the feature set.
Individual paths can be set or a text file with a list of paths. Covariate files must be in
geotiff format, and each file must have the same dimensions and projection.

'transforms' specifies scaling that will be applied to that set of features. Multiple can be 
provided. Available transforms are:

- 'centre'
- 'log'
- 'sqrt'
- 'standardise'
- 'whiten'
- 'onehot'
- 'randomhot'

'imputation' is the imputation (filling of no data values) method. Only one can be provided for
each feature set. Available methods are:

- 'none'
- 'mean'
- 'gaus'
- 'nn' (nearest neighbour)

.. code:: yaml

  targets:
    file: $UNCOVERML_SRC/tests/test_data/sirsam/targets/geochem_sites_log.shp
    property: Na_log

The 'targets' block contains details for the training data. 'file' is the path to the shapefile
containing the targets. 'property' is the name of the field in the shapefile to train on. UncoverML
works by intersecting patches of the covariate data with corresponding target locations.

.. code:: yaml

  validation:
    feature_rank: True
    k-fold:
      parallel: True
      folds: 5
      random_seed: 1

The 'validation' block is optional and contains parameters for performing k-fold cross validation,
feature ranking and permutation importance. In this config file, 'feature_ranking' has been 
enabled and 'k-fold' has also been enabled. 'k-fold' cross validation has some parameters to set.
'parellel' will allow the cross validation to take advantage of multiprocessing: if you are running
UncoverML with MPI and more than one processor, setting this to 'True' will accelerate the 
validation. 'folds' is the number of folds to split the training data into. 'random_seed' is the 
seed provided to numpy for getting random permutations of data to split into folds. The permutation
is pseudorandom, i.e. using the same seed will provide deterministic results.

.. code:: yaml

  prediction:
    quantiles: 0.95
    outbands: 10

  output:
    directory: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out
    model: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/sirsam_Na_randomforest.model
    plot_feature_ranks: True
    plot_intersection: True
    plot_real_vs_pred: True
    plot_correlation: True
    plot_target_scaling: True

  pickling:
    covariates: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/features.pk
    targets: $UNCOVERML_SRC/tests/test_data/sirsam/random_forest/out/targets.pk

