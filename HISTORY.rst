.. :changelog:

History
=======
Unreleased (2019-12 - ongoing)
-----------------------
Added
+++++
- Install instructions for VDI
- "extents" parameter for config. Providing bounding box will crop targets and covariates.
- "shiftmap" command for generating covariate shift map. Can work with randomly generated targets
  or user can provide their own shapefile via the 'shiftmap' property of the config 'target' block.
- "covdiag" command for 'diagnosing' covariate files. Similar to gdalinfo but with room for 
  additions if required.
- "targetsearch" comamnd which finds targets similar to prediction data outside of prediction area.
- More demonstration configs.
- Reimplemented target sampling/binning code (originally by Sudipta).
- Bootstrap SVR model.
- User defined metrics for optimisation.
- Addition of writing aribtrary fields to rawcovariates.csv.
- Sample weighting for certain models
- Doc overhaul

Changed
+++++++
- Tweaked plots.
- 'pbs' directory is now 'scripts'.
- Moved all CLI commands to be under the 'uncoverml' command.
- Cleaned up project structure - remove outdated 'pbs' directory and scripts that no longer apply.
- Training data is now shared via MPI shared memory during the learn step.
- Models are now exported with their TransformSet objects, so the transform statistics can be
  reused for prediction. Previsouly the entire Config object was pickled.

Fixed
+++++
- MRF results being determined by number of processors used. Changed to provide each random forest 
  with its own RNG seed.
- Crash when plotting feature ranking if using a model without 'MLL' metric.

0.3.1 (2019-11-28)
------------------
Added
+++++
- Ability to specify number of processors to run tests with using `pytest --num_procs N tests/`
- Ability to specify number of partitions to run tests with using `pytest --num_parts N tests/`

Fixed
+++++
- Random forest caching with multirandomforest 
- Random seeds causing different results for multirandomforest when using different number of processors.

0.3.0 (2019-11-25)
------------------
Added
+++++
- Diagnostics module for plotting various metrics.
- Diagnostics notebook for viewing plots.
- Target coordinates are now automatically reprojected if a \*.prj file is provided with the input 
  shapefile.

Changed
+++++++
- Optional output parameters in config, e.g. plots, are now boolean. User no longer provides a
  path, instead they are placed in output directory.

Removed
+++++++
- Old plotting code.
- HDF5 crossval results file.

0.2.1 (2019-11-07)
-----------------------
Added
+++++
- Temporary workaround for 'get_image_spec', at least one covariate file is now required
  even if using pickled data. 

Fixed
+++++
- Prediction thumbnails

0.2.0 (2019-11-05)
------------------
Added
+++++
- Python 3.6 and 3.7 support.
- Metadata profiler: a 'metadata.txt' file will be generated as part of predict output that
  will allow reproducability of results.
- Codecov integration.
- 'dist' and 'release' steps to Makefile.
- Doc deployment from CircleCI on master commits.
- Config module documentation.
- Script tests and test data.
- Can now provide env vars in YAML config in the form '$ENV_VAR'.

Fixed
+++++
- Broken tests due to usage of fixtures and new verison of PyTest
- Override of MPI pickling.
- Cubist install no longer prevents 'install_requires' from running as part of 'setup.py install'.
- Output directories: outputs no longer dumped in working directory. Optional files need to have a
  path specified. Compulsory files that have no path specified will default to 'output' directory.
- Config object no longer pickled with model file. This was causing issues with tests and also
  prevents making tweaks (e.g. selecting outbands) without having to rerun the 'learn' command
  and retrain the model. *You now need to specify a path under 'model' in the 'output' block of
  the config*. This is where the model will be saved when running 'learn' and loaded from running
  'predict'.
- Multirandomforest and multicubist 'temp' files. Training these models generates files which
  are required for making predictions using the model. These were previously dumped in working
  directory in a 'results' folder. These files aren't temporary and need to passed with the model.
  To make models portable these are longer saved as files but are stored in the model object 
  itself.

Changed
+++++++
- Updated depdency versions.
- Package version is now dervied from 'git describe' command. This is also written to metadata
  file when making predictions.
- Config:

  - Raise errors when required parameters are missing.
  - 'learning' block no longer required when clustering and vice versa.
  - Removed pickling of training data
  - Separate 'pickling' block for pickling data
  - Separate non-pickle outputs from pickling block into output block
  - No longer need to provide 'features' or 'target' blocks when loading from pickled data.
  - Renamed 'preprocessing' to 'final_transform' to better reflect its purpose.
  - Crossval parallel has to be set as 'parallel: True/False' under 'k-fold' block
  - 'optimisation_output' no longer required, writes results to output directory.
  - 'algorithm' no longer required for optimistion. Gets this from 'learning' algorithm.

Removed
+++++++ 
- Support for Python versions below 3.6.

0.1.0 (2019-09-22)
------------------
- Start of versioning
