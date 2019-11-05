.. :changelog:

History
=======
0.2.0 (2019-10-xx)
-----------------
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

Removed
+++++++ 
- Support for Python versions below 3.6.

0.1.0 (2019-09-22)
------------------
- Start of versioning
