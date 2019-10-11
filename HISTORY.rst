.. :changelog:

History
=======
Unreleased (2019-10-xx)
-----------------
Added
+++++
- Python 3.7 support.
- Metadata profiler.
- Codecov integration.
- 'dist' and 'release' steps to Makefile.
- Doc deployment from CircleCI on master commits.

Fixed
+++++
- Broken tests due to usage of fixtures and new verison of PyTest
- Override of MPI pickling.
- Cubist install no longer prevents 'install_requires' from running as part of 'setup.py install'.

Changed
+++++++
- Updated depdency versions.

Removed
+++++++ 
- Support for Python versions below 3.7.

0.1.0 (2016-05-01)
------------------
- First release on PyPI.
