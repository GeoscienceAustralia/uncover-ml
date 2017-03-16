#!/usr/bin/env python
import os
import sys
from setuptools import setup
from setuptools.command.test import test as TestCommand

# If testing in python 2, use subprocess32 instead of built in subprocess
if os.name == 'posix' and sys.version_info[0] < 3:
    exta_test_deps = ['subprocess32']
else:
    exta_test_deps = []


class PyTest(TestCommand):

    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        super(PyTest, self).initialize_options()
        self.pytest_args = []

    def finalize_options(self):
        super(PyTest, self).finalize_options()
        self.test_suite = True
        self.test_args = []

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import pytest
        exit(pytest.main(self.pytest_args))


readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://GeoscienceAustralia.github.io/uncover-ml/."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='uncover-ml',
    version='0.1.0',
    description='Machine learning tools for the Geoscience Australia uncover '
                'project',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='Geoscience Australia Mineral Systems Group, NICTA Spatial '
           'Inference Systems Team',
    author_email='John.Wilford@ga.gov.au',
    url='https://github.com/GeoscienceAustralia/uncover-ml',
    packages=['uncoverml', 'uncoverml.scripts', 'uncoverml.transforms',
              'preprocessing', 'uncoverml.optimise'],
    package_dir={'uncover-ml': 'uncoverml'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'uncoverml = uncoverml.scripts.uncoverml:cli',
            'gammasensor = uncoverml.scripts.gammasensor:cli',
            'tiff2kmz = uncoverml.scripts.tiff2kmz:main',
            'subsampletargets = uncoverml.scripts.subsampletargets:cli',
            'geoinfo = preprocessing.geoinfo:cli',
            'resample = preprocessing.resample:cli',
            'rasteraverage = preprocessing.raster_average:cli',
            'gridsearch = uncoverml.scripts.gridsearch:cli'
        ]
    },
    install_requires=[
        'numpy >= 1.9.2',
        'pycontracts == 1.7.9',
        'tables >= 3.2.2',
        'rasterio == 0.36.0',
        'affine == 2.0.0.post1',
        'pyshp == 1.2.3',
        'click == 6.6',
        'revrand >= 0.9.10',
        'mpi4py == 2.0.0',
        'scipy >= 0.15.1',
        'scikit-learn == 0.18.1',
        'scikit-image >= 0.12.3',
        'wheel >= 0.29.0',
        'PyYAML >= 3.11',
        'GDAL >= 2.0.0',
        'pandas >= 0.18.1',
        'geopandas == 0.2.1',
        'matplotlib == 1.5.1',
        'PyKrige == 1.3.0',
    ],
    extras_require={
        'demos': [
            'matplotlib'
        ],
        'kmz': [
            'simplekml',
            'pillow'
        ],
        'dev': [
            'sphinx',
            'ghp-import',
            'sphinxcontrib-programoutput'
        ]
    },
    cmdclass={
        'test': PyTest
    },
    tests_require=[
        'pytest',
        'pytest-cov',
        'coverage',
        'codecov',
        'tox',
    ] + exta_test_deps,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='uncover-ml',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.4",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
