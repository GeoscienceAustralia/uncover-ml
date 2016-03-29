#!/usr/bin/env python

from setuptools import setup
from setuptools.command.test import test as TestCommand


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

The full documentation is at http://nicta.github.io/uncover-ml/."""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

setup(
    name='uncover-ml',
    version='0.1.0',
    description='Machine learning tools for the Geoscience Australia uncover '
                'project',
    long_description=readme + '\n\n' + doclink + '\n\n' + history,
    author='NICTA Spatial Inference Systems Team',
    author_email='daniel.steinberg@nicta.com.au',
    url='https://github.com/NICTA/uncover-ml',
    packages=['uncoverml', 'uncoverml.scripts'],
    package_dir={'uncover-ml': 'uncoverml'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'pointspec = uncoverml.scripts.pointspec:main',
            'cvindexer = uncoverml.scripts.cvindexer:main',
            'extractfeats = uncoverml.scripts.extractfeats:main',
            'composefeats = uncoverml.scripts.composefeats:main',
            'uncoverml-worker = uncoverml.scripts.worker:main',
            'maketargets = uncoverml.scripts.maketargets:main'
        ]
    },
    install_requires=[
        'wheel',
        'scipy',
        'numpy',
        'tables',
        'affine',
        'rasterio',
        'pyshp',
        'ipyparallel',
        'click'
    ],
    extras_require={
        'demos': [
            'matplotlib'
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
    ],
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='uncover-ml',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
