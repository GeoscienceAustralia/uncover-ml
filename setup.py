#!/usr/bin/env python
import os
import sys
import subprocess
from setuptools import setup

# If testing in python 2, use subprocess32 instead of built in subprocess
if os.name == 'posix' and sys.version_info[0] < 3:
    extra_test_deps = ['subprocess32']
else:
    extra_test_deps = []

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://GeoscienceAustralia.github.io/uncover-ml/.
"""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

from setuptools.command.install import install
from setuptools.command.develop import develop


def build_cubist():
    try:
        from uncoverml import cubist_config
        out = subprocess.run([cubist_config.invocation, '-h'])
        print(out)
    except:
        out = subprocess.run(['./cubist/makecubist', '.'])
    git_hash = subprocess.check_output(['git', 'rev-parse',
                                        'HEAD']).decode().strip()
    with open('uncoverml/git_hash.py', 'w') as f:
        f.write("git_hash = '{}'".format(git_hash))


class CustomInstall(install):
    def run(self):
        build_cubist()
        install.run(self)


class CustomDevelop(develop):
    def run(self):
        build_cubist()
        develop.run(self)


setup(
    cmdclass={'install': CustomInstall,
              'develop': CustomDevelop},
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
              'uncoverml.optimise'],
    package_dir={'uncover-ml': 'uncoverml'},
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'uncoverml = uncoverml.scripts.uncoverml:cli',
            'gammasensor = uncoverml.scripts.gammasensor:cli',
            'tiff2kmz = uncoverml.scripts.tiff2kmz:main',
            'subsampletargets = uncoverml.scripts.subsampletargets:cli',
            'gridsearch = uncoverml.scripts.gridsearch:cli',
            'resample = uncoverml.scripts.resample_cli:cli',
            'shape = uncoverml.scripts.shapes_cli:cli',
        ]
    },
    setup_requires=[
        'numpy==1.26.4',
        'cython==3.0.9',
    ],
    install_requires=[
        'numpy==1.26.4',
        'cython==3.0.9',
        'pycontracts==1.8.12',
        'tables==3.9.2',
        'rasterio==1.3.9',
        'catboost==1.2.3',
        'affine==2.4.0',
        'pyshp==2.3.1',
        'click==8.1.7',
        'revrand==1.0.0',
        'mpi4py==3.1.5',
        'scipy==1.12.0',
        'scikit-learn==1.1',
        'scikit-image==0.22.0',
        'scikit-optimize==0.10.1',
        'wheel==0.43.0',
        'PyYAML==6.0.1',
        'pandas==2.2.1',
        'matplotlib==3.8.3',
        'PyKrige==1.7.1',
        'xgboost==2.0.3',
        'setuptools==69.2.0',
        'eli5==0.13.0',
        'networkx==3.2.1',
        'geopandas==0.14.3',
        'hyperopt==0.2.7',
        'Pillow==10.2.0',
        'PyWavelets==1.5.0',
        'imageio==2.34.0',
        'colorama==0.4.6',
        'shap==0.45.0',
        'boto3==1.34.67',
        'seaborn==0.13.2',
        'requests==2.31.0',
        'vecstack==0.4.0',
        'mlens==0.2.3'
    ],
    extras_require={
        'kmz': [
            'simplekml',
        ],
        'dev': [
                   'sphinx',
                   'ghp-import',
                   'sphinxcontrib-programoutput',
                   'pytest>=5.2.1',
                   'pytest-cov',
                   'coverage',
                   'codecov',
                   'tox==3.24.5',
               ] + extra_test_deps
    },
    license="Apache Softwa  re License 2.0",
    zip_safe=False,
    keywords='uncover-ml',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
