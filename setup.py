#!/usr/bin/env python
import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

readme = open('README.rst').read()
doclink = """
Documentation
-------------

The full documentation is at http://GeoscienceAustralia.github.io/uncover-ml/.
"""
history = open('HISTORY.rst').read().replace('.. :changelog:', '')

def build_cubist():
    try:
        from uncoverml import cubist_config
        out = subprocess.run([cubist_config.invocation, '-h'])
        print(out)
    except:
        out = subprocess.run(['./cubist/makecubist', '.'])

    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
    with open('uncoverml/git_hash.py', 'w') as f:
        f.write("git_hash = '{}'".format(git_hash))

class build_py(_build_py):
    """
    Override build_py to ensure cubist is installed as part of 'pip install'
    and when using 'python setup.py install'.
    """
    def run(self):
        build_cubist()
        _build_py.run(self)

class develop(_develop):
    """
    Override develop to ensure cubist is installed as part of 'pip -e install'. 
    """
    def run(self):
        build_cubist()
        _develop.run(self)

setup(
    cmdclass={
        'build_py': build_py,
        'develop': develop
    },
    name='uncover-ml',
    version='0.2.0',
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
            'gridsearch = uncoverml.scripts.gridsearch:cli'
        ]
    },
    setup_requires=[
        'numpy==1.17.2',
        'Cython==0.29.13',
    ],
    install_requires=[
        'tables==3.5.2',
        'rasterio==1.1.0',
        'affine==2.3.0',
        'pyshp==1.2.3',
        'click==7.0',
        'revrand==1.0.0',
        'mpi4py==3.0.2',
        'scipy==1.3.1',
        'scikit-learn==0.21.3',
        'scikit-image==0.15.0',
        'PyYAML==5.1.2',
        'pandas==0.25.1',
        'ppretty==1.3',
        'matplotlib==3.1.1',
        'PyKrige==1.4.1',
        'xgboost==0.90',
        'eli5==0.10.1',
    ],
    extras_require={
        'kmz': [
            'simplekml',
            'pillow'
        ],
        'dev': [
            'sphinx==2.2.0',
            'ghp-import==0.5.5',
            'sphinxcontrib-programoutput==0.15',
            'pytest==5.2.1',
            'pytest-cov==2.8.1',
            'tox==3.2.1',
            'setuptools==41.4.0',
            'wheel==0.33.6'
        ] 
    },
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='uncover-ml',
    classifiers=[
        'Development Status :: 4 - Beta',
        "Operating System :: POSIX",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
