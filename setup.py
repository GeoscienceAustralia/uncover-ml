#!/usr/bin/env python
import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.develop import develop as _develop

readme = open('README.rst').read()


def parse_requirements(filename):
    """Gets list of requirements from a requirements file.
    """
    with open(filename) as f:
        return f.read().splitlines()


def build_cubist():
    out = subprocess.run(['./cubist/makecubist', '.'])
    print(out)

class build_py(_build_py):
    """ Override to ensure cubist is installed.

    Ensures cubist installs when running 'pip install' or 'python
    setup.py install.'
    """
    def run(self):
        build_cubist()
        _build_py.run(self)


class develop(_develop):
    """ Override to ensure cubist is installed.

    Ensures cubist installs when running 'pip install .[dev]'.
    """
    def run(self):
        build_cubist()
        _develop.run(self)

def git_desc():
    desc = ['git', 'describe', '--tags', '--abbrev=0']
    return subprocess.check_output(desc).decode().strip()

setup(
    cmdclass={
        'build_py': build_py,
        'develop': develop
    },
    name='uncover-ml',
    version=git_desc(),
    description='Machine learning tools for the Geoscience Australia uncover '
                'project',
    long_description=readme,
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
            'covdiag = uncoverml.scripts.covdiag:cli'
        ]
    },
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        'kmz': parse_requirements('requirements-kmz.txt'),
        'dev': parse_requirements('requirements-dev.txt'),
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
        "Programming Language :: Python :: 3.6",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
