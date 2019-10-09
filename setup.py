#!/usr/bin/env python
import os
import sys
import subprocess
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop

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


class CustomInstall(install):
    def run(self):
        build_cubist()
        install.do_egg_install(self)


class CustomDevelop(develop):
    def run(self):
        build_cubist()
        develop.run(self)


setup(
    cmdclass={'install': CustomInstall,
              'develop': CustomDevelop},
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
        'numpy',
        'Cython==0.29.13',
    ],
    install_requires=[
        'pycontracts == 1.7.9',
        'tables >= 3.2.2',
        'rasterio >= 1.0.8',
        'affine >= 2.2.1',
        'pyshp == 1.2.3',
        'click >= 6.6',
        'revrand >= 0.9.10',
        'mpi4py == 3.0.2',
        'scipy >= 0.15.1',
        'scikit-learn >= 0.21.1',
        'scikit-image >= 0.12.3',
        'wheel >= 0.29.0',
        'PyYAML >= 3.11',
        'pandas == 0.25.1',
        'ppretty==1.3',
        'matplotlib >= 1.5.1',
        'PyKrige == 1.3.0',
        'xgboost >= 0.72.1',
        'setuptools >= 30.0.0',
        'eli5 >= 0.8.2',
    ],
    extras_require={
        'kmz': [
            'simplekml',
            'pillow'
        ],
        'dev': [

            'sphinx',
            'ghp-import',
            'sphinxcontrib-programoutput',
            'pytest == 5.2.1',
            'pytest-cov',
            'coverage',
            'codecov',
            'tox==3.2.1',
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
