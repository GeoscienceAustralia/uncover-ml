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
        ]
    },
    setup_requires=[
        'numpy==1.17.2',
        'Cython==0.29.14',
    ],
    install_requires=[
        'pycontracts == 1.7.9',
        'tables >= 3.2.2',
        'rasterio == 1.2.10',
        'catboost == 1.0.3',
        'affine >= 2.2.1',
        'pyshp == 2.1.0',
        'click >= 6.6',
        'revrand >= 0.9.10',
        'mpi4py == 3.0.3',
        'scipy==1.7.0',
        'scikit-learn ~= 0.22.2',
        'scikit-image == 0.19.1',
        'scikit-optimize == 0.8.1',
        'wheel >= 0.29.0',
        'PyYAML >= 3.11',
        'pandas == 1.2.5',
        'matplotlib>=1.5.1',
        'PyKrige==1.3.0',
        'xgboost==1.4.2',
        'setuptools>=30.0.0',
        'eli5>=0.8.2',
        'networkx==2.5.1',
        'geopandas==0.9.0',
        'hyperopt==0.2.5',
        'pillow==8.3.2'
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
        "Programming Language :: Python :: 3.9",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
)
