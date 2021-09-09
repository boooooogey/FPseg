#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages

os.chdir(os.path.dirname(sys.argv[0]) or ".")

setup(
    name="FPseg",
    version="0.1",
    description="Fast Poisson Segmentation for genome",
    long_description=open("README.rst", "rt").read(),
    url="https://github.com/boooooogey/FPseg",
    author="Ali Tugrul Balci",
    author_email="",
    scripts=["bin/FPseg"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=[
        "./FPseg/l0approximator/buildgaussianl0.py:ffi",
        "./FPseg/l0approximator/buildpoissonl0.py:ffi",
        "./FPseg/l0approximator/buildexponentiall0.py:ffi"
    ],
)
