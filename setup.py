"""Install script for setuptools."""
from __future__ import absolute_import, division, print_function

from setuptools import setup

setup(
    name="sox",
    version="0.0.1",
    description="sox: operations to not forget, like your socks.",
    author="Max Smith",
    packages=["sox"],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.7.5",
    ],
)
