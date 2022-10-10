#!/usr/bin/env python

import os

from setuptools import setup


def get_long_description():
    """Finds the README and reads in the description"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, "README.md")) as f:
        long_description = f.read()
    return long_description


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


long_description = get_long_description()

setup(
    name="pygwb",
    description="Lighweight python stochastic GWB analysis pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/pygwb/pygwb",
    author="Arianna Renzini, Sylvia Biscoveanu, Shivaraj Khandasamy, Kamiel Janssens, Max Lalleman, Katarina Martinovic, Andrew Matas, Patrick Meyers, Alba Romero, Colm Talbot, Leo Tsukada, Kevin Turbang",
    author_email="arianna.renzini@ligo.org",
    license="MIT",
    packages=["pygwb"],
    package_dir={"pygwb": "pygwb"},
    scripts=['pygwb_pipe/pygwb_pipe','pygwb_pipe/pygwb_combine', 'pygwb_pipe/pygwb_stats', 'pygwb_pipe/DAG/pygwb_dag'],
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy==1.8.0",
        "bilby",
        "gwpy",
        "astropy",
        "lalsuite",
        "loguru",
        "json5",
        "jinja2==3.0.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

