#!/usr/bin/env python

import os
import subprocess

from setuptools import setup


def write_version_file(version):
    """Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%h %ai"]
        ).decode("utf-8")
        git_diff = (
            subprocess.check_output(["git", "diff", "."])
            + subprocess.check_output(["git", "diff", "--cached", "."])
        ).decode("utf-8")
        if git_diff == "":
            git_status = "(CLEAN) " + git_log
        else:
            git_status = "(UNCLEAN) " + git_log
    except Exception as e:
        print(f"Unable to obtain git version information, exception: {e}")
        git_status = ""

    _version_file = "pygwb/.version"
    if not os.path.isfile(_version_file):
        with open(_version_file, "w+") as f:
            f.write(f"{version}: {git_status}")

    return _version_file


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


VERSION = "0.0.1"
version_file = write_version_file(VERSION)
long_description = get_long_description()

setup(
    name="pygwb",
    description="Lighweight python stochastic GWB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ligo.org/pygwb/pygwb",
    author="Arianna Renzini, Sylvia Biscoveanu, Shivaraj Khandasamy, Kamiel Janssens, Max Lalleman, Katarina Martinovic, Andrew Matas, Patrick Meyers, Alba Romero, Colm Talbot, Leo Tsukada, Kevin Turbang",
    author_email="arianna.renzini@ligo.org",
    license="MIT",
    version=VERSION,
    packages=["pygwb"],
    package_dir={"pygwb": "pygwb"},
    package_data={"pygwb": [".version"]},
    scripts=['pygwb_pipe/pygwb_pipe','pygwb_pipe/pygwb_combine', 'pygwb_pipe/pygwb_stats', 'pygwb_pipe/DAG/pygwb_dag'],
    install_requires=[
        "numpy==1.19.5",
        "matplotlib",
        "scipy==1.8.0",
        "bilby",
        "gwpy",
        "astropy==4.3.1",
        "lalsuite==7.3",
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

