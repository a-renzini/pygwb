#!/usr/bin/env python

import os
import subprocess

from setuptools import find_packages, setup


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

    _version_file = "src/.version"
    if not os.path.isfile(_version_file):
        with open(_version_file, "w+") as f:
            f.write(f"{version}: {git_status}")

    return _version_file


def get_long_description():
    """ Finds the README and reads in the description """
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
    name="stochastic_lite",
    description="Lighweight python stochastic GWB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git.ligo.org/andrew.matas/stochastic_lite",
    author="Andrew Matas",
    author_email="andrew.matas@ligo.org",
    license="MIT",
    version=VERSION,
    packages=["stochastic_lite"],
    # packages=find_packages(exclude=["test", "venv", "tutorials", "src", "docs"])
    # + ["stochastic_lite"],
    package_dir={"stochastic_lite": "src"},
    package_data={"stochastic_lite": [".version"]},
    install_requires=["numpy", "matplotlib", "scipy", "bilby"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
