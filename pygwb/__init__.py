"""pygwb"""

import os

__dir = "/".join((__file__.split("/")[:-1]))
with open(os.path.join(__dir, ".version")) as version_file:
    __version__ = version_file.read().strip()
