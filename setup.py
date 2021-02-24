"""
Setup script for obsplus
"""
import glob
import sys
from os.path import join, exists, isdir

try:  # not running python 3, will raise an error later on
    from pathlib import Path
except ImportError:
    pass

from setuptools import setup

# define/enforce python versions
python_version = (3, 6)  # tuple of major, minor version requirement
python_version_str = str(python_version[0]) + "." + str(python_version[1])

# produce an error message if the python version is less than required
if sys.version_info < python_version:
    msg = "dbscan1d only runs on python version >= %s" % python_version_str
    raise Exception(msg)

# get path references
here = Path(__file__).absolute().parent
version_file = here / "dbscan1d" / "version.py"

# --- get version
with version_file.open() as fi:
    for line in fi.readlines():
        if line.startswith("__version__"):
            __version__ = (
                line.replace('"', "").replace("'", "").split(" = ")[1].rstrip()
            )


# --- get readme
with open("README.md") as readme_file:
    readme = readme_file.read()


# --- get sub-packages
def find_packages(base_dir="."):
    """ setuptools.find_packages wasn't working so I rolled this """
    out = []
    for fi in glob.iglob(join(base_dir, "**", "*"), recursive=True):
        if isdir(fi) and exists(join(fi, "__init__.py")):
            out.append(fi)
    out.append(base_dir)
    return out


# --- requirements paths


def read_requirements(path):
    """ Read a requirements.txt file, return a list. """
    with Path(path).open("r") as fi:
        return fi.readlines()


package_req_path = here / "requirements.txt"
test_req_path = here / "tests" / "requirements.txt"

setup(
    name="dbscan1d",
    version=__version__,
    description="dbscan1d is a package for DBSCAN on 1D arrays",
    long_description=readme,
    author="Derrick Chambers",
    author_email="djachambeador@gmail.com",
    url="https://github.com/d-chambers/dbscan1d",
    packages=find_packages("dbscan1d"),
    package_dir={"dbscan1d": "dbscan1d"},
    include_package_data=True,
    license="GNU Lesser General Public License v3.0 or later (LGPLv3.0+)",
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
    ],
    test_suite="tests",
    install_requires=read_requirements(package_req_path),
    tests_require=read_requirements(test_req_path),
    setup_requires=["pytest-runner>=2.0"],
    python_requires=">=%s" % python_version_str,
    long_description_content_type="text/markdown",
)
