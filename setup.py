#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright 2014 ankostis@gmail.com
#
# This file is part of evilometer.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

''''Rates arbitrary names based on a pre-rated list of names on some characteristic (ie "evilness")

Overview
========
Given a pre-rated list of names on some characteristic,
it decomposes them using n_grams and applies information retrieval rating[inv_index]_
to estimate the rating of any other name on that characteristic.

Install:
========

To install it, assuming you have download the sources,
do the usual::

    python setup.py install

Or get it directly from the PIP repository::

    pip3 install evilometer


Tested with Python 3.4.


Usage:
======

Fuefit accepts as input 2 vectors of names, the "training" set and the set of names to be rated.
A usage example::

    >> import evilometer

    >> train_names = {'trendy': 1, 'good':2, 'better':2, 'talon':-5, 'bad_ass':-10}
    >> asked_names = {'kolon':1, 'trekking':2, 'trepper':-10}
    >> name_scores = evilometer(train_names, asked_names)
    >> print_scored_names_sorted(name_scores)


@author: ankostis@gmail.com, Apr-2014, (c) AGPLv3 or later

.. rubric:: Footnotes

.. [inv_index] http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html

'''

#from setuptools import setup
from cx_Freeze import setup, Executable
import os

projname = 'evilometer'
mydir = os.path.dirname(__file__)

## Version-trick to have version-info in a single place,
## taken from: http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
##
def readversioninfo(fname):
    fglobals = {'__version_info__':('x', 'x', 'x')} # In case reading the version fails.
    exec(open(os.path.join(mydir, fname)).read(), fglobals)  # To read __version_info__
    return fglobals['__version_info__']

# Trick to use README file as long_description.
#  It's nice, because now 1) we have a top level README file and
#  2) it's easier to type in the README file than to put a raw string in below ...
def readtxtfile(fname):
    with open(os.path.join(mydir, fname)) as fd:
        return fd.read()

_myverstr = '.'.join(str(s) for s in readversioninfo('_version.py'))
setup(
    name = projname,
#    packages = [projname],
#     package_data= {'projname': ['data/*.csv']},
    py_modules = ['evilometer'],
#    test_suite="fuefit.test", #TODO: check setup.py testsuit indeed works.
    version = _myverstr,
    description = __doc__.strip().split("\n")[0],
    author = "ankostis",
    author_email = "ankostis@gmail.com",
    url = "https://github.com/ankostis/%s" % projname,
    license = "GNU Affero General Public License v3 or later (AGPLv3+)",
    keywords = ['text-rating', 'text-processing', 'natural-language', 'mind-game', 'fun'],
    classifiers = [
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Artistic Software",
        "Topic :: Games/Entertainmen",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Indexing",
        "Topic :: Text Processing :: Linguistic",
        "",
    ],
    long_description = __doc__,
    install_requires = [
        'numpy',
        'pandas',
        'xlrd',
    ],
    test_requires = [
    ],
    options = {
        'build_exe': {
            'include_msvcr': True,
            'compressed': True,
            'include_in_shared_zip': True,
        }, 'bdist_msi': {
            'add_to_path': False,
        },
    },
    executables = [Executable("evilometer.py")]
)
