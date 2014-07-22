==============================================
evilometer: Rates arbitrary names based on a pre-rated list of names on some characteristic (ie "evilness")
==============================================
  * Copyright : July-2014, (c) AGPLv3 or later
  * Developed : by ankostis@gmail.com & Nikifors Zacharoff
  * License   : GNU Affero General Public License v3 or later (AGPLv3+)


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

Fuefit accepts as input 2 vectors of names, the pre-rated "training" set of and
the set of names to be rated, like that::

    >> import evilometer

    >> train_names = {'trendy': 1, 'good':2, 'better':2, 'talon':-5, 'bad_ass':-10}
    >> asked_names = {'kolon':1, 'trekking':2, 'trepper':-10}

    >> name_scores = evilometer(train_names, asked_names)
    >> print_scored_names_sorted(name_scores)


Contributors
==============

* Nikifors Zacharoff & Nikias Fontaras:  Original idea of the game




.. rubric:: Footnotes

.. [inv_index] http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html

