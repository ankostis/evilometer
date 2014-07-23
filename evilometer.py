'''
:authors: ankostis@gmail.com, zachani
:created: 21 Jul 2014

Rates text based on its n_gram similarity with a pre-rated list of names on some arbitrary characteristic (ie "evilness")

Given a pre-rated list of names on some characteristic,
it decomposes them using n_grams and applies information retrieval rating[inv_index]_
to estimate the rating of any other name on that characteristic.

Example
-------

    >> python -m evilometer     prerated.csv asked.txt
    >> python -m evilometer     prerated.csv asked1.txt asked2.txt
    >> python -m evilometer     -i prerated1.csv prerated2.csv  -\
                                o asked.txt asked2.txt

'''

from collections import defaultdict, Counter
import math
import re
import numpy as np
import pandas as pd


"""The maximum length opf the n_grams to consider (ie when 3 --> 'abc') """
max_n_grams = 3


def main(prgram_name):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[4])

    prerated_group = parser.add_mutually_exclusive_group(required=True)
    prerated_help = "The name of 2-column csv-file:    txt,score "
    prerated_group.add_argument('-i', nargs='*', metavar='CSVFILE', dest='prerated_fnames',
            help="%s \n%s"%(prerated_help, "STDIN implied if filename omitted."))
    prerated_group.add_argument('prerated_fname', nargs='?', metavar='CSVFILE', help=prerated_help)

    asked_group = parser.add_mutually_exclusive_group(required=True)
    asked_help = "The name of a file with the text-lines to be rated. "
    asked_group.add_argument('-o', nargs='*', metavar='TXTFILE', dest='asked_fnames',
            help="%s \n%s"%(asked_help, "STDIN implied if filename omitted."))
    asked_group.add_argument('asked_fname', nargs='?', metavar='TXTFILE', help=asked_help)
    opts = parser.parse_args()

    #print(opts)

    if opts.prerated_fname:
        opts.prerated_fnames = [opts.prerated_fname]
    prerated_names = read_prerated_csv(opts.prerated_fnames)

    if opts.asked_fname:
        opts.asked_fnames = [opts.asked_fname]
    asked_txt = read_txt_lines(opts.asked_fnames)

    #ngram_scores = generate_and_score_ngrams(prerated_names)
    #print_score_map_sorted(ngram_scores)

    import time
    start = time.clock()
    evil_names = train_evilometer_and_rate_txts(prerated_names, asked_txt)
    end = time.clock()
    ## 0.024
    ## 0.025

    from _version import __version_info__ as ver
    args = list(ver)
    args.append(end-start)
    print("ver{}.{}.{}: {:.4f}ms".format(*args))
    print_score_map_sorted(evil_names)



def read_prerated_csv(csv_fnames):
    csv_fnames = [locate_file(fname) for fname in csv_fnames]

    prerated_txt = {}
    for csv_fname in csv_fnames:
        prerated_txt.update(pd.Series.from_csv(csv_fname, header=None).to_dict())

    return prerated_txt

def read_txt_lines(txt_fnames):
    txt_fnames = [locate_file(fname) for fname in txt_fnames]

    txt = set()
    for fn in txt_fnames:
        with open(fn) as fd:
            txt.update(set(fd.readlines()))

    word_set = set()

    ## Ask scores also for splitted-words.
    #
    for ln in txt:
        word_set.update(clean_chars(ln).split(' '))
    txt.update(word_set)

    return txt



def train_evilometer_and_rate_txts(prerated_names, ask_txts):
    """
    Main entry point that builds the index from pre-rated names, and calcs score of input texts

    .. Seealso:
        generate_and_score_ngrams()
        evilometer_txt()
    """

    ## Train index from sample.
    ngram_scores = generate_and_score_ngrams(prerated_names)
    #print_score_map_sorted(ngram_scores)

    ## Calc score of input.
    txt_scores = {txt: evilometer_txt(ngram_scores, txt) for txt in ask_txts}

    return txt_scores





def evilometer_txt(ngram_scores, txt):
    """
    Rates a txt based on precalculated n_gram index of scores

    :param  ngram_scores: the precalculated n_gram index of scores, see :func:`generate_and_score_ngrams()`
    :param  str txt: the txt-line to rate

    """

    name_freqs = extract_ngrams(txt)
    ngram_len = np.fromiter(iter(name_freqs.values()), dtype=int).sum()

    ## Score = sum(ngram_score * ngram_freq)
    #
    scores = [freq * ngram_scores[ngram] for (ngram, freq) in name_freqs.items() if ngram in ngram_scores]
    score = np.asarray(scores).sum() / ngram_len

    return score


def generate_and_score_ngrams(prerated_names):
    """
    Constructs the n_gram index of scores from pre-rated names

    :param map prerated_names: a map of ``{name(str) --> score(number)}``
    :return: the n_gram index, a map of ``{n_gram(str) --> score(number)] }``

   First it constructs an inverse-index of ``{ngram_freqs --> [word_frequency, cummulative_score]}``
   and then "averages" the scores of names on each for n_gram as a single number using the
   following formula ::

       n_gram_score = cummulative_score * log(N / wf)

   where:

   * `cummulative_score`: is the product-sum of the frequency of each
             n_gram times the scores of the prerated_names it was found in
   * `N`: is the total number of prerated_names
   * `wf`: if the word_freq, that is, how many words contain this n_gram.

   .. Seealso::
       Inverse document frequency: http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html

       In doc-retrieving parlance, the ''word'' or ''prerated_names'' terms above are equivalent to the ''document''.
    """

    names_len = len(prerated_names)


    ## Initialize the inverse index.
    #
    def initial_scores():
        return [0, 0]                           ## ``[word_freq, cummulative_score]``
    ngram_counters = defaultdict(initial_scores)

    for (name, score) in prerated_names.items():
        ngram_freqs = extract_ngrams(name.lower())

        ##
        for (ng, ng_freq) in ngram_freqs.items():
            assert ng_freq > 0 and score != 0, (ng_freq, score)
            ng_counters = ngram_counters[ng]
            ng_counters[0] += 1              	## update ``word_freq``.
            ng_counters[1] += score * ng_freq   ## update ``cummulative_score``

    def rate_ngram(doc_freq, cummulative_score):
        """Produces the scores for each n_gram according to the formula in :func:`generate_and_score_ngrams()`"""
        return cummulative_score * math.log(names_len / doc_freq)

    ngram_scores = {ng: rate_ngram(*counters) for (ng, counters) in ngram_counters.items()}

    return ngram_scores


def extract_ngrams(name, n=max_n_grams):
    """
    Returns the frequencies of ngrams of a word after having appended the `^` and `$` chars at its start/end, respectively

    :param int n: the maximum length of the ngrams to extract, inclusive (ie: ``n=3 --> 'abc'``)
    :return: a map of ``{n_gram(str) --> freq(int)}``
    """

    name = clean_chars(name)

    ## Gather 1 ngrams them without ``^$`` bracket-chars.
    #
    ngrams = list(name)
    #ngrams = list()

    ## Gather 2+ ngrams after bracketingt words.
    #
    name = mark_word_boundaries(name)
    for n in range(2, n+1):
        ngrams.extend([name[i:i+n] for i in range(0, len(name) - n + 1)])

    ## Consolidate the The ngrams repetitions
    #    from the list above.
    #
    ngram_freqs = Counter()
    ngram_freqs.update(ngrams)

    ## Remove artifacts from word-baracketing.
    #
    assert not (set(['  ', '^', '$']) & ngram_freqs.keys())
    ngrams_to_remove = ('$ ^', '$ ', ' ^')
    for ng in ngrams_to_remove:
        ngram_freqs.pop(ng, None)

    return ngram_freqs


_mark_word_regex = re.compile(r'\b')
_mark_prefix_regex = re.compile(r'\b\^')
def mark_word_boundaries(txt):
    """ Makes: ``"some name " --> "^some$ ^name$ "`` """

    txt = _mark_word_regex.sub('^', txt)
    txt = _mark_prefix_regex.sub('$', txt)

    return txt


_nonword_char_regex = re.compile(r'(\W|\d)+')
_single_char_regex = re.compile(r'\b\w\b')
_deduplicate_chars_regex = re.compile(r'(\w)\1+')
_deduplicate_spaces_regex = re.compile(r' {2}')
def clean_chars(txt):
    """
    Simplify text before n_gram extraction by replacing non-ascii chars with space or turning them to lower
    """

    txt = _nonword_char_regex.sub(' ', txt).strip().lower()
    txt = _deduplicate_chars_regex.sub(r'\1', txt)
    txt = _single_char_regex.sub('', txt)
    txt = _deduplicate_spaces_regex.sub('', txt)

    return txt


def locate_file(fname):
    """
    Finds a file in current-dir or relative to this prog's dir

    :param str fname: the filename of a file in current-dir or prog's dir containing the lines to return
    :return: a path(str)
    """
    import os.path as path

    if not (path.isfile(fname) or path.isabs(fname)):
        fname = path.join(path.dirname(__file__), fname)

    return fname



def print_score_map_sorted(name_scores):
    """Prints a sorted map by values (sorting copied from: http://stackoverflow.com/questions/613183/python-sort-a-dictionary-by-value)"""
    import operator
    sorted_pairs = sorted(name_scores.items(), key=operator.itemgetter(1))
    for (key, val) in sorted_pairs:
        print("%s, %4.2f" % (key.strip(), val))



if __name__ == "__main__":
    import sys, os

    program_name    = os.path.basename(sys.argv[0])[:-3]
    main(program_name)
