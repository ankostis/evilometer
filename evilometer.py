'''
Created on 21 Jul 2014

@author: zachani
'''

from collections import defaultdict, Counter
import math
import re
import numpy as np


prerated_names = {
    "Nathaniel": -18,
    "Earendil": -14,
    "Alustriel": -12,
    "Lothiriel": -12,
    "Nefelrith": -12,
    "Earwen": -10,
    "Anteimar": -9,
    "Elanor": -9,
    "Pippin": -9,
    "Kalidar": -8,
    "Alice": -8,
    "Gallandriel": -8,
    "Lavender": -8,
    "Melian": -8,
    "Aragorn": -7,
    "Firiel": -7,
    "Nora": -7,
    "Nikiforos": -6,
    "Endrahil": -6,
    "Esmeralda": -6,
    "Antonis": -5,
    "Bregalad": -5,
    "Finrod": -5,
    "Laura": -5,
    "Primula": -5,
    "Rufus": -5,
    "Nikias": -4,
    "Beruthiel": -4,
    "Celeborn": -4,
    "Finarfin": -4,
    "Holman": -4,
    "Mirabella": -4,
    "Robin": -4,
    "Konstantinos": -3,
    "pullman": -3,
    "Afgar": -3,
    "Cirdan": -3,
    "Durin": -3,
    "Hobson": -3,
    "Thorin": -3,
    "Stefanos": -2,
    "Zacharof": -2,
    "Ispra": -2,
    "Ivy": -2,
    "Peregrin": -2,
    "Strider": -2,
    "Theoden": -2,
    "spiderman": -1,
    "Andreth": -1,
    "Dina": -1,
    "rachmaninof": 1,
    "thanos": 3,
    "Hulk": 3,
    "Fengel": 3,
    "Hallatan": 3,
    "Yavanna": 3,
    "Roggoff": 4,
    "thanatos": 4,
    "dexter": 4,
    "Victor": 4,
    "Hildribrad": 4,
    "Lugdush": 4,
    "Pharazon": 4,
    "Wulfgar": 5,
    "Varese": 5,
    "Drogo": 5,
    "Druda": 5,
    "Orchaldor": 5,
    "Thorondor": 5,
    "Giorgos": 6,
    "Viconia": 6,
    "Damrod": 6,
    "Frar": 6,
    "Hatholdir": 6,
    "Ufthak": 6,
    "Vilna": 7,
    "Verbania": 7,
    "Eosfor": 7,
    "Odovocar": 7,
    "Zimrahin": 7,
    "Fastolph": 8,
    "Melkor": 8,
    "Poszaron": 8,
    "Thuringwelthin": 8,
    "Zimrathon": 8,
    "Manwendil": 9,
    "Sadoc": 9,
    "Madoc": 10,
    "Saradas": 11,
    "Sigismond": 11,
    "Vanda": 12,
    "Asfaloth": 12,
    "Isembold": 12,
    "Muzgash": 12,
    "Gilmikhad": 13,
    "Isumbras": 13,
    "Marmadoc": 13,
    "Radagast": 13,
    "Smaug": 13,
    "Inzilbeth": 14,
    "Gorbadoc": 15,
    "Carcharoth": 16,
    "Sakalthor": 16,
    "Morgoth": 17,
    "Gothmog": 18,
    "Gorgoroth": 20,
}

"""The maximum length opf the n_grams to consider (ie when 3 --> 'abc') """
max_n_grams = 3


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
            assert ng_freq>0 and score != 0, (ng_freq, score)
            ng_counters = ngram_counters[ng]
            ng_counters[0] += ng_freq        ## update ``word_freq``.
            ng_counters[1] += score * ng_freq   ## update ``cummulative_score``

    def rate_ngram(word_freq, cummulative_score):
        """Produces the scores for each n_gram according to the formula in :func:`generate_and_score_ngrams()`"""
        return cummulative_score * math.log10(names_len / word_freq)

    ngram_scores = {ng: rate_ngram(*counters) for (ng, counters) in ngram_counters.items()}

    return ngram_scores


def extract_ngrams(name, n=max_n_grams):
    """
    Returns the frequencies of ngrams of a word after having appended the `^` and `$` chars at its start/end, respectively

    :param int n: the maximum length of the ngrams to extract, inclusive (ie: ``n=3 --> 'abc'``)
    :return: a map of ``{n_gram(str) --> freq(int)}``
    """

    ## 1 ngrams:
    ##    gather them without ``^$`` bracket-chars.
    #
    name = clean_chars(name)
    ngrams = list(name)
    #ngrams = list()

    ## 2+ ngrams:
    ##     bracket words them before gathering.
    #
    name = mark_word_boundaries(name)
    for n in range(2, n+1):
        ngrams.extend([name[i:i+n] for i in range(0, len(name) - n + 1)])
    #
    ##  The `ngrams` list above contains repetitions.

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
def mark_word_boundaries(sentence):
    """ Makes: ``"some name " --> "^some$ ^name$ "`` """

    sentence = _mark_word_regex.sub('^', sentence)
    sentence = _mark_prefix_regex.sub('$', sentence)

    return sentence


_nonword_char_regex = re.compile('\W+')
def clean_chars(txt):
    """
    Simplify text before n_gram extraction by replacing non-ascii chars with space or turning them to lower
    """

    _nonword_char_regex.sub(' ', txt).lower().strip()

    return txt


def read_lines(fname):
    """
    Reads file-lines from current-dir or relative to this prog's dir

    :param str fname: the filename of a file in current-dir or prog's dir containing the lines to return
    :return: a list of cleaned lines(str)
    """
    import os.path as path

    if path.isfile(fname) or path.isabs(fname):
        nfname = fname
    else:
        nfname = path.join(path.dirname(__file__), inp_fname)

    with open(nfname) as fd:
        lines = fd.readlines()

    return lines



def print_score_map_sorted(name_scores):
    """Prints a sorted map by values (sorting copied from: http://stackoverflow.com/questions/613183/python-sort-a-dictionary-by-value)"""
    import operator
    sorted_pairs = sorted(name_scores.items(), key=operator.itemgetter(1))
    for (key, val) in sorted_pairs:
        print("%s, %4.2f" % (key.strip(), val))



if __name__ == "__main__":
    import sys

    #ngram_scores = generate_and_score_ngrams(prerated_names)
    #print_score_map_sorted(ngram_scores)

    inp_fname = sys.argv[1]

    ask_names = read_lines(inp_fname)

    import time
    start = time.clock()
    evil_names = train_evilometer_and_rate_txts(prerated_names, ask_names)
    end = time.clock()
    ## 0.024
    ## 0.025

    print_score_map_sorted(evil_names)

    print(end-start)
