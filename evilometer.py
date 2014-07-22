'''
Created on 21 Jul 2014

@author: zachani
'''

from collections import defaultdict, Counter
import math
import re
import numpy as np
import pandas as pd


rated_names = {
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

def train_evilometer_and_rate_names(rated_names, ask_names):
    ## Train index from sample.
    #
    ngram_scores = rate_ngrams(rated_names)

    return {name: evilometer_name(ngram_scores, name) for name in ask_names}





def evilometer_name(ngram_scores, name):
    """
    Rates a name based on precalculated n_gram index of scores

    :param  scored_names: the precalculated n_gram index of scores, see :func:`rate_ngrams()`
    """

    name_freqs = extract_ngrams(name.lower())
    ngram_len = np.fromiter(iter(name_freqs.values()), dtype=int).sum()

    ## Score = sum(ngram_score * ngram_freq)
    #
    scores = [freq * ngram_scores[ngram] for (ngram, freq) in name_freqs.items() if ngram in ngram_scores]
    score = np.asarray(scores).sum() / ngram_len

    return score


def rate_ngrams(rated_names):
    """
    Constructs the n_gram index of scores from already rated real-names

    :param map rated_names: a map of ``{name(str) --> score(number)}``
    :return: a map of ``{n_gram(str) --> score(number)] }``

   First it constructs an inverse-index of ``{ngram_freqs --> [word_frequency, cummulative_score]}``
   and then "averages" the scores of names on each for n_gram as a single number using the
   following formula ::

       n_gram_score = cummulative_score * log(N / wf)

   where:

   * `cummulative_score`: is the product-sum of the frequency of each
             n_gram times the scores of the rated_names it was found in
   * `N`: is the total number of rated_names
   * `wf`: if the word_freq, that is, how many words contain this n_gram.

   .. Seealso::
       Inverse document frequency: http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html

       In doc-retrieving parlance, the ''word'' or ''rated_names'' terms above are equivalent to the ''document''.
    """
    names_len = len(rated_names)

    ## Initialize the inverse index.
    #
    def initial_counter_values():
        return [0, 0]                           ## ``[word_freq, cummulative_score]``
    ngram_counters = defaultdict(initial_counter_values)

    for (name, score) in rated_names.items():
        ngram_freqs = extract_ngrams(name.lower())

        ##
        for (ng, ng_freq) in ngram_freqs.items():
            ng_counters = ngram_counters[ng]
            ng_counters[0] += 1                 ## update ``word_freq``.
            ng_counters[1] += score * ng_freq   ## update ``cummulative_score``

    def rate_ngram(word_freq, cummulative_score):
        """Produces the scores for each n_gram according to the formula in :func:`rate_ngrams()`"""
        return cummulative_score * math.log(names_len / word_freq)
    ngram_scores = {ng: rate_ngram(*counters) for (ng, counters) in ngram_counters.items()}

    return ngram_scores


def extract_ngrams(name, n=3):
    """
    Returns the frequencies of ngrams of a word after having appended the `^` and `$` chars at its start/end, respectively

    :param int n: the maximum length of the ngrams to extract, inclusive (ie: ``n=3 --> 'abc'``)
    :return: a map of ``{n_gram(str) --> freq(int)}``
    """

    ## 1 ngrams:
    ##    gather them without ``^$`` bracket-chars.
    #
    #ngrams = list(name)
    ngrams = list()

    ## 2+ ngrams:
    ##     bracket them before gathering.
    #
    name = mark_word_boundaries(name) # some name --> ^some$ ^name$
    for n in range(2, n+1):
        ngrams.extend([name[i:i+n] for i in range(0, len(name) - n + 1)])
    #
    ##  The `ngrams` list now contains repetitions.

    ngram_freqs = Counter()
    ngram_freqs.update(ngrams)

    return ngram_freqs


_mark_word_regex = re.compile(r'\b')
_mark_prefix_regex = re.compile('\b\^')
def mark_word_boundaries(sentence):
    """ Makes: ``"some name " --> "^some$ ^name$ "`` """
    sentence = _mark_word_regex.sub('^', sentence)
    sentence = _mark_prefix_regex.sub('$', sentence)

    return sentence



def print_scored_names_sorted(name_scores):
    """Prints a sorted map by values (sorting copied from: http://stackoverflow.com/questions/613183/python-sort-a-dictionary-by-value)"""
    import operator
    sorted_pairs = sorted(name_scores.items(), key=operator.itemgetter(1))
    for pair in sorted_pairs:
        print("%s, %4.2f" % pair)



if __name__ == "__main__":
    import os
    import sys

    #ngram_scores = rate_ngrams(rated_names)
    #print_scored_names_sorted(ngram_scores)

    inp_fname = sys.argv[1]

    with open(os.path.join(os.path.dirname(__file__), inp_fname)) as fd:
        ask_names = fd.readlines()
        ask_names = [n.strip() for n in ask_names]

    evil_names = train_evilometer_and_rate_names(rated_names, ask_names)
    print_scored_names_sorted(evil_names)
