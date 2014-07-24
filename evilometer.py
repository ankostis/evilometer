'''
:authors: ankostis@gmail.com, zachani
:created: 21 Jul 2014

Rates text for arbitrary characteristic (ie "evilness") by n_gram-comparing it with pre-rated ones.

Given a pre-rated list of texts, it decomposes them using n_grams and
constructs and google-like inverse-index.[1]_
It then uses this index to estimate the rating of any other text.
When finished, it prints back the  pairs of ``text, score``, ascending-ordered.

Example
-------

    >> python -m evilometer     prerated.csv asked.txt
    >> python -m evilometer     prerated.csv asked1.txt asked2.txt
    >> python -m evilometer     -i prerated1.csv prerated2.csv  \\
                                -o asked.txt asked2.txt
                                --dedupe-chars False


.. [1] http://nlp.stanford.edu/IR-book/html/htmledition/inverse-document-frequency-1.html

'''

from collections import defaultdict, Counter
import math
import re
import numpy as np
import pandas as pd
import argparse



class Indexer:
    """
    Rates text based on its n_gram similarity with a pre-rated texts on some arbitrary characteristic (ie "evilness")

    .. py:attribute:: ngram_scores
        The n_gram index of scores generated from the prerated input-texts by :meth:`_rate_ngrams()` on construction

    .. Seealso::
        rate_text()
    """

    def __init__(self, ngramer, prerated_names, **kws):
        """
        Builds the index from pre-rated texts

        :param NGramer ngramer: the algo to split words into ngrams
        """
        self.ngramer = ngramer

        ## Train index from sample.
        self.ngram_scores = self._rate_ngrams(prerated_names)


    def _rate_ngrams(self, prerated_names):
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
            ngram_freqs = self.ngramer.extract_ngrams(name.lower())

            ##
            for (ng, ng_freq) in ngram_freqs.items():
                assert ng_freq > 0 and score != 0, (ng_freq, score)
                ng_counters = ngram_counters[ng]
                ng_counters[0] += 1              	## update ``word_freq``.
                ng_counters[1] += score * ng_freq   ## update ``cummulative_score``

        def rate_ngram(doc_freq, cummulative_score):
            """Produces the scores for each n_gram according to the formula in :meth:`_rate_ngrams()`"""
            return cummulative_score * math.log(names_len / doc_freq)

        ngram_scores = {ng: rate_ngram(*counters) for (ng, counters) in ngram_counters.items()}

        return ngram_scores





    def rate_text(self, txt):
        """
        Rates a txt based on precalculated n_gram index of scores

        :param  str txt: the txt-line to rate
        """

        name_freqs = self.ngramer.extract_ngrams(txt)
        ngram_len = np.fromiter(iter(name_freqs.values()), dtype=int).sum()

        ## Score = sum(ngram_score * ngram_freq)
        #
        scores = [freq * self.ngram_scores[ngram] for (ngram, freq) in name_freqs.items() if ngram in self.ngram_scores]
        score = np.asarray(scores).sum() / ngram_len

        return score







class NGramer:
    """
    Splits texts in n_grams
    """

    Default_max_ngram = 3
    """The maximum length opf the n_grams to consider (ie when 3 --> 'abc'). """

    Default_add_1_ngrams = True
    """When `True`, scores derived also on single-letters. """

    def __init__(self, text_cleaner, add_1_ngrams=Default_add_1_ngrams, max_ngram=Default_max_ngram, **kws):
        """
        :param TextCleaner text_cleaner: used to preprocess all text by removing duplicates, non-ascii, etc
        """
        self.text_cleaner = text_cleaner
        self.add_1_ngrams = add_1_ngrams
        self.max_ngram = max_ngram


    def extract_ngrams(self, name):
        """
        Returns the frequencies of ngrams of a word after having appended the `^` and `$` chars at its start/end, respectively

        :param int n: the maximum length of the ngrams to extract, inclusive (ie: ``n=3 --> 'abc'``)
        :return: a map of ``{n_gram(str) --> freq(int)}``
        """

        name = self.text_cleaner.clean_chars(name)

        ## Gather 1 ngrams them without ``^$`` bracket-chars.
        #
        if self.add_1_ngrams:
            ngrams = list(name)
        else:
            ngrams = list()

        ## Gather 2+ ngrams after bracketingt words.
        #
        name = self.mark_word_boundaries(name)
        for n in range(2, self.max_ngram+1):
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
    def mark_word_boundaries(self,txt):
        """ Makes: ``"some name " --> "^some$ ^name$ "`` """

        txt = NGramer._mark_word_regex.sub('^', txt)
        txt = NGramer._mark_prefix_regex.sub('$', txt)

        return txt






class TextCleaner:
    """
    Simplifies text before n_gram extraction by replacing non-ascii chars with space or turning them to lower
    """

    Default_dedupe_chars = True
    """When `True`, ``cocodrillo --> cocodrilo``. """

    def __init__(self, dedupe_chars=Default_dedupe_chars, **kws):
        self.dedupe_chars = dedupe_chars


    _nonword_char_regex = re.compile(r'(\W|\d)+')
    _single_char_regex = re.compile(r'\b\w\b')
    _deduplicate_chars_regex = re.compile(r'(\w)\1+')
    _deduplicate_spaces_regex = re.compile(r' {2}')
    def clean_chars(self, txt):

        txt = TextCleaner._nonword_char_regex.sub(' ', txt).strip().lower()
        if self.dedupe_chars:
            txt = TextCleaner._deduplicate_chars_regex.sub(r'\1', txt)
        txt = TextCleaner._single_char_regex.sub('', txt)
        txt = TextCleaner._deduplicate_spaces_regex.sub('', txt)

        return txt





def str2bool(v):
    vv = v.lower()
    if (vv in ("yes", "true", "on")):
        return True
    if (vv in ("no", "false", "off")):
        return False
    try:
        return float(v)
    except:
        raise argparse.ArgumentTypeError('Invalid boolean(%s)!' % v)


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


def read_prerated_csv(csv_fnames):
    csv_fnames = [locate_file(fname) for fname in csv_fnames]

    prerated_txt = {}
    for csv_fname in csv_fnames:
        prerated_txt.update(pd.Series.from_csv(csv_fname, header=None).to_dict())

    return prerated_txt

def read_txt_lines(txt_fnames, text_cleaner, split_words=False):
    """
    :param split_words: When `True`, add also splitted-texts in the returned results.
    :return: a set with the text-lines read from all the files
    """

    txt_fnames = [locate_file(fname) for fname in txt_fnames]

    txt = set()
    for fn in txt_fnames:
        with open(fn) as fd:
            txt.update(set(fd.readlines()))

    word_set = set()

    if split_words:
        for ln in txt:
            word_set.update(text_cleaner.clean_chars(ln).split())
        txt.update(word_set)

    return txt


def print_score_map_sorted(name_scores):
    """Prints a sorted map by values (sorting copied from: http://stackoverflow.com/questions/613183/python-sort-a-dictionary-by-value)"""
    import operator
    sorted_pairs = sorted(name_scores.items(), key=operator.itemgetter(1))
    for (key, val) in sorted_pairs:
        print("%s, %4.2f" % (key.strip(), val))



def main(prgram_name):
    my_docstr = __doc__.splitlines()
    parser = argparse.ArgumentParser(description=my_docstr[4],
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='\n'.join(my_docstr[6:]))

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

    parser.add_argument('--dedupe-chars', type=str2bool, default=TextCleaner.Default_dedupe_chars, help='[default: %(default)s]')
    parser.add_argument('--max-ngram', type=int, default=NGramer.Default_max_ngram, help='[default: %(default)s]')
    parser.add_argument('--add-1-ngrams', type=str2bool, default=NGramer.Default_add_1_ngrams, help='[default: %(default)s]')
    parser.add_argument('--split-words', type=str2bool, default=False, help='[default: %(default)s]')

    opts = parser.parse_args()
    print(opts)


    text_cleaner = TextCleaner(**opts.__dict__)

    if opts.prerated_fname:
        opts.prerated_fnames = [opts.prerated_fname]
    opts.prerated_names = read_prerated_csv(opts.prerated_fnames)

    if opts.asked_fname:
        opts.asked_fnames = [opts.asked_fname]
    asked_txts = read_txt_lines(opts.asked_fnames, text_cleaner, opts.split_words)

    ngramer = NGramer(text_cleaner, **opts.__dict__)
    indexer = Indexer(ngramer, **opts.__dict__)

    #ngram_scores = generate_and_score_ngrams(prerated_names)
    #print_score_map_sorted(ngram_scores)

    import time
    start = time.clock()
    evil_names = {txt: indexer.rate_text(txt) for txt in asked_txts}
    end = time.clock()
    ## 0.024
    ## 0.025
    ## 0.0312
    # 0.0277

    from _version import __version_info__ as ver
    args = list(ver)
    args.append(end-start)
    print("ver{}.{}.{}: {:.4f}ms".format(*args))
    print_score_map_sorted(evil_names)



if __name__ == "__main__":
    import sys, os

    program_name = os.path.basename(sys.argv[0])[:-3]
    main(program_name)
