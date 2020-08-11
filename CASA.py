# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:28:45 2017

@author: nicolas.berube

"""
import os
import sys
import re
from unidecode import unidecode
import subprocess
import pickle
from datetime import datetime
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
from bisect import bisect_left, bisect_right
from gensim.models import Word2Vec
import multiprocessing
from sklearn.decomposition import IncrementalPCA
from googletrans import Translator
from nltk.corpus import stopwords


def progressBar(value, endvalue, bar_length=20):
    """Prints progress bar on screen.

    Takes about 0.5ms, so should not be used on every iteration of a big loop.

    Parameters
    ----------
    value: int
        current progress of the progress bar, out of a total of endvalue

    endvalue: int
        Final value of the progress bar
    """
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces,
                                                    int(round(percent * 100))))
    sys.stdout.flush()


def cleanpassage(passage):
    """Cleans a the characters of a sentence (lowercase alphanum ascii)

    All the word tokens will be separated by a space.

    Parameters
    ----------
    passage: str
        The sentence to clean

    Returns
    -------
    str
        The sentence in lowercase alphanumeric ascii characters
    """

    oldpassage = unidecode(passage)
    newpassage = ''
    # Iteration over individual characters
    for i, c in enumerate(oldpassage):
        if c.isalnum():
            newpassage += c
        # Keeping punctuation in numbered data and time
        elif (c in '.,:' and
              i > 0 and
              i < len(oldpassage)-1 and
              oldpassage[i-1].isnumeric() and
              oldpassage[i+1].isnumeric()):
            newpassage += c
        # Keeping the apostrophes and hyphens in words
        elif (c in '\'-' and
              i > 0 and
              i < len(oldpassage)-1 and
              oldpassage[i-1].isalpha() and
              oldpassage[i+1].isalpha()):
            newpassage += c
        elif c == '(':
            # If there's a closing parenthesis in the rest of the word,
            # without any space separating them
            try:
                k = oldpassage[i:].index(')')+i
                addflag = True
                # If the parenthesis is right after, don't add any
                if k == i+1:
                    addflag = False
                # If there's only part of a word between the 2 parentheses
                for j, d in enumerate(oldpassage[i+1:k]):
                    if (not d.isalnum() and
                        not (d in '.' and
                             j > 0 and
                             j < k-i-2 and
                             oldpassage[i+j].isnumeric() and
                             oldpassage[i+j+2].isnumeric()) and
                        not (d in '-' and
                             j > 0 and
                             j < k-i-2 and
                             oldpassage[i+j].isalpha() and
                             oldpassage[i+j+2].isalpha())):
                        addflag = False
                        break
            # If there's just no closing parenthesis in the rest of
            # the sentence
            except ValueError:
                addflag = False
            if addflag:
                newpassage += c
            else:
                newpassage += ' '
        elif c == ')':
            # If the closing parenthesis is associated to an opening
            # parenthesis previously analysed and confirmed valid
            if '(' in newpassage:
                if ')' in newpassage:
                    if newpassage.count('(') > newpassage.count(')'):
                        newpassage += c
                    else:
                        newpassage += ' '
                else:
                    newpassage += c
            else:
                newpassage += ' '
        else:
            newpassage += ' '
    return ' '.join(newpassage.split()).lower()


def span_shift(span,
               string):
    """
    Returns the character position shifts from usage of unidecode().

    Using unidecode sometimes creates 2 characters out of a single one,
    which shifts the character position in the string.

    Parameters
    ----------
    span: (int, int)
        The position range of the cleaned unidecoded string to consider
        in format [starting:ending (exclusively)]

    string : str
        The raw sentence that was cleaned with unidecode

    Returns
    -------
    (int, int)
        Range of the original, raw, untreated text corresponding
        to the characters that would create the ones in the specified
        range on a cleaned string.
    """
    weird_char = {}
    for c in set(string):
        if (c not in ('0123456789'
                      'abcdefghijklmnopqrstuvwxyz'
                      'ABCDEFGHIJKLMNOPQRSTUVWXYZ') and
                len(unidecode(c)) != 1):
            weird_char[c] = len(unidecode(c))
    if not weird_char:
        return span

    if span[0] < 0:
        span[0] += len(unidecode(string))
    if span[1] < 0:
        span[1] += len(unidecode(string))
    if span[0] < 0 or span[1] < 0 or span[1] < span[0]:
        raise NameError('Incorrect span')

    old_span = []
    i_clean = 0
    end_flag = False
    for i_raw, c in enumerate(string):
        if c in weird_char:
            i_clean += weird_char[c]-1
        if not old_span and i_clean > span[0]:
            old_span.append(i_raw)
        if end_flag and i_clean >= span[1]:
            old_span.append(i_raw)
            break
        if i_clean >= span[1] - 1:
            end_flag = True
        i_clean += 1

    return old_span


def ngram_positions(ngram,
                    phrase):
    """Returns all positions of the ngram in a sentence

    Parameters
    ----------
    ngram: str
        list of characters to match in the sentence

    phrase: str
        sentence to analyse

    Returns
    -------
    List of (int, int)
        List of all starting/ending(exclusively) positions where the ngrams
        are located in the sentence string
    """
    positions = []
    index = 0
    while index < len(phrase):
        index = phrase.find(ngram, index)
        if index == -1:
            break
        positions.append([index, index+len(ngram)])
        index += 1
    return positions


def fuse_positions(positions):
    """
    Fuses different integer spans when needed be.

    For example, [1,4] and [4,6] will be merged as [1,6]

    Parameters
    ----------
    positions: list of (int, int)
        The range of indexing position to be merged together
        in format [starting:ending (exclusively)]

    Returns
    -------
    list of (int, int)
        The new list of ranges, but merged when needed be
    """
    fpos = sorted(positions, key=lambda k: k[0])
    i = 1
    while i < len(fpos):
        if fpos[i][0] <= fpos[i-1][1]:
            fpos[i-1][1] = max(fpos[i-1][1], fpos[i][1])
            del fpos[i]
        else:
            i += 1
    return fpos


def import_dictionaries(folder=''):
    """
    Imports english and french dictionaries

    Needs to have the word_dictionary.json and
    liste.de.mots.francais.sansaccents.txt in the specified folder

    Parameters
    ----------
    folder: dict of {str: str}
        Folder containing the english and french language files.
        Default is '', corresponding to current working directory

    Returns
    -------
    dict of str: set of str, optional
        The word_dictionaries object necessary for the create_index() and
        query() class objects.
        Each dictionary (in the values of the object) is a set of strings
        containing all dictionary words for a specific language.
        Languages are identified by a 2-letters string (in the keys of the
        object) which are the results of the
        googletrans.Translator().detect().lang function.
        Dictionaries are used for filtering the gibberish from the pdf texts
        and recognize language.
        If None/empty, will not use those functionalities.
        Default is None.
    """
    # English
    try:
        import json
        filename = "words_dictionary.json"
        with open(Path(folder) / filename, "r") as english_dictionary:
            valid_words = json.load(english_dictionary)
            english_words = valid_words.keys()
    except Exception as e:
        print(str(e))

    # French
    try:
        filename = "liste.de.mots.francais.sansaccents.txt"
        with open(Path(folder) / filename, "r") as french_dictionnary:
            data = french_dictionnary.read()
        valid_words = dict((el, 0) for el in data.split('\n'))
        french_words = valid_words.keys()
    except Exception as e:
        print(str(e))

    word_dictionaries = {'en': set(english_words),
                         'fr': set(french_words)}
    return word_dictionaries


class text_treatment():
    """
    Functions for text treatment with dictionary filters.

    Parameters
    ----------
    word_dictionaries: dict of str: set of str, optional
        Each dictionary (in the values of the object) is a set of strings
        containing all dictionary words for a specific language.
        Languages are identified by a 2-letters string (in the keys of the
        object) which are the results of the
        googletrans.Translator().detect().lang function.
        Dictionaries are used for filtering the gibberish from the pdf texts
        and recognize language.
        If None/empty, will not use those functionalities.
        Default is None.

    default_lang: str, optional
        Default language of the dictionaries. Corresponds to a key of the
        word_dictionaries object to default to if the language
        recognition fails.
        If None or empty, no dictionary will be used if the language
        recognition fails.
        Default is None.

    Attributes
    ----------
    translator:
        Google api corresponding to the googletrans.Translator() object
        for language recognition
    """

    def __init__(self,
                 word_dictionaries=None,
                 default_lang=None):
        if word_dictionaries is None:
            self.word_dictionaries = {}
        else:
            self.word_dictionaries = word_dictionaries
        self.default_lang = default_lang
        self.translator = Translator()

    def dictionary_detect(self,
                          string):
        """
        Detect the language of a string of words for dictionary usage.

        This uses the googletrans.Translator().detect().lang function.

        Parameters
        ----------
        string: str
            string of words to input in

        Returns
        -------
        set of str
            Set of all words in the detected language of the string,
            corresponding to a value of the self.word_dictionaries object.
            Is used for filtering the gibberish from the pdf texts.
            If no language was detected, returns an empty set and
            won't use the functionality.
        """
        lang = self.translator.detect(string[:5000]).lang
        if lang not in self.word_dictionaries:
            if (self.default_lang and
                    self.default_lang in self.word_dictionaries):
                return self.word_dictionaries[self.default_lang]
            else:
                return {}
        return self.word_dictionaries[lang]

    def format_txt(self,
                   pages_text,
                   n_ignore=1):
        """
        Formats a list of strings from pages of a pdf into a list of sentences.

        It will analyze the page breaks with simple regex and identify if a
        sentence continues on the next page.
        If that is the case, the sentence will be joined to
        the previous sentence in the previous page, and treated as if it
        was written on that page entirely.

        This is because searches on legal documents tend to aim at retrieving
        sentences/clauses, and therefore the position of the start of the
        sentence.

        Parameters
        ----------
        pages_text: list of str
            List of strings where each string is the text on the page
            (obtained from pdftotext, for example).

        n_ignore: int, optional
            Number of pages to ignore at the start of the pdf.
            Default is 1 (ignores the cover page).

        Returns
        -------
        list of list of str
            List of the text on the page, where the text is separated
            into a list of the sentences on the page.
        """

        dict_words = self.dictionary_detect(
            ' '.join(' '.join(pages_text[n_ignore:]).split())
            )

        data = pages_text + ['.Ending final flag.']
        # File data that will be returned by the function
        fdata = []
        # Page data. fdata will be a list of newpage
        newpage = []
        # Sentence data
        newdat = ''
        # Flag indicating that we are in a new page
        newpageflag = False

        # All periods will delimit the end of a sentence
        # Therefore, the following expressions will be replaced
        replacelist = {'Mr.': 'Mr',
                       'Ms.': 'Ms',
                       'Mrs.': 'Mrs',
                       'Dr.': 'Dr'}

        for pagex in data:
            page = pagex
            # Replacement to get rid of periods that aren't at the end of
            # a sentence
            for k, v in replacelist.items():
                page = page.replace(k, v)

            # This flag will be True at the start of a page is the previous
            # page was completely empty, therefore, adding an empty page here
            if newpageflag:
                fdata.append([])
            newpageflag = True

            # Iteration over all sentences
            for datx in page.replace('. ', '.\n').split('\n'):
                # Gets rids of non-space characters
                dat = ' '.join(datx.split())
                # iflag will check if the  sentence contains any real word
                iflag = False
                # Gets rid of non-alphanumeric characters for
                # dictionary analysis
                for w in re.sub('[^a-zA-Z]+',
                                ' ',
                                unidecode(dat)).lower().split():
                    if (not dict_words) or w in dict_words:
                        iflag = True
                        break

                # if sentence devoid of true words
                if not iflag:
                    continue

                # ending  of the previous sentence
                soft_end = False
                hard_end = False
                if len(newdat) > 0:
                    soft_end = newdat[-1].isnumeric() or newdat[-1].isalpha()
                    hard_end = newdat[-1].islower() or newdat[-1] in '-):;,"\''
                # start of the current sentence
                soft_start = (dat[0].isalpha() or
                              dat[0].isnumeric() or
                              dat[0] in '($"\'')
                hard_start = dat[0].islower()
                very_hard_start = (hard_start and
                                   len(dat.split()[0]) > 1 and
                                   dat.split()[0] in dict_words)

                # if the sentence (dat) should merge with
                # previous sentence (newdat)
                if (very_hard_start or
                        (soft_start and hard_end) or
                        (hard_start and soft_end)):
                    newdat += ' '+dat
                # if note, current sentence should be a new sentence
                # and previous sentence has to be added
                else:
                    iflag = False
                    # Gets rid of non-alphanumeric characters for
                    # dictionary analysis
                    for w in re.sub('[^a-zA-Z]+',
                                    ' ',
                                    unidecode(newdat)).lower().split():
                        if (not dict_words or
                                (w in dict_words and len(w) > 1)):
                            iflag = True
                            break
                    # If sentence contains real words that are
                    # not single letters (devoid of information)
                    # then adds the data to the page
                    if iflag:
                        newpage.append(newdat)
                    # if we are in a new page, adding the prvious page to data
                    if newpageflag:
                        fdata.append(newpage)
                        newpage = []
                        newpageflag = False
                    # creating new sentence
                    newdat = dat
        return fdata[n_ignore:]


class pdf_treatment():
    def __init__(self,
                 exec_path=''):
        """
        Contains functions to treat pdfs into text.

        Parameters
        ----------
        exec_path: str/Path, optional
            Path to the pdftotext and pdfinfo executables.
            Both can be downloaded through the Windows xpf tools at
            http://www.xpdfreader.com/download.html  --
            http://www.xpdfreader.com/dl/xpdf-tools-win-4.00.zip
            If empty, the shell PATH variable will be used.
            Default is ''.
        """
        self.exec_path = str(Path(exec_path).absolute()) + '/'

    def pdftotext(self,
                  input_folder,
                  output_folder=None,
                  rerun=False,
                  verbose=True):
        """
        Runs pdftotext on all pdfs presents in a folder.

        Exports the results in pickle files in an output folder where
        results are a list of strings, and each strings is the output
        from a page in the pdf.

        Parameters
        ----------
        input_folder: str
            Path of the folder to all the pdfs to be treated

        output_folder: str, optional
            Path of the folder to put the exported pickle files.
            If None, will take Parent(input_folder)/txt-pdftotext
            Default is None.

        rerun: bool, optional
            False indicates that pdf files that have a pickle file of the
            same name will not be computed again.
            Default is False.

        verbose: bool, optional
            Will print progress on screen.
            Default is True.
        """
        if output_folder is None:
            output_folder = (Path(input_folder).absolute().parent /
                             'txt-pdftotext')
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        notfound = []
        donenames = set()
        if not rerun:
            donenames = set([f for f in os.listdir(input_folder)
                             if Path(f).suffix == '.pkl'])
        filenames = [f for f in os.listdir(input_folder)
                     if Path(f).suffix == '.pdf' and f not in donenames]
        for filename in tqdm(filenames):
            filepath = Path(input_folder).absolute() / filename

            # Number of pages
            nopage = self.nopagepdf(filepath)
            if nopage == 0:
                notfound.append(filename)

            # Running pdftotext on all pages in parallel
            processlist = []
            for i in range(1, nopage):
                processlist.append(subprocess.Popen([
                    self.exec_path + 'pdftotext',
                    '-f',
                    f'{i}',
                    '-l',
                    f'{i}',
                    '-layout',
                    str(filepath),
                    str(output_folder / (filepath.stem + f'{i:04}.txt'))
                    ]))
            _ = [pr.wait() for pr in processlist]

            # Merging data from all pages
            filedata = []
            for i in range(1, nopage):
                try:
                    with open(output_folder /
                              (filepath.stem + f'{i:04}.txt'),
                              'r',
                              encoding='latin-1') as f:
                        filedata.append(f.read())
                except UnicodeDecodeError:
                    with open(output_folder /
                              (filepath.stem + f'{i:04}.txt'),
                              'r',
                              encoding='utf-8') as f:
                        filedata.append(f.read())

            # Deleting temporary files
            for i in range(1, nopage):
                os.remove(output_folder / (filepath.stem + f'{i:04}.txt'))

            # Creating pickle files of the result
            pickle.dump(filedata,
                        open(output_folder / (filepath.stem + '.pkl'), 'wb'))

        if verbose:
            print()
            print(str(datetime.now()), '\tDONE')

            if len(notfound) == 0:
                print('No emply pdfs')
            else:
                notfound_path = output_folder / 'empty_pdfs.txt'
                with open(notfound_path, 'w') as nf:
                    nf.write('\n'.join(notfound))
                print(f'{len(notfound)} empty pdfs in file {notfound_path}')

            obsolete = [f for f in os.listdir(output_folder)
                        if (Path(f).stem + '.pdf') not in filenames and
                        f not in {'.DS_Store', 'empty_pdfs.txt'}]
            print(len(obsolete), 'obsolete files')
            for f in obsolete:
                print('\t', f)

    def nopagepdf(self,
                  filepath):
        """Returns the number of pages in a pdf using pdfinfo

        Parameters
        ----------
        filepath: str
            Path of the pdf file

        Returns
        -------
        int
            Number of pages in the pdf file
        """
        pageline = 'Pages: 0'
        try:
            meta = subprocess.check_output(
                self.exec_path + 'pdfinfo ' + str(filepath),
                shell=True
                ).decode("utf-8")
        except UnicodeDecodeError:
            try:
                meta = subprocess.check_output(
                    self.exec_path+'pdfinfo ' + str(filepath),
                    shell=True
                    ).decode("latin-1")
            except UnicodeDecodeError:
                meta = ''
                print('ERROR: Can\'t found encoding in metadata')
        meta.replace('\r\n', '\n')
        for i in meta.split('\n'):
            if i[:5].lower() == 'pages':
                pageline = i
        return int(pageline.split()[-1])


class Sentences():
    """
    Iterator over corpus documents, returning all sentences.

    Will use its own attributes to show progress bar when used with
    gensim.models.Word2Vec

    Parameters
    ----------
    pkl_folder: str
        Path of the folder containing the corpus documents.
        Each corpus document is a pickle file containing a
        list of list of strings, where a string is a single sentence,
        grouped in a single list per page, and grouped together for the
        document.
        Corresponds the output_folder of the pdf_treatment().pdftotext()
        function.

    text_treatment: text_treatment()
        text_treatment() class object containing word dictionaries
        for filtering text when reading.

    dict_words: set of str
        Set of words of the language of the document in filter_text() function.
        Is used for filtering the gibberish from the pdf texts.
        If None, will not use this functionality.
        Default is None.

    iter_total: int, optional
        Total of iterator passes. Used to show progress bar. Default is 1.

    verbose: bool, optional
        Will print progress of the functions on screen.
        Default is True.

    Attributes
    ----------
    iter_i: int
        Current index of iterator passes. Used by an iterator to show
        progress bar. Is changing through the computation.

    Yields
    ------
    list of str
        List of words in the sentence of the corpus
    """

    def __init__(self,
                 pkl_folder,
                 text_treatment,
                 dict_words=None,
                 iter_total=1,
                 verbose=True):
        self.pkl_folder = pkl_folder
        self.text_treatment = text_treatment
        self.iter_total = iter_total
        if dict_words is None:
            self.dict_words = {}
        else:
            self.dict_words = dict_words
        self.iter_i = 1
        self.verbose = True

    def __iter__(self):
        icount = 0
        listnames = os.listdir(self.pkl_folder)
        ntotal = len(listnames)
        if self.verbose:
            print(f'\rIteration {self.iter_i}/{self.iter_total}' + ' '*50)
        for fname in listnames:
            icount += 1
            if (icount % max(1, ntotal//1000) == 0 and self.verbose):
                progressBar(icount, ntotal)
            pages_text = None
            if fname[-4:] == '.pkl':
                pages_text = pickle.load(open(Path(self.pkl_folder) / fname,
                                              'rb'))
            elif fname[-4:] == '.txt':
                with open(open(Path(self.pkl_folder) / fname)) as f:
                    pages_text = [f.read()]
            if pages_text:
                for page in self.text_treatment.format_txt(pages_text):
                    for sent in page:
                        yield cleanpassage(sent).split()
        self.iter_i += 1


class create_index():
    """
    Functions to create the index files for the search engine.

    Parameters
    ----------
    pkl_folder: str, optional
        Path of the folder containing the corpus documents.
        Each corpus document is a pickle file containing a
        list of list of strings, where a string is a single sentence,
        grouped in a single list per page, and grouped together for the
        document.
        Corresponds the output_folder of the pdf_treatment().pdftotext()
        function.
        All other output and model files will be put in Parent(pkl_folder).
        If None, will use the current Path()/txt-pdftotext. Default is None.

    word_dictionaries: dict of str: set of str, optional
        Each dictionary (in the values of the object) is a set of strings
        containing all dictionary words for a specific language.
        Languages are identified by a 2-letters string (in the keys of the
        object) which are the results of the
        googletrans.Translator().detect().lang function.
        Dictionaries are used for filtering the gibberish from the pdf texts
        and recognize language.
        If None/empty, will not use those functionalities.
        Default is None.

    default_lang: str, optional
        Default language of the dictionaries. Corresponds to a key of the
        word_dictionaries object to default to if the language
        recognition fails.
        If None or empty, no dictionary will be used if the language
        recognition fails.
        Default is None.

    verbose: bool, optional
        Will print progress of the functions on screen.
        Default is True.

    Attributes
    ----------
    root_folder: str
        Path to the main repository of the analysis, where model files
        and outputs will be saved. Corresponds to Parent(pkl_folder).

    text_treatment: text_treatment()
        text_treatment() class object containing word dictionaries
        for filtering text when reading.

    allvocab_to_widx: dict of {str: int}
        Dictionary where the key is the string of a word in the corpus,
        and the value (word index) is an int index identifying the word.
        The index starts as the default index of order of apparition in the
        corpus, but then switches to an index ordered by descending word count
        (widx_ord)
        The exact type of the value is identified via self.type_word

    idx_to_filename: dict of {int: str}
        Dictionary where the key is an index representing a document
        in the database, and the value is the string of the file name
        of the document

    widx_to_count_total: dict of {int: int}
        Dictionary where the key is the word index (from allvocab_to_widx)
        and the value if the word count, number of occurences of the token
        in the corpus

    widx_to_count_temp: dict of {int: int}
        Subset of the widx_to_count_total dictionary to avoid
        filling in the RAM in the calculation.
        Is used for the inversted index calculation of temporary_inverted

    sentence_words: 1D int Numpy array
        Sequential identification of all the word token in the corpus.
        Will build inverted index via numpy.argsort.
        The type of the array is the type of the word index self.type_word

    sidx: int
        Current index of the analyzed word (also called "inverted" in the
        code). It is simply the counts from the beginning of the corpus
        and increments through all pages, then all documents.
        Is changing through the computation.

    n_swords: int
        Counts the number of sub-objects of sentence_words to avoid
        filling the RAM

    temporary_filenames: list of str
        List of all the file names of temporary files that will be reimported
        and fused at the end of the computation. This is to avoid filling up
        the RAM.

    pag_breaks: list of int
        List of the index of words (self.sidx) corresponding to the
        beginning of a new page

    doc_breaks: list of int
        List of the index of words (self.sidx) corresponding to the
        beginning of a new page.
        Will eventually be transformed in the list of page numbers instead
        of word indexes.

    type_word: str
        Type of the word index encoding. If the number of distinct words
        tokens in the corpus is under 65535 (small corpuses), it should
        be 'u2'. If not, 'u4' (with an upper limit of 4294967295)

    type_invidx: str
        Type of the inverted index self.sidx encoding. Normally 'u4'.
        If the whole corpus is more than 4294967295 words
        (approximately 16 Gb), should be changed for 'u8'.
        The implementation of 'u8' was never tested.

    size_swords: int
        Variable limiting the use of the RAM when the code is executed.
        This is the size of the sentence_words objects.
        A value of 250M corresponds to 250M * 'u4' (self.type_word) = 1Gb.
        This will take approximately to 4 Gb in RAM with the argsort function.
        This value (times 4) will also be used to limit the RAM of the
        incremental PCA for sentence2vec.

    size_sindex: int
        Variable limiting the size of the objects saved on the hard drive
        for the use of numpy memmap functions on older systems.
        32-bit systems cannot support numpy memmap on object bigger
        than 2 Gb, so size_sindex should not be higher than 500M
        (times type_invidx = 'u4' which equals 2 Gb)

    widxord_to_cumcount:  numpy array
        Cumulative count of all words, ordered by descending word count.
        The word count of a word corresponding to index widx_ord is
        widxord_to_cumcount[widx_ord] - widxord_to_cumcount[widx_ord-1]

    wvvocab_to_wvidx: dict of {str:int}
        Dictionary where key=word string, where the word
        occurences > min_word_count specified in the word2vec calculation,
        and value=wvidx, word2vec_index an index corresponding to the
        embedding

    wvidx_to_widxord: numpy array
        Link between wvidx (word2vec_index) and widxord (word_index_ord)
        for the number of occurences calculation for sentence2vec

    embedding_matrix: 2D numpy array
        Matrix of vectors representing the embeddings for each word,
        which are indexed as wvidx, a specific word2vec index
    """

    def __init__(self,
                 pkl_folder=None,
                 word_dictionaries=None,
                 default_lang=None,
                 verbose=True):
        if pkl_folder is None:
            self.pkl_folder = str(Path().absolute().parent /
                                  'txt-pdftotext')
        else:
            self.pkl_folder = pkl_folder
        self.root_folder = str(Path(self.pkl_folder).parent)
        self.verbose = verbose
        self.text_treatment = text_treatment(
            word_dictionaries=word_dictionaries,
            default_lang=default_lang)

        # Meta-parameters
        self.sidx = 0
        self.n_swords = 1
        self.type_word = 'u4'
        self.type_invidx = 'u4'
        self.size_swords = 250000000
        self.size_sindex = 500000000

        # Model files
        self.idx_to_filename = {}
        self.widxord_to_cumcount = None
        self.pag_breaks = []
        self.doc_breaks = []

        # Variable used during the calculation
        self.allvocab_to_widx = {}
        self.widx_to_count_total = {}
        self.widx_to_count_temp = {}
        self.sentence_words = None
        self.temporary_filenames = []

    def saving_objects(self):
        """
        Saving objects method during self.import_documents()

        Method that, once the RAM is full, will create an inverted index
        with the collected data so far and will save a
        temporary_inverted.npy and a temporary_count.npy object on the disk,
        in the {self.root_folder}/models/ folder.

        The way the temporary index is stored is a serie of positions
        (sidx) in the corpus in order of the word index.
        Since all the words have a count, calculating the cumulative count
        is actually the index ot the inverted index. In other words,
        all the positions (sidx) of word corresponding to word index X are
        located at positions Y to Y+Z in the inverted index, where Y is the
        cumulative count of all words with word index lower than X,
        and Z is the word count of word index X.

        Uses widx_to_count_temp, sentence_words.
        Updates widx_to_count_total and temporary_filenames.
        """
        folder = Path(self.root_folder) / 'models'

        if self.verbose:
            sys.stdout.write(f'\nSaving object {self.n_swords}' + ' '*20)
            sys.stdout.flush()
        # Sorting of the current sentence_words of the corpus slice
        words_inverted = np.argsort(
            self.sentence_words[:(self.sidx-1) % self.size_swords + 1]
            ).astype(self.type_invidx)+(self.n_swords-1)*self.size_swords

        # Calculating cumulative word token count of words for the current
        # sentence_words slice.
        widx_to_cumcount = np.empty(len(self.widx_to_count_temp)+1,
                                    dtype=self.type_invidx)
        widx_to_cumcount[0] = 0
        wcount_sum = 0
        for widx in range(len(self.widx_to_count_temp)):
            wcount = self.widx_to_count_temp[widx]
            # Sorting the indexes for each word
            words_inverted[wcount_sum: wcount_sum+wcount] = \
                np.sort(words_inverted[wcount_sum: wcount_sum+wcount])
            # Updating the total word count through the whole corpus
            self.widx_to_count_total[widx] += wcount
            # Updating the cumulative word count for the current corpus slice
            widx_to_cumcount[widx+1] = wcount + widx_to_cumcount[widx]
            wcount_sum += wcount
            # Resetting the word count for the next slice
            self.widx_to_count_temp[widx] = 0
        np.save(folder / f'temporary_inverted{self.n_swords}.npy',
                words_inverted)
        np.save(folder / f'temporary_count{self.n_swords}.npy',
                widx_to_cumcount)
        self.temporary_filenames.append(
            [folder / f'temporary_inverted{self.n_swords}.npy',
             folder / f'temporary_count{self.n_swords}.npy']
            )

        if self.verbose:
            print(f'\rSaving object {self.n_swords} DONE')
        # Updating the slice index
        self.n_swords += 1
        # Reseting the sentence_words slice
        self.sentence_words = np.empty(self.size_swords,
                                       dtype=self.type_word)

    def import_documents(self,
                         checksum=False):
        """
        Reads all document text files and create temporary index files.

        Will save the temporary_inverted.npy and temporary_count.npy
        files on the disk, as well as doc_breaks.npy, pag_breaks.npy,
        idx_to_filename.pkl, widxord_to_cumcount.npy and
        allvocab_to_widxord.pkl

        All model files will be saved in {self.root_folder}/models/

        Parameters
        ----------
        checksum: bool, optional
            If True, will run extra checks to make sure the code is working
            properly when building the full inverted index from temporary
            files. Will slow down the code. Default is False.
        """
        input_folder = Path(self.pkl_folder)
        model_folder = Path(self.root_folder) / 'models'

        if self.verbose:
            print(str(datetime.now())+'\t'+'Step 1: First pass of the corpus')

        if not os.path.exists(model_folder):
            os.mkdir(model_folder)
        else:
            for f in os.listdir(model_folder):
                os.remove(model_folder / f)

        self.sentence_words = np.empty(self.size_swords,
                                       dtype=self.type_word)

        # Iterating over all documents
        all_names = [f for f in sorted(os.listdir(input_folder))
                     if f[-4:] in {'.pkl', '.txt'}]
        icount = 0
        ntotal = len(all_names)
        for name in all_names:
            icount += 1
            if icount % max(1, ntotal//1000) == 0:
                progressBar(icount, ntotal)
            path = os.path.join(input_folder, name)
            filename, file_extension = os.path.splitext(path)
            filename = filename.split("/")[-1]

            # Updating the filenames of the documents
            self.idx_to_filename[len(self.idx_to_filename)] = filename

            # Updating the indexes for the document breaks
            self.doc_breaks.append(self.sidx)

            # Importing the text data from the pickle
            if file_extension == '.pkl':
                pages_text = pickle.load(open(path, 'rb'))
            elif file_extension == '.txt':
                with open(path) as f:
                    pages_text = [f.read()]
            else:
                raise NameError('filetype introuvable')
            for page in self.text_treatment.format_txt(pages_text):
                # Updating the indexes for the page breaks
                self.pag_breaks.append(self.sidx)

                # Iterating over sentences
                for word in [w for sent in page
                             for w in cleanpassage(sent).split()]:
                    # If the word token is new, adding to the
                    # dictionaries
                    if word not in self.allvocab_to_widx:
                        self.widx_to_count_temp[
                            len(self.allvocab_to_widx)
                            ] = 0
                        self.widx_to_count_total[
                            len(self.allvocab_to_widx)
                            ] = 0
                        self.allvocab_to_widx[word] = \
                            len(self.allvocab_to_widx)

                    widx = self.allvocab_to_widx[word]
                    # Updating the index
                    self.sentence_words[
                        self.sidx % self.size_swords
                        ] = widx
                    self.widx_to_count_temp[widx] += 1
                    self.sidx += 1
                    # If we're at the end of the RAM-limited object
                    if self.sidx >= self.n_swords*len(self.sentence_words):
                        self.saving_objects()
        self.saving_objects()

        # Freeing memory
        del self.sentence_words
        del self.widx_to_count_temp

        # Saving the remaining corpus objects
        self.pag_breaks = np.array(self.pag_breaks, dtype=self.type_invidx)
        np.save(model_folder / 'pag_breaks.npy', self.pag_breaks)
        self.doc_breaks = np.array(self.doc_breaks, dtype=self.type_invidx)
        self.doc_breaks = np.concatenate((
            np.searchsorted(self.pag_breaks,
                            self.doc_breaks).astype(self.type_invidx),
            np.array([len(self.pag_breaks)], dtype=self.type_invidx)
            ))
        np.save(model_folder / 'doc_breaks.npy', self.doc_breaks)
        pickle.dump(self.idx_to_filename,
                    open(model_folder / 'idx_to_filename.pkl', 'wb'))

        if self.verbose:
            print(str(datetime.now())+'\t'+'Step 2: Indexing of the corpus')

        # re-indexing the word index where the new index order will be
        # based on word frequencies, now that the whole corpus has
        # been read once.

        # Calculating all word counts
        widx_to_count = np.empty(len(self.widx_to_count_total),
                                 dtype=self.type_invidx)
        for widx, wcount in self.widx_to_count_total.items():
            widx_to_count[widx] = wcount

        del self.widx_to_count_total

        # Correspondance between original word index and new ordered word index
        widxord_to_widx = np.argsort(
            widx_to_count
            )[::-1].astype(self.type_word)
        widx_to_widxord = np.argsort(
            widxord_to_widx
            ).astype(self.type_word)

        # New allvocab_to_widx with the new ordered index by word count
        self.allvocab_to_widx = {k: int(widx_to_widxord[v])
                                 for k, v in self.allvocab_to_widx.items()}
        pickle.dump(self.allvocab_to_widx,
                    open(model_folder / 'allvocab_to_widxord.pkl', 'wb'))

        # widxord_to_cumcount: Cumulatif word occurence count for each
        # word with corresponding index widxord
        self.widxord_to_cumcount = np.empty(len(widx_to_count),
                                            dtype=self.type_invidx)
        for widxord, widx in enumerate(widxord_to_widx):
            if widxord == 0:
                self.widxord_to_cumcount[widxord] = widx_to_count[widx]
            else:
                self.widxord_to_cumcount[widxord] = \
                    self.widxord_to_cumcount[widxord-1] + widx_to_count[widx]

        np.save(model_folder / 'widxord_to_cumcount.npy',
                self.widxord_to_cumcount)
        del widx_to_count

        # list of (int, int) corresponding to the (start, end) widxord to
        # consider to include in an object full_inverted[X].npy
        mem_batches = []
        idx_a = 0
        used_mem = 0
        while idx_a < len(self.widxord_to_cumcount):
            idx_b = min(len(self.widxord_to_cumcount),
                        bisect_left(self.widxord_to_cumcount,
                                    used_mem + self.size_sindex)+1)
            used_mem = min(used_mem + self.size_sindex,
                           self.widxord_to_cumcount[idx_b-1])
            mem_batches.append([idx_a, idx_b])
            idx_a = bisect_right(self.widxord_to_cumcount,
                                 used_mem)

        # Loop over full_inverted[X].npy objects
        # Every loop will import all temporary object sequentially and grab
        # the necessary information to build the full_inverted.npy object
        # The size of full_inverted.npy is capped by size_sindex
        # This is mostly a reordering/defragmentation of the temporary objects
        used_mem = 0
        full_inv_idx = 0
        for (idx_a, idx_b) in mem_batches:
            full_inv_idx += 1
            # Identifying the cumulative counts of the words for the current
            # batch
            if idx_a == 0:
                position_words = np.concatenate((
                    np.array([0], dtype=self.type_invidx),
                    self.widxord_to_cumcount[idx_a: idx_b-1]
                    ))
            else:
                position_words = np.copy(
                    self.widxord_to_cumcount[idx_a-1:idx_b-1]
                    )
            # Size of the full_inverted.npy object
            size_batch = min(self.size_sindex,
                             self.widxord_to_cumcount[idx_b-1] - used_mem)
            # Full_inverted.npy object for the current batch
            widxord_to_inverted = np.empty(size_batch,
                                           dtype=self.type_invidx)

            # The following line is for debugging purposes
            # to make sure all elements of the vector widxord_to_inverted are
            # written and accounted and the fusion of temporary objects worked
            if checksum:
                widxord_to_inverted = \
                    np.zeros(size_batch, dtype=self.type_invidx)

            # Importing relevant data in the temporary objects
            for i, filenames in enumerate(self.temporary_filenames):

                # Inverted index, listing all the positions of the words
                temporary_inverted = np.load(filenames[0])
                # Cumulative count of word, sorted by old word index
                # corresponding to the positions of the inverted index
                # of that specific word
                temporary_count = np.load(filenames[1])

                for widxord in range(idx_a, idx_b):

                    if (widxord %
                            max(1, len(self.widxord_to_cumcount)//1000)) == 0:
                        progressBar(widxord +
                                    i*(idx_b-idx_a) +
                                    idx_a*(len(self.temporary_filenames)-1),
                                    len(self.widxord_to_cumcount) *
                                    len(self.temporary_filenames))

                    # Old word index, before the sorting per count
                    widx = widxord_to_widx[widxord]

                    # If the word is present in this slice of the corpus
                    if widx < len(temporary_count)-1:
                        # Grabbing the list of inverted index of this word
                        temp = temporary_inverted[temporary_count[widx]:
                                                  temporary_count[widx+1]]
                        # Position of the full_inverted object
                        # to put the temporary object's slice
                        pos_bot = position_words[widxord-idx_a] - used_mem
                        pos_top = min(pos_bot + len(temp), size_batch)
                        position_words[widxord-idx_a] += pos_top-pos_bot

                        if pos_top > 0 and pos_top > pos_bot:
                            pos_bot = max(pos_bot, 0)
                            # The following line is for debugging purposes
                            # to make sure all elements of the vector
                            # widxord_to_inverted are written and accounted
                            # and the fusion of temporary objects worked
                            if checksum:
                                if sum(widxord_to_inverted[pos_bot:
                                                           pos_top]) != 0:
                                    raise NameError('SUMZERO')
                            widxord_to_inverted[pos_bot:pos_top] = \
                                temp[:pos_top-pos_bot]
            used_mem += size_batch

            if checksum:
                zero_pos = np.where(widxord_to_inverted == 0)[0]
                if len(zero_pos) != 1 and self.verbose:
                    print(zero_pos)
                    raise NameError('SUMZERO')

            progressBar(widxord * len(self.temporary_filenames),
                        len(self.widxord_to_cumcount) *
                        len(self.temporary_filenames))

            if self.verbose:
                sys.stdout.write(f'\nSaving object {full_inv_idx}' + ' '*20)
                sys.stdout.flush()
            # Number of digits for writing the identifier of full_inverted.npy
            nzer = int(np.log10(
                self.widxord_to_cumcount[-1]//self.size_sindex+1
                ))+1
            np.save(model_folder / (('full_inverted%0' +
                                     '%i' % nzer +
                                     'i.npy') % full_inv_idx),
                    widxord_to_inverted)

            if self.verbose:
                print(f'\rSaving object {full_inv_idx} DONE')
                progressBar(widxord * len(self.temporary_filenames),
                            len(self.widxord_to_cumcount) *
                            len(self.temporary_filenames))
        # Deletion of all temporary files
        for filenames in self.temporary_filenames:
            os.remove(filenames[0])
            os.remove(filenames[1])
        if self.verbose:
            progressBar(1, 1)
            print()

    def word2vec(self,
                 num_iter=5,
                 num_features=100,
                 min_word_count=30,
                 context_size=7,
                 seed=23):
        """
        Trains word2vec on the corpus to create embeddings.

        Creates wvvocab_to_wvidx.pkl, wvidx_to_widxord.npy and
        embedding_matrix.npy.

        Uses self.allvocab_to_widx generated from import_documents()

        All model files will be saved in {self.root_folder}/models/

        Parameters
        ----------
        num_iter : int, optional
            The total number of iterations to be run. Note that the
            gensim model will need an extra first iteration to
            create the vocabulary.
            The default is 5.

        num_features: int, optional
            Number of embedding dimensions of the word2vec embeddings.
            The default is 100.

        min_word_count: int, optional
            Number of word count necessary for en embedding. If a word shows
            up less than min_word_count in the corpus, its embedding won't
            be generated. The default is 30.

        context_size: int, optional
            Number of words before and after a token in the copurs to build
            the context window necessary to create the Word2vec embeddings.
            In other words, tokens with similar words +-context_size around
            it in the corpus will have similar embeddings.
            The default is 7.

        seed: int, optional
            Seed for the random components of Word2vec. Default is 23.
        """
        model_folder = Path(self.root_folder) / 'models'

        if self.verbose:
            print(str(datetime.now())+'\t'+'Word2vec training')

        num_workers = multiprocessing.cpu_count()

        sentences = Sentences(self.pkl_folder,
                              self.text_treatment,
                              iter_total=num_iter+1,
                              verbose=self.verbose)

        # For progress bar through the iterator self.Sentences()
        model = Word2Vec(
            # bigrams[clauses_by_words],
            sentences,
            seed=seed,
            workers=num_workers,
            size=num_features,
            min_count=min_word_count,
            window=context_size,
            iter=num_iter
            )
        print()

        print(str(datetime.now())+'\t'+'Saving Word2vec')

        # Default gensim model save line.
        # Leads to bigger files on disk and is not used in this code
        # model.save('models/word2vec.model')

        self.embedding_matrix = np.empty((len(model.wv.vocab),
                                          num_features),
                                         dtype='f4')
        self.wvvocab_to_wvidx = {}
        self.wvidx_to_widxord = np.empty(len(model.wv.vocab),
                                         dtype=self.type_word)

        for i, word in enumerate(model.wv.vocab):
            self.wvvocab_to_wvidx[word] = i
            self.wvidx_to_widxord[i] = self.allvocab_to_widx[word]
            self.embedding_matrix[i] = model.wv[word]
        pickle.dump(self.wvvocab_to_wvidx,
                    open(model_folder / 'wvvocab_to_wvidx.pkl', 'wb'))
        np.save(model_folder / 'wvidx_to_widxord.npy', self.wvidx_to_widxord)
        np.save(model_folder / 'embedding_matrix.npy', self.embedding_matrix)

    def s2v_nopca(self,
                  sentence):
        """
        Transforms a sentence (list of strings) in a single vector.

        Does not use removal of the main PCA component for noise removal,
        as described in sentence2vec algorithm

        Uses self.wvvocab_to_wvidx, self.wvidx_to_widxord,
        self.widxord_to_cumcount, self.embedding_matrix generated from
        methods import_documents() and word2vec()

        Parameters
        ----------
        sentence: list of str
            List of word tokens in the sentence.

        Returns
        -------
            numpy array
            Vector of the embedding of the sentence
        """
        a = 1e-3
        embedding_size = self.embedding_matrix.shape[1]
        vs = np.zeros(embedding_size, dtype='f4')
        sentence_length = 0
        for word in sentence:
            if word in self.wvvocab_to_wvidx:
                sentence_length += 1
                wvidx = self.wvvocab_to_wvidx[word]
                widxord = self.wvidx_to_widxord[wvidx]
                if widxord == 0:
                    count = self.widxord_to_cumcount[widxord]
                else:
                    count = (self.widxord_to_cumcount[widxord] -
                             self.widxord_to_cumcount[widxord-1])

                # a_value is the smooth inverse frequency, (sif)
                # a_value could be also the tf-idf score of the word
                # word_frequency in document set goes higher, a_value is less.
                a_value = a / (a + count/self.widxord_to_cumcount[-1])

                # vs += sif * word_vector
                vs = np.add(vs, np.multiply(a_value,
                                            self.embedding_matrix[wvidx]))
        if sentence_length != 0:
            vs = np.divide(vs, sentence_length)  # weighted average
        return(vs)

    def Pages(self):
        """
        Generator over corpus documents, yielding pages, reading from disk.

        Yields
        ------
        list of str
            List of words in the page of the corpus
        """
        for fname in os.listdir(self.pkl_folder):
            pages_text = None
            if fname[-4:] == '.pkl':
                pages_text = pickle.load(
                    open(Path(self.pkl_folder) / fname, 'rb')
                    )
            elif fname[-4:] == '.txt':
                with open(open(Path(self.pkl_folder) / fname)) as f:
                    pages_text = [f.read()]
            if pages_text:
                for page in self.text_treatment.format_txt(pages_text):
                    yield ' '.join([cleanpassage(sent)
                                    for sent in page]).split()

    def sentence2vec(self):
        """
        Trains sentence2vec on the corpus to create sentence embeddings.

        The sentence embeddings are made on pages instead of sentences.
        The cosine similarity performed better this way.

        Creates page_vectors.npy and the hyper_param.pkl file containing
        self.size_sindex, embedding_size, self.type_word, self.type_invidx

        Uses self.pag_breaks, self.embedding_matrix
        generated from import_documents() and word2vec()

        All model files will be saved in {self.root_folder}/models/

        WARNING: Because of incremental PCA, this code uses numpy memmap
        without memory limitations and will work only on 64-bits operating
        systems for big corpuses (> 2Go = 1000M/EMBEDDING_DIM number of pages).
        """
        model_folder = Path(self.root_folder) / 'models'
        embedding_size = self.embedding_matrix.shape[1]

        print(str(datetime.now())+'\t'+'Importing corpus for sentence2vec')

        pages = self.Pages()
        page_vectors = np.memmap(model_folder / 'page_vectors.npy',
                                 dtype='f4',
                                 mode='w+',
                                 shape=(len(self.pag_breaks), embedding_size))
        for icount, page in enumerate(pages):
            if (icount % max(1, len(self.pag_breaks)//1000) == 0 and
                    self.verbose):
                progressBar(icount, len(self.pag_breaks))
            page_vectors[icount] = self.s2v_nopca(page)

        # Freeing the numpy memmap
        del page_vectors
        progressBar(1, 1)
        print()

        print(str(datetime.now())+'\t'+'Incremental PCA for sentence2vec')

        page_vectors = np.memmap(model_folder / 'page_vectors.npy',
                                 dtype='f4',
                                 mode='r',
                                 shape=(len(self.pag_breaks), embedding_size))
        ipca = IncrementalPCA(n_components=embedding_size,
                              batch_size=self.size_swords*4//embedding_size)
        ipca.fit(page_vectors)
        u = ipca.components_[0]  # the main PCA component for sentence2vec
        np.save(model_folder / 'pca_component.npy', u)

        # Freeing the numpy memmap and ipca module
        del ipca
        del page_vectors

        print(str(datetime.now())+'\t'+'Removal of main PCA for sentence2vec')

        proj_u = np.multiply(u, u.reshape(-1, 1))
        page_vectors = np.memmap(model_folder / 'page_vectors.npy',
                                 dtype='f4',
                                 mode='r+',
                                 shape=(len(self.pag_breaks), embedding_size))
        for i, page in enumerate(page_vectors):
            vsn = np.subtract(page, proj_u.dot(page))
            vsn_norm = np.linalg.norm(vsn)
            if vsn_norm != 0:
                vsn = np.divide(vsn, vsn_norm)
            page_vectors[i] = vsn

        # Freeing the numpy memmap and ipca module
        del page
        del page_vectors

        print(str(datetime.now())+'\t'+'Saving sentence2vec model files')

        # Like full_inverted in import_documents(), page_vectors will be
        # separated in sub-objects limited by self.size_sindex for future
        # use of numpy memmap on 32-bits machines
        # However, the previous code can't run on 32-bits machines so...

        page_vectors = np.memmap(model_folder / 'page_vectors.npy',
                                 dtype='f4',
                                 mode='r',
                                 shape=(len(self.pag_breaks), embedding_size))
        size_page = self.size_sindex//embedding_size
        mem_batches = ([i*size_page
                        for i in range((len(page_vectors)-1)//size_page+1)] +
                       [len(page_vectors)])
        nzer = int(np.log10(len(mem_batches)-1))+1
        for i in range(1, len(mem_batches)):
            np.save((model_folder /
                     (('page_vectors%0'+'%i' % nzer+'i.npy') % i)),
                    np.array(page_vectors[mem_batches[i-1]:mem_batches[i]]))

        # Freeing the numpy memmap
        del page_vectors
        pickle.dump([self.size_sindex,
                     embedding_size,
                     self.type_word,
                     self.type_invidx],
                    open(model_folder / 'hyper_params.pkl', 'wb'))
        os.remove(model_folder / 'page_vectors.npy')

    def compute(self):
        """
        Computes all index files of the corpus for the query tool.
        """
        self.import_documents()
        self.word2vec()
        self.sentence2vec()


class query_object():
    """
    Parameters
    ----------
    root_folder: str, optional
        Path to the main repository of the analysis, where model files
        and outputs will be saved.
        Default is '', which is current working directory.

    word_dictionaries: dict of str: set of str, optional
        Each dictionary (in the values of the object) is a set of strings
        containing all dictionary words for a specific language.
        Languages are identified by a 2-letters string (in the keys of the
        object) which are the results of the
        googletrans.Translator().detect().lang function.
        Dictionaries are used for filtering the gibberish from the pdf texts
        and recognize language.
        If None/empty, will not use those functionalities.
        Default is None.

    default_lang: str, optional
        Default language of the dictionaries. Corresponds to a key of the
        word_dictionaries object to default to if the language
        recognition fails.
        If None or empty, no dictionary will be used if the language
        recognition fails.
        Default is None.

    Attributes
    ----------
    text_treatment: text_treatment()
        text_treatment() class object containing word dictionaries
        for filtering text when reading.

    allvocab_to_widxord: dict of {str: int}
        Dictionary where the key is the string of a word in the corpus,
        and the value (word index) is an int index identifying the word.
        The index ordered by descending word count (widx_ord)
        The exact type of the value is identified via self.type_word

    idx_to_filename: dict of {int: str}
        Dictionary where the key is an index representing a document
        in the database, and the value is the string of the file name
        of the document

    filename_to_idx: dict of {str: int}
        Inversion of idx_to_filename
        Dictionary where the key is the string of the file name
        of the document, and the value is an index representing a document
        in the database.

    pag_breaks: list of int
        List of the index of words (self.sidx) corresponding to the
        beginning of a new page

    doc_breaks: list of int
        List of the index of words (self.sidx) corresponding to the
        beginning of a new page.
        Will eventually be transformed in the list of page numbers instead
        of word indexes.

    type_word: str
        Type of the word index encoding. If the number of distinct words
        tokens in the corpus is under 65535 (small corpuses), it should
        be 'u2'. If not, 'u4' (with an upper limit of 4294967295)
        Imported through hyper_params.pkl

    type_invidx: str
        Type of the inverted index self.sidx encoding. Normally 'u4'.
        If the whole corpus is more than 4294967295 words
        (approximately 16 Gb), should be changed for 'u8'.
        The implementation of 'u8' was never tested.
        Imported through hyper_params.pkl

    embedding_size: int
        Dimension size of the word embeddings.
        Imported through hyper_params.pkl

    size_sindex: int
        Variable limiting the size of the objects saved on the hard drive
        for the use of numpy memmap functions on older systems.
        32-bit systems cannot support numpy memmap on object bigger
        than 2 Gb, so size_sindex should not be higher than 500M
        (times type_invidx = 'u4' which equals 2 Gb)
        Imported through hyper_params.pkl

    widxord_to_cumcount:  numpy array
        Cumulative count of all words, ordered by descending word count.
        The word count of a word corresponding to index widx_ord is
        widxord_to_cumcount[widx_ord] - widxord_to_cumcount[widx_ord-1]

    wvvocab_to_wvidx: dict of {str:int}
        Dictionary where key=word string, where the word
        occurences > min_word_count specified in the word2vec calculation,
        and value=wvidx, word2vec_index an index corresponding to the
        embedding

    wvidx_to_widxord: numpy array
        Link between wvidx (word2vec_index) and widxord (word_index_ord)
        for the number of occurences calculation for sentence2vec

    embedding_matrix: 2D numpy memmap
        Numpy memory map pointing to the hard drive location of the
        matrix of vectors representing the embeddings for each word,
        which are indexed as wvidx, a specific word2vec index

    full_inverted_memmaps: dict of {str: 2D numpy memmap}
        Dictionary where key=file name (with extension) of the inverted
        index ('full_inverted{X}.npy'), and the value is the
        numpy memory map pointing to the hard drive location of the
        inverted index used for word searched.

    page_vectors_memmaps: list of 2D numpy memmap
        List where the index if the integer {X} corresponding to the
        'page_vectors{X}.npy' model files, and the objects are
        numpy memory map pointing to the hard drive location of the
        specified page_vectors.npy object containing sentence2vec
        embeddings of pages of the corpus

    pca_component: numpy array
        The main PCA component of the page sentence2vec raw embeddings,
        to compute the corrected sentence2vec embeddings

    size_accessload: int
        Parameter to optimize speed of importing page_vectors and calculating
        cosine similarity. Correspond to the number of vectors on which to do
        simultaneous cosine similarity in a single matrix. Good performance
        results are obtained for value of 100000.

    stopwords_keys: dict of {str: str}
        Corresponding language string between word_dictionaries object
        (from googletrans module) and stopwords modules from nltk.
    """

    def __init__(self,
                 root_folder='',
                 word_dictionaries=None,
                 default_lang=None):
        self.root_folder = root_folder
        self.text_treatment = text_treatment(
            word_dictionaries=word_dictionaries,
            default_lang=default_lang)
        self.stopwords_keys = {'en': 'english',
                               'fr': 'french'}

        # Import all model files here
        model_path = Path(self.root_folder) / 'models'

        self.idx_to_filename = pickle.load(
            open(model_path / 'idx_to_filename.pkl', 'rb')
            )
        self.filename_to_idx = {v: k for k, v in self.idx_to_filename.items()}

        self.pag_breaks = np.load(model_path / 'pag_breaks.npy')
        self.doc_breaks = np.load(model_path / 'doc_breaks.npy')
        (self.size_sindex,
         self.embedding_size,
         self.type_word,
         self.type_invidx) = \
            pickle.load(open(model_path / 'hyper_params.pkl', 'rb'))

        self.size_accessload = 100000

        self.allvocab_to_widxord = pickle.load(
            open(model_path / 'allvocab_to_widxord.pkl', 'rb')
            )

        # widxord_to_allvocab = \
        #     {v: k for k, v in self.allvocab_to_widxord.items()}
        self.widxord_to_cumcount = \
            np.load(model_path / 'widxord_to_cumcount.npy')

        self.full_inverted_memmaps = {}
        type_invidx_size = int(self.type_invidx[-1])
        full_inverted_list = sorted([model
                                     for model in os.listdir(model_path)
                                     if (model[:13] == 'full_inverted' and
                                         model[-4:] == '.npy')])
        for full_inverted_path in full_inverted_list:
            if full_inverted_path != full_inverted_list[-1]:
                offset = (os.path.getsize(model_path / full_inverted_path) -
                          self.size_sindex*type_invidx_size)
            else:
                offset = (os.path.getsize(model_path / full_inverted_path) -
                          ((self.widxord_to_cumcount[-1] % self.size_sindex) *
                           type_invidx_size))
            self.full_inverted_memmaps[full_inverted_path] = \
                np.memmap(model_path / full_inverted_path,
                          dtype=self.type_invidx,
                          mode='r',
                          offset=offset)

        self.page_vectors_memmaps = []
        type_pv_size = int('f4'[-1])
        size_page = (self.size_sindex//self.embedding_size)
        page_vectors_list = sorted([model
                                    for model in os.listdir(model_path)
                                    if (model[:12] == 'page_vectors' and
                                        model[-4:] == '.npy')])
        for page_vectors_path in page_vectors_list:
            if page_vectors_path != page_vectors_list[-1]:
                shape = (size_page, self.embedding_size)
                offset = (os.path.getsize(model_path / page_vectors_path) -
                          shape[0]*shape[1]*type_pv_size)
            else:
                shape = (len(self.pag_breaks) % size_page, self.embedding_size)
                offset = (os.path.getsize(model_path / page_vectors_path) -
                          shape[0]*shape[1]*type_pv_size)
            self.page_vectors_memmaps.append(
                np.memmap(model_path / page_vectors_path,
                          dtype='f4',
                          mode='r',
                          shape=shape,
                          offset=offset)
                )

        self.wvvocab_to_wvidx = pickle.load(
            open(model_path / 'wvvocab_to_wvidx.pkl', 'rb')
            )
        self.wvidx_to_widxord = np.load(model_path / 'wvidx_to_widxord.npy')

        type_wv_size = int('f4'[-1])
        shape = (len(self.wvvocab_to_wvidx), self.embedding_size)
        offset = (os.path.getsize(model_path / 'embedding_matrix.npy') -
                  shape[0]*shape[1]*type_wv_size)
        self.embedding_matrix = np.memmap(model_path / 'embedding_matrix.npy',
                                          dtype='f4',
                                          mode='r',
                                          shape=shape,
                                          offset=offset)
        self.pca_component = np.load(model_path / 'pca_component.npy')

    def answers_to_output(self,
                          answers,
                          n_results):
        """
        Generates a text string of all answers to output in a .csv format

        Parameters
        ----------
        answers: List of (List of int)*3
            The first object as outputted by retrieve_closest_passage().
            This first object is a nested list of the index of results.
            Each item [n] in the list contains all results from the most
            relevant collective agreement, in order of relevance.
            Each item contains 3 lists:
            [n][0]: The 1-length list of the most relevant result of this
                    particular collective agreement
            [n][1]: Every other possible relevent results in this particular
                    collective agreement (that matches words from the query)
                    in order of relevance
            [n][2]: All other results in the collective agreement
                    -including irrelevant ones-  in order of relevance.

        n_results: List of (int)*3
            The second object as outputted by retrieve_closest_passage().
            This second object is a list with the numbers of results.
            [0]: Number of results including all words of the query
            [1]: Number of results including at least one word of the query
            [2]: Number of documents in the database
                 (potential maximum number of results)

        Returns
        -------
        str
            Data to be exported in .csv format, containing the
            results in two column, the first being the filename
            of the document, the second being the page number
        """
        output_data = 'File name,Page numbers'
        for ans in answers[:n_results[0]]:
            doc_idx = bisect_right(self.doc_breaks, ans[0][0])-1
            lin = (self.idx_to_filename[doc_idx] + ',' +
                   str(ans[0][0]-self.doc_breaks[doc_idx]))
            for a in ans[1]:
                lin += '/' + str(a-self.doc_breaks[doc_idx])
            output_data += '\n'+lin
        return output_data

    def widxord_to_invidx(self,
                          widxord):
        """Returns the positions in full_inverted indexes containing a word

        Parameters
        ----------
        widxord: int
            The ordered index corresponding to the desired word, obtained
            with self.allvocab_to_widxord()

        Returns
        -------
        list of (str, int, int)
            Each element of the returned list is a tuple.
            The first element is the filename of the inverted index, which
            should be located in the models folder
            The second is the position in that file where the information
            about the word (the inverted indexes related to that word)
            begins
            The third element is the position in that file where
            the information about the word ends, exclusively
            Normally, this list is of length 1, because the inverted
            index is sorted. However, it's possible that a word information
            is separated over 2 index files if it began at the end of
            a file.
        """
        fix_list = []
        nzer = int(np.log10(
            self.widxord_to_cumcount[-1]//self.size_sindex+1
            ))+1

        full_inverted_path = 'full_inverted%0'

        if widxord == 0:
            idx_sa = 0
        else:
            idx_sa = self.widxord_to_cumcount[widxord-1]
        idx_sb = self.widxord_to_cumcount[widxord]-1
        full_inv_idx_a = ((idx_sa - idx_sa % self.size_sindex) //
                          self.size_sindex) + 1
        full_inv_idx_b = ((idx_sb - idx_sb % self.size_sindex) //
                          self.size_sindex) + 1
        for full_inv_idx in range(full_inv_idx_a, full_inv_idx_b):
            fix_list.append([
                (full_inverted_path+'%i' % nzer+'i.npy') % full_inv_idx_a,
                idx_sa % self.size_sindex,
                self.size_sindex
                ])
        if full_inv_idx_a != full_inv_idx_b:
            i_start = 0
        else:
            i_start = idx_sa % self.size_sindex
        fix_list.append([
            (full_inverted_path+'%i' % nzer+'i.npy') % full_inv_idx_b,
            i_start,
            idx_sb % self.size_sindex+1])
        return fix_list

    def query_to_s2v(self,
                     query):
        """
        Transforms a sentence in a single vector.

        Uses removal of the main PCA component for noise removal,
        as described in sentence2vec algorithm

        Uses self.wvvocab_to_wvidx, self.wvidx_to_widxord,
        self.widxord_to_cumcount, self.embedding_matrix,
        self.pca_component

        This code is very similar to create_index().s2v_nopca,
        could it be grouped?

        Parameters
        ----------
        query: str
            String corresponding to the sentence to vectorize

        Returns
        -------
        numpy array
            Vector of the embedding of the sentence
        """
        a = 1e-3
        vs = np.zeros(self.embedding_size, dtype='f4')
        sentence_length = 0
        for word in cleanpassage(query).split():
            if word in self.wvvocab_to_wvidx:
                sentence_length += 1
                wvidx = self.wvvocab_to_wvidx[word]
                widxord = self.wvidx_to_widxord[wvidx]
                if widxord == 0:
                    count = self.widxord_to_cumcount[widxord]
                else:
                    count = (self.widxord_to_cumcount[widxord] -
                             self.widxord_to_cumcount[widxord-1])
                # a_value is the smooth inverse frequency, (sif)
                # a_value could be also the tf-idf score of the word
                # word_frequency in document set goes higher,
                # a_value is less.
                a_value = a / (a + count/self.widxord_to_cumcount[-1])

                # vs += sif * word_vector
                vs = np.add(vs, np.multiply(a_value,
                                            self.embedding_matrix[wvidx]))
        if sentence_length != 0:
            vs = np.divide(vs, sentence_length)  # weighted average

        u = self.pca_component
        proj_u = np.multiply(u, u.reshape(-1, 1))
        vsn = np.subtract(vs, proj_u.dot(vs))
        vsn_norm = np.linalg.norm(vsn)
        if vsn_norm != 0:
            vsn = np.divide(vsn, vsn_norm)
        return vsn

    def query_to_answers(self,
                         query_idx):
        """
        Returns page numbers of pages containing words from the query.

        Uses pag_breaks, full_inverted_memmaps, widxord_to_cumcount

        Takes about 1 sec per 200 000 occurences of a query word in the
        corpus, per query word.

        Parameters
        ----------
        query_idx: List of list of int
            List of list of word indexes (widxord) in the query
            A sublist of length > 1 means an exact expression in
            quotation marks

        Returns
        -------
        List of list of int
            For each word (widxord) or group of words in exact expression,
            returns the list of all page numbers  (as indexed in pag_breaks)
            containing that word or exact expression
        """
        answers_per_word = []
        for words_idx in query_idx:

            # If exact expression (in quotation marks)
            if len(words_idx) > 1:
                # find rarest word
                rare_word_idx = -1
                rare_rarity = self.widxord_to_cumcount[-1]
                for word_idx, widxord in enumerate(words_idx):
                    rarity = sum(
                        dat[2] - dat[1]
                        for dat in self.widxord_to_invidx(widxord)
                        )
                    if rarity < rare_rarity:
                        rare_word_idx = word_idx
                        rare_rarity = rarity

                # find results for rarest word first (for quicker results)
                rare_pags = []
                rare_poss = []
                for full_inv_idx, start_i, end_i in \
                        self.widxord_to_invidx(words_idx[rare_word_idx]):
                    for inverted in self.full_inverted_memmaps[
                            full_inv_idx
                            ][start_i:end_i]:
                        rare_pag = bisect_right(self.pag_breaks,
                                                inverted) - 1
                        rare_pos = inverted - self.pag_breaks[rare_pag]
                        rare_pags.append(rare_pag)
                        rare_poss.append(rare_pos)

                rare_poss_np = np.array(rare_poss, dtype=self.type_invidx)

                # find results that match the other words of the
                # exact expression in the right order in the document
                answers_list = []
                for word_idx, widxord in [
                        (i, w) for i, w in enumerate(words_idx)
                        if i != rare_word_idx
                        ]:
                    if self.type_invidx == 'u4':
                        word_idx_np = np.uint32(word_idx)
                        rare_word_idx_np = np.uint32(rare_word_idx)
                    else:
                        word_idx_np, rare_word_idx_np = \
                            np.array([word_idx, rare_word_idx],
                                     dtype=self.type_invidx)
                    idx_rare = 0
                    answer = []
                    rare_inverted = \
                        (rare_poss_np[idx_rare] +
                         self.pag_breaks[rare_pags[idx_rare]] -
                         rare_word_idx_np +
                         word_idx_np)
                    for inverted in [
                            i for i in self.full_inverted_memmaps[
                                full_inv_idx][start_i:end_i]
                            for full_inv_idx, start_i, end_i in
                            self.widxord_to_invidx(widxord)
                            ]:

                        while inverted > rare_inverted:
                            idx_rare += 1
                            if idx_rare != len(rare_pags):
                                rare_inverted = \
                                    (rare_poss_np[idx_rare] +
                                     self.pag_breaks[rare_pags[idx_rare]] -
                                     rare_word_idx_np +
                                     word_idx_np)
                            else:
                                rare_inverted = \
                                    (self.widxord_to_cumcount[-1] +
                                     np.uint32(1))
                        if inverted == rare_inverted:
                            pag = bisect_right(self.pag_breaks,
                                               inverted)-1
                            pos = inverted - self.pag_breaks[pag]
                            if (idx_rare != len(rare_pags) and
                                    pag == rare_pags[idx_rare] and
                                    (pos - word_idx) == (rare_poss[idx_rare] -
                                                         rare_word_idx)):
                                answer.append(idx_rare)
                    answers_list.append(answer)
                answer = set(answers_list[0])
                for ans in answers_list[1:]:
                    answer = answer.intersection(ans)
                answers_per_word.append(set([rare_pags[ans]
                                             for ans in answer]))

            # If single word
            elif len(words_idx) == 1:
                answers_per_word.append(
                    set([bisect_right(self.pag_breaks, inverted) - 1
                         for full_inv_idx, start_i, end_i in
                         self.widxord_to_invidx(words_idx[0])
                         for inverted in self.full_inverted_memmaps[
                             full_inv_idx][start_i:end_i]])
                    )
        return answers_per_word

    def retrieve_closest_passages(self,
                                  query,
                                  from_pdfs=None,
                                  trans_flag=True,
                                  time_flag=False):
        """
        Returns the results (pages) in the corpus that match the query.

        A query string can include quotation marks to signify an exact
        expression comprised of multiple work tokens.

        Each possible result (pages, in that case) are grouped together
        per document. Each document will be ranked according to
        its best result (page). Each result is attributed two metrics:
        a) the amount of word token (excluding common stopwords)
           that are textually present in the result
        b) the cosine similarity between the sentence2vec
           embeddings (with main component of PCA taken off) of the
           query and the result.
        Each possible result is ordered by a), and THEN b).
        Then, for convenience, all other possible results (pages) from the
        document are ordered through the same process. Therefore, it is easy
        to access other pages/results from the documents with the top overall
        match with the query.
        More comments related to how the sorting is done below.

        NOTE: The bottleneck of this code is NOT the calculation of cosine
        similarities with all pages in the database, but rather the
        ordering of the results.

        Uses attributes doc_breaks, pag_breaks, idx_to_filename,
        filename_to_idx, full_inverted_memmaps, page_vectors_memmaps
        widxord_to_cumcount,
        size_sindex, embedding_size, type_invidx, size_accessload

        Parameters
        ----------
        query: str
            The query in string format

        from_pdfs: List of str, optional
            List of pdf filenames (without extensions) to perform the query in.
            None is in all pdfs. Default is None

        trans_flag: bool, optional
            True means that the query will be translated to all languages
            in self.text_treatement.word_dictionaries.keys()
            Default is True.

        time_flag: bool, optional
            If true, prints the time the query took in the console,
            Default is False

        Returns
        -------
        (List of (List of int)*3, List of (int)*3)
            The first object is a nested list of the index of results.
            Each item [n] in the list contains all results from the most
            relevant collective agreement, in order of relevance.
            Each item contains 3 lists:
            [n][0]: The 1-length list of the most relevant result of this
                    particular collective agreement
            [n][1]: Every other possible relevent results in this particular
                    collective agreement (that matches words from the query)
                    in order of relevance
            [n][2]: All other results in the collective agreement
                    -including irrelevant ones-  in order of relevance.

            The second object is a list with the numbers of results.
            [0]: Number of results including all words of the query
            [1]: Number of results including at least one word of the query
            [2]: Number of documents in the database
                 (potential maximum number of results)
        """
        # For calculation of time the query took
        start_time = time.time()

        # pdf_list = clean set of pdfs filenames to search in.
        if from_pdfs is not None:
            pdf_list = set([x.replace('-', '') for x in from_pdfs])
        else:
            pdf_list = set(self.idx_to_filename.values())

        # First object to return
        answers = []
        # Second object to return
        n_results = [[], [], []]

        # Hardcoded: If query is empty, return all results (about 1sec)
        if len(cleanpassage(query)) == 0:
            ans = []
            doc_idx = 0
            for filename in pdf_list:
                doc_idx = self.filename_to_idx[filename]
                pag_idx_a = self.doc_breaks[doc_idx]
                pag_idx_b = self.doc_breaks[doc_idx+1]
                ans = [[int(pag_idx_a)],
                       [pag_idx for pag_idx in range(pag_idx_a+1, pag_idx_b)],
                       []]
                answers.append(ans)
            n_results[0] = len(pdf_list)
            n_results[1] = len(pdf_list)
            n_results[2] = len(pdf_list)
            return sorted(answers, key=lambda x: x[0][0]), n_results

        # doc_list = clean set of document index of the documents in pdf_list
        doc_list = set([self.filename_to_idx[filename]
                        for filename in pdf_list
                        if filename in self.filename_to_idx])

        # from_pdfs = None is the flag indicating there's no filter
        if len(doc_list) == len(self.doc_breaks)-1:
            from_pdfs = None

        # QUERY TREATMENT SECTION (about 300-500ms)
        # Clean quotation marks format
        for c in '“”«»':
            query = query.replace(c, '"')

        # Translation of the query in query_trans and language detection TO FIX
        query_lang = self.text_treatment.translator.detect(query[:5000]).lang
        all_lang = self.text_treatment.word_dictionaries.keys()
        if query_lang not in all_lang:
            if self.text_treatment.default_lang:
                query_lang = self.text_treatment.default_lang
            else:
                query_lang = None

        # List of all queries in all languages
        list_query = [query]
        list_lang = [query_lang]
        if trans_flag and query_lang:
            for dest in [lang for lang in all_lang
                         if lang != query_lang]:
                list_query.append(
                    self.text_treatment.translator.translate(
                        query[:5000], dest=dest).text)
                list_lang.append(dest)

        # List of list of word indexes (widxord) in the query
        # A list of length > 1 means an exact expression in quotation marks
        list_query_idx = []
        # Transforming query in sentence2vec vector
        list_query_vect = []
        for j, query_trans in enumerate(list_query):
            query_idx_trans = []
            # Makes sure to have an even number of quotation marks.
            # If not, delete the last one
            if query_trans.count('"') % 2 != 0:
                idx = query_trans.rfind('"')
                list_query[j] = query_trans[:idx]+query_trans[idx+1:]

            for i, s in enumerate(query_trans.split('"')):
                if i % 2 == 0:
                    if list_lang[j]:
                        stopwords_set_trans = set(stopwords.words(
                            self.stopwords_keys[list_lang[j]]
                            ))
                    else:
                        stopwords_set_trans = {}
                    query_idx_trans += \
                        [[self.allvocab_to_widxord[word]]
                         for word in cleanpassage(s).split()
                         if (word in self.allvocab_to_widxord and
                             word not in stopwords_set_trans)]
                # If exact expression (between quotation marks)
                else:
                    query_idx_trans.append(
                        [self.allvocab_to_widxord[word]
                         for word in cleanpassage(s).split()
                         if word in self.allvocab_to_widxord])
            list_query_idx.append(query_idx_trans)
            list_query_vect.append(self.query_to_s2v(query))

        # CREATING RESULTS OBJECT
        # This is the longest part of the code because it contains multiple
        # sortings of numpy array which is a long process. Individual sorting
        # times are indicated below

        # Creating object, filling data about pages, documents and greps
        # (about 800ms per 1Go of page data vectors = 3M pages with 100
        # dimensions)

        # 'page' is the page index of the corpus. Results contains data for all
        # pages in the corpus, but will be filtered eventually with from_pdfs
        # if from_pdfs!=None
        # 'doc' is the document index corresponding to the page
        # 'grep' is the proportion of the keywords of the query that are
        # textually present in the page
        # 'distance' is the cosine similarity between the page and the query
        # 'order' is the order of the document if all pages from the same
        # document are grouped together and documents are ordered according
        # to the result of its best page
        # results are ordered by 'order', then 'grep', then 'distance'

        results = np.zeros(shape=len(self.pag_breaks),
                           dtype=[('doc', self.type_invidx),
                                  ('page', self.type_invidx),
                                  ('order', self.type_invidx),
                                  ('grep', 'f4'),
                                  ('distance', 'f4')])
        results['page'] = np.arange(len(self.pag_breaks))
        docs = -np.ones(len(self.pag_breaks), dtype=self.type_invidx)

        for doc in doc_list:
            docs[self.doc_breaks[doc]: self.doc_breaks[doc+1]] = \
                doc*np.ones(self.doc_breaks[doc+1]-self.doc_breaks[doc],
                            dtype=self.type_invidx)
        results['doc'] = docs

        grep = np.zeros(shape=(len(list_query), len(self.pag_breaks)),
                        dtype='f4')
        for j, query_idx_trans in enumerate(list_query_idx):
            for answer in self.query_to_answers(query_idx_trans):
                for ans in answer:
                    grep[j][ans] += 1/len(query_idx_trans)
            results['grep'] = grep.max(axis=0)
        # Freeing memory
        del grep

        # Calculating distances - sentence2vec
        # (about 200ms per 1Go of page data vectors = 3M pages
        # with 100 dimensions)
        dist = -np.ones(shape=(len(list_query), len(self.pag_breaks)),
                        dtype='f4')
        size_page = self.size_sindex//self.embedding_size
        for pvidx, page_vectors in enumerate(self.page_vectors_memmaps):
            mem_batches = ([i*self.size_accessload
                            for i in range((len(page_vectors) - 1) //
                                           self.size_accessload + 1)] +
                           [len(page_vectors)])
            for i in range(len(mem_batches)-1):
                idx_a = mem_batches[i]
                idx_b = mem_batches[i+1]
                for j, query_vect_trans in enumerate(list_query_vect):
                    dist[j][pvidx*size_page+idx_a:pvidx*size_page+idx_b] = \
                        page_vectors[idx_a:idx_b].dot(query_vect_trans)

        results['distance'] = dist.max(axis=0)
        # Freeing memory
        del dist

        # Filtering for from_pdfs
        # (up to 400ms per 1Go of page data vectors = 3M pages
        # with 100 dimensions)
        if from_pdfs is not None:
            results = results[np.where(
                docs != -np.ones(1, dtype=self.type_invidx)
                )[0]]
        # Freeing memory
        del docs

        # First ordering:
        # (1,5s for 1Go of page data vectors = 3M pages with 100 dimensions)
        results = results[np.lexsort((-results['distance'], -results['grep']))]

        # Finding order of documents for grouping documents together
        # (1,5s for 1Go of page data vectors = 3M pages with 100 dimensions)
        order = -np.ones(len(results), dtype='f4')
        doc_order = {}
        for i, doc in enumerate(results['doc']):
            if doc not in doc_order:
                doc_order[doc] = len(doc_order)
            order[i] = doc_order[doc]
        results['order'] = order
        # Freeing memory
        del order

        # Second ordering:
        # (1,2s for 1Go of page data vectors = 3M pages with 100 dimensions)
        results = results[np.lexsort((-results['distance'],
                                      -results['grep'],
                                      results['order']))]

        # Final answers object + nresults:
        # (700ms for 1Go of page data vectors = 3M pages with 100 dimensions)
        answers = []
        n_results = [[], [], []]
        docs = results['doc']
        pages = results['page']
        greps = results['grep']
        ia = 0
        for ib in np.concatenate((
                (np.where(np.ediff1d(results['doc']) != 0)[0]+1),
                np.array([len(results)])
                )):
            filename = self.idx_to_filename[docs[ia]]
            n_results[2].append(filename)
            if greps[ia] > 0:
                n_results[1].append(filename)
            if greps[ia] == 1:
                n_results[0].append(filename)
            delim = np.searchsorted(-greps[ia:ib], 0)
            ans = pages[ia:ib].tolist()
            answers.append([ans[0:1], ans[1:delim], ans[delim:]])
            ia = ib

        n_results[0] = len(n_results[0])
        n_results[1] = len(n_results[1])
        n_results[2] = len(n_results[2])

        # Printing result time
        if time_flag:
            print('Search took %.1f sec' % (time.time()-start_time))
        return answers, n_results

    def print_closest_passages(self,
                               query,
                               answers,
                               num_answers=None,
                               trans_flag=True):
        """
        Generates info necessary to print  results in a UI

        Uses idx_to_filename, doc_breaks, text_treatment

        The query treatment part is taken from retrieve_closest_passage() and
        should be it's own thing. Also, language detection is used throughout
        this and retrieve_closest_passage().

        Parameters
        ----------
        query : str
            The query in string format.

        answers : List of int
            The list of answers page indexes as returned by
            retrieve_closest_passages(query, from_pdfs),
            but in a single list

        num_answers : int, optional
            Number of displayed answers.
            None means all answers are going to be displayed, which is a very
            long calculation. The default is None.

        trans_flag: bool, optional
            True means that the query will be translated to all languages
            in self.text_treatement.word_dictionaries.keys()
            Default is True.

        Returns
        -------
        List of (str, str, List of [int, int])
            Each object of the list is the three elements of a result
            [n][0]: The name of the document/answer in format
                    [filename]-[page number]
            [n][1]: Raw textual data of the page containing the answer,
                    obtained from the .pkl of the pdf
            [n][2]: List of (int, int) corresponding to the index pairs
                    (i_start, i_stop) of text to highlight in a UI display,
                    which should be the location of the word token included
                    in the query.
                    The indexes should be used as text[i_start:i_strop]
        """

        if num_answers is None:
            num_answers = len(answers)

        # Query treatment
        # Clean quotation marks format
        for c in '“”«»':
            query = query.replace(c, '"')

        # Translation of the query in query_trans and language detection TO FIX
        query_lang = self.text_treatment.translator.detect(query[:5000]).lang
        all_lang = self.text_treatment.word_dictionaries.keys()
        if query_lang not in all_lang:
            if self.text_treatment.default_lang:
                query_lang = self.text_treatment.default_lang
            else:
                query_lang = None

        # List of all queries in all languages
        list_query = [query]
        list_lang = [query_lang]
        if trans_flag and query_lang:
            for dest in [lang for lang in all_lang
                         if lang != query_lang]:
                list_query.append(
                    self.text_treatment.translator.translate(
                        query[:5000], dest=dest).text)
                list_lang.append(dest)

        metalist = []
        for pag in answers[:num_answers]:
            doc_idx = bisect_right(self.doc_breaks, pag) - 1
            metalist.append((doc_idx, pag-self.doc_breaks[doc_idx]))
        pdflist = list(set([meta[0] for meta in metalist]))

        textCA = []
        for pdf in pdflist:
            fname = self.idx_to_filename[pdf]+'.pkl'
            pages_text = pickle.load(
                open(Path(self.root_folder) / 'txt-pdftotext' / fname, 'rb')
                )
            if pages_text:
                textCA.append(self.text_treatment.format_txt(pages_text))

        results = []
        for meta in metalist:
            text = '\n'.join(textCA[pdflist.index(meta[0])][meta[1]])
            cleantext = unidecode(text).lower()
            highlight_pos = []

            if len(cleanpassage(query)) != 0:
                # Language detection of the answers text is actually this
                # function's bottleneck
                if trans_flag:
                    lang_scores = [0]*len(list_query)
                    for j, query_trans in enumerate(list_query):
                        w_list = cleanpassage(query_trans).split()
                        for word in w_list:
                            if list_lang[j]:
                                stopwords_set_trans = set(stopwords.words(
                                    self.stopwords_keys[list_lang[j]]
                                    ))
                            else:
                                stopwords_set_trans = {}
                            if (list_lang[j] and
                                    word not in stopwords_set_trans and
                                    word in cleantext):
                                lang_scores[j] += 1
                        lang_scores[j] = lang_scores[j]/len(w_list)

                    max_idx = [i for i, s in enumerate(lang_scores)
                               if s == max(lang_scores)]
                    if len(max_idx) == 1:
                        text_lang = list_lang[max_idx[0]]
                    else:
                        text_lang = self.text_treatment.translator.detect(
                            cleantext[:5000]
                            ).lang
                        if text_lang not in all_lang:
                            if self.text_treatment.default_lang:
                                text_lang = self.text_treatment.default_lang
                            else:
                                text_lang = None
                else:
                    text_lang = query_lang

                if text_lang and query_lang:
                    query_trans = list_query[list_lang.index(text_lang)]
                else:
                    query_trans = query

                # Makes sure to have an even number of quotation marks.
                # If not, delete the last one
                if query_trans.count('"') % 2 != 0:
                    idx = query_trans.rfind('"')
                    query_highlight = query_trans[:idx]+query_trans[idx+1:]
                else:
                    query_highlight = query_trans

                query_words = []
                if text_lang:
                    stopwords_set_trans = set(stopwords.words(
                        self.stopwords_keys[text_lang]
                        ))
                else:
                    stopwords_set_trans = {}
                for i, s in enumerate(query_highlight.split('"')):
                    if i % 2 == 0:
                        for word in cleanpassage(s).split():
                            if (word not in stopwords_set_trans and
                                    len(word) >= 3):
                                query_words.append(word)
                    # If exact expression
                    else:
                        query_words.append(cleanpassage(s))

                for word in query_words:
                    highlight_pos += ngram_positions(word, cleantext)
                highlight_pos = fuse_positions(highlight_pos)
                for i in range(len(highlight_pos)):
                    highlight_pos[i] = span_shift(highlight_pos[i],
                                                  cleantext)
            results.append([self.idx_to_filename[meta[0]]+'-%i' % meta[1],
                            text,
                            highlight_pos])
        return results


if __name__ == '__main__':
    # Treatment of pdftotext on the pdf folder
    # PT = pdf_treatment(exec_path='xpdf-tools-mac-4.02/bin64/')
    # PT.pdftotext('test_data/pdfs/')

    # Dictionaries import
    word_dictionaries = import_dictionaries()

    # Creating index
    """
    CI = create_index('test_data/txt-pdftotext',
                      word_dictionaries=word_dictionaries,
                      default_lang='en')
    CI.compute()
    """

    Q = query_object('test_data',
                     word_dictionaries=word_dictionaries,
                     default_lang='en')
    query = 'présente convention'
    answers_rich, n_results = Q.retrieve_closest_passages(query)
    answers = [a[0][0] for a in answers_rich]
    results = Q.print_closest_passages(query, answers, num_answers=10)
    for filecode, text, highlight_pos in results:
        print(filecode)
        print()
        for i_start, i_stop in highlight_pos:
            print(text[i_start:i_stop])
        print()
