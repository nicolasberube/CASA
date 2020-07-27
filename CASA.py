# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:28:45 2017

@author: nicolas.berube

"""
import os
import re
from unidecode import unidecode
import subprocess
import pickle
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import numpy as np
from bisect import bisect_left, bisect_right


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


class pdf_treatment():
    def __init__(self,
                 exec_path=''):
        """
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


class create_index():
    """
    Functions to create the index files for the search engine.

    Parameters
    ----------
    english_words: set of str
        String containing all dictionary words for the English language

    french_words: set of str
        String containing all dictionary words for the French language

    Attributes
    ----------
    allvocab_to_widx: dict of {str: int}
        Dictionary where the key is the string of a word in the corpus,
        and the value (word index) is an int index identifying the word.
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
        This will take approximately to 4 Gb in RAM with the argsort function

    size_sindex: int
        Variable limiting the size of the objects saved on the hard drive
        for the use of numpy memmap functions on older systems.
        32-bit systems cannot support numpy memmap on object bigger
        than 2 Gb, so size_sindex should not be higher than 500M
        (times type_invidx = 'u4' which equals 2 Gb)

    widx_to_cumcount:
    """

    def __init__(self,
                 english_words=None,
                 french_words=None):
        self.english_words = english_words
        self.french_words = french_words

        # Meta-parameters
        self.sidx = 0
        self.n_swords = 1
        self.type_word = 'u4'
        self.type_invidx = 'u4'
        self.size_swords = 250000000
        self.size_sindex = 500000000

        # Model files
        self.allvocab_to_widxord = {}
        self.idx_to_filename = {}
        self.widx_to_cumcount = None
        self.pag_breaks = []
        self.doc_breaks = []

        # Variable used during the calculation
        self.allvocab_to_widx = {}
        self.widx_to_count_total = {}
        self.widx_to_count_temp = {}
        self.sentence_words = None
        self.temporary_filenames = []

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

        #language recognition?
        #if filename.split('.')[-2][-1] == 'a':
        #    dict_words = english_words
        #elif filename.split('.')[-2][-1] == 'c':
        #    dict_words = french_words
        #else:
        #    raise NameError('langue introuvable')
        dict_words = {}

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

    def saving_objects(self):
        """
        Method that, once the RAM is full, will create an inverted index
        with the collected data so far and will save a
        temporary_inverted.npy and a temporary_count.npy object on the disk

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

        print(f'\rSaving object {self.n_swords}' + ' '*20)
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
        np.save(f'models/temporary_inverted{self.n_swords}.npy',
                words_inverted)
        np.save(f'models/temporary_count{self.n_swords}.npy',
                widx_to_cumcount)
        self.temporary_filenames.append(
            [f'models/temporary_inverted{self.n_swords}.npy',
             f'models/temporary_count{self.n_swords}.npy']
            )
        # Updating the slice index
        self.n_swords += 1
        # Reseting the sentence_words slice
        self.sentence_words = np.empty(self.size_swords,
                                       dtype=self.type_word)
        print('\r\tDone', ' '*20)

    def import_documents(self,
                         input_folder,
                         checksum=False):
        """
        Reads all document text files and create temporary index files.

        Will save the temporary_inverted.npy and temporary_count.npy
        files on the disk, as well as doc_breaks.npy, pag_breaks.npy,
        idx_to_filename.pkl, widx_to_cumcount.npy and
        allvocab_to_widxord.pkl

        Parameters
        ----------
        input_folder: str
            Path of the folder containing the corpus documents.
            Each corpus document is a pickle file containing a
            list of list of strings, where a string is a single sentence,
            grouped in a single list per page, and grouped together for the
            document.
            Corresponds the output_folder of the pdf_treatment().pdftotext()
            function.

        checksum: bool, optional
            If True, will run extra checks to make sure the code is working
            properly when building the full inverted index from temporary
            files. Will slow down the code. Default is False.
        """
        print('Step 1: First pass of the corpus')
        self.sentence_words = np.empty(self.size_swords,
                                       dtype=self.type_word)

        # Iterating over all documents
        all_names = sorted(os.listdir(input_folder))
        for name in tqdm(all_names):
            if name[-4:] == '.pkl':
                path = os.path.join(input_folder, name)
                filename, file_extension = os.path.splitext(path)
                filename = filename.split("/")[-1]

                # Updating the filenames of the documents
                self.idx_to_filename[len(self.idx_to_filename)] = filename

                # Updating the indexes for the document breaks
                self.doc_breaks.append(self.sidx)

                # Importing the text data from the pickle
                if filename.split('.')[-1] == 'pkl':
                    pages_text = pickle.load(open(filename, 'rb'))
                elif filename.split('.')[-1] == 'txt':
                    with open(filename) as f:
                        pages_text = [f.read()]
                else:
                    raise NameError('filetype introuvable')

                for page in self.format_txt(pages_text):
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
        np.save('models/pag_breaks.npy', self.pag_breaks)
        self.doc_breaks = np.array(self.doc_breaks, dtype=self.type_invidx)
        self.doc_breaks = np.concatenate((
            np.searchsorted(self.pag_breaks,
                            self.doc_breaks).astype(self.type_invidx),
            np.array([len(self.pag_breaks)], dtype=self.type_invidx)
            ))
        np.save('models/doc_breaks.npy', self.doc_breaks)
        pickle.dump(self.idx_to_filename,
                    open('models/idx_to_filename.pkl', 'wb'))

        print('Step 2: Indexing of the corpus')
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
                    open('models/allvocab_to_widxord.pkl', 'wb'))

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

        np.save('models/widxord_to_cumcount.npy',
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
            used_mem = min(used_mem+self.size_sindex,
                           self.widxord_to_cumcount[idx_b-1])
            mem_batches.append([idx_a, idx_b])
            idx_a = bisect_right(self.widxord_to_cumcount,
                                 used_mem)
        used_mem = 0

        # Loop over full_inverted[X].npy objects
        # Every loop will import all temporary object sequentially and grab
        # the necessary information to build the full_inverted.npy object
        # The size of full_inverted.npy is capped by size_sindex
        # This is mostly a reordering/defragmentation of the temporary objects
        for full_inv_idx, (idx_a, idx_b) in enumerate(mem_batches):
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
                self.widxord_to_inverted = \
                    np.zeros(self.size_batch, dtype=self.type_invidx)

            # Importing relevant data in the temporary objects
            for i, filenames in enumerate(self.temporary_filenames):

                # Inverted index, listing all the positions of the words
                temporary_inverted = np.load(filenames[0])
                # Cumulative count of word, sorted by old word index
                # corresponding to the positions of the inverted index
                # of that specific word
                temporary_count = np.load(filenames[1])

                for widxord in range(idx_a, idx_b):
                    #FOR MANUAL TQDM
                    #if widxord % max(1, len(self.widxord_to_cumcount)//1000) == 0:
                    #    progressBar(widxord+i*(idx_b-idx_a)+idx_a*(len(temporary_filenames)-1),len(widxord_to_cumcount)*len(temporary_filenames))

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
                print(np.where(widxord_to_inverted == 0))

            print('\rSaving object', ' '*20)
            #FOR MANUAL TQDM
            # progressBar(widxord*len(temporary_filenames),len(widxord_to_cumcount)*len(temporary_filenames))

            # Number of digits for writing the identifier of full_inverted.npy
            nzer = int(np.log10(
                self.widxord_to_cumcount[-1]//self.size_sindex+1
                ))+1
            np.save(('models/full_inverted%0' +
                     '%i' % nzer +
                     'i.npy') % full_inv_idx,
                    self.widxord_to_inverted)
            print('\r\tDone', ' '*20)
            #FOR MANUAL TQDM
            #progressBar(widxord*len(temporary_filenames),len(widxord_to_cumcount)*len(temporary_filenames))

        # Deletion of all temporary files
        for filenames in self.temporary_filenames:
            os.remove(filenames[0])
            os.remove(filenames[1])
        #progressBar(1,1)
        #print()


if __name__ == '__main__':
    # Treatment of pdftotext on the pdf folder
    # PT = pdf_treatment(exec_path='xpdf-tools-mac-4.02/bin64/')
    # PT.pdftotext('test_data/pdfs/')

    filename = 'test_data/txt-pdftotext/1512101c.pkl'
    if filename.split('.')[-1] == 'pkl':
        pages_text = pickle.load(open(filename, 'rb'))
    elif filename.split('.')[-1] == 'txt':
        with open(filename) as f:
            pages_text = [f.read()]
    else:
        raise NameError('filetype introuvable')

    CI = create_index()
    a = CI.format_txt(pages_text)