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
    """

    def __init__(self,
                 english_words=None,
                 french_words=None):
        self.english_words = english_words
        self.french_words = french_words

    def format_txt(self,
                   pages_text,
                   n_ignore=1):
        """
        Formats a list of strings from pages of a pdf into a list of sentences.

        TO BE FILLED: grouping sentences together, for legal documents

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