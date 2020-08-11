# CASA

This code is a search engine that can search through a database of pdf based on a mix of exact token matching, as well as word and sentence embeddings. The query is also translated into multiple languages to match a corpus composed of different language (French and English, in our test case). It is also possible to search for exact expression of multiple word tokens when they are put between quotation marks in the query.

The purpose of this code was to search through a database of legal documents (in this case, collective agreements), where exact matching of legal terms is heavily linked to the users desire. Exact token matching was implemented after the results of the user beta-testing of the code.

The search is designed to be as fast as possible, using as less memory as possible. The downside is big index files that are read from disk as numpy arrays with numpy memmaps (where each word token is encoded as an integer for efficiency), but don't have to be loaded in memory. What is loaded in memory are clever metaindexation information which allows us to look-up the necessary information from disk very efficiently.

Also based on user testing results, a "sentence" (possible result of the query) has been defined as a whole page of a document, as opposed to a single legal clause, since the former led to better search results. Single legal clauses were still used as word groups to train custom word embeddings on the database corpus (approximately 50 000 pdfs) based on the word2vec approach, but whole pages were used for sentence embeddings.

The code was developed as a series of function to be interacted with from a javascript UI. Most function currently return a list of different objects that are compatible with json, but it is obviously not as ideal as returning a *results* object with the desired information as attributes, which still needs to be done. Such objects include meta-information about the document, textual data of the results and integer position of the text to highlight in the UI.

# How to use

The first step is to create a folder containing machine readable textual data of the pdf. It assumes that the pdfs are properly encapsulated, and that textual information can be extracted with pdf2text tools. If not, optical character recognition (like tesseract) should be used instead.

This function will create pickle files of the textual data for each pdfs, stored in a *txt-pdftotext/* folder.
```
# Step 1: Treatment of pdftotext on the pdf folder
PT = pdf_treatment(exec_path='xpdf-tools-mac-4.02/bin64/')
PT.pdftotext('pdfs/')
```

The following steps will needs *word_dictionaries* objects, containing a list of all words (from a dictionary) for all languages that will be present in the corpus, which is used to filter gibberish. The word dictionaries are optional, and if empty, these functions will simply not be used. However, the *word_dictionaries* object preferably needs to include keys, which are the list of all languages, as well as a default language to revert to.

The second step is to create the index files necessary for the queries. It will create index files in a *models/* folder.
```
# Imports word dictionaries object
word_dictionaries = import_dictionaries()

# Step 2: creating index files
CI = create_index('txt-pdftotext',
                  word_dictionaries=word_dictionaries,
                  default_lang='en')
CI.compute()
```

Once those steps have been completed, everything is ready for the queries. Queries are made by loading a query_object() into memory (which is only about 100 Mb for a 10 Gb corpus) and then calling its functions (from a flask app for example) and collecting the results.

```
Q = query_object(word_dictionaries=word_dictionaries,
                 default_lang='en')
query = 'test query'
answers_rich, n_results = Q.retrieve_closest_passages(query)
answers = [a[0][0] for a in answers_rich]
results = Q.print_closest_passages(query, answers, num_answers=10)
```
Each object of the *results* list is the three elements of a result
[n][0]: The name of the document/answer in format
        [filename]-[page number]
[n][1]: Raw textual data of the page containing the answer,
        obtained from the .pkl of the pdf
[n][2]: List of (int, int) corresponding to the index pairs
        (i_start, i_stop) of text to highlight in a UI display,
        which should be the location of the word token included
        in the query.
        The indexes should be used as text[i_start:i_strop]

# Other (better) work

Other free and open source search engines exists (ElasticSearch) and might be more efficient and much easier to implement. This project was meant as a completely internal in-house alternative to consider.

Also, embeddings used in this project are custom word2vec (trained on the corpus itself), and sentence2vec which is basically a bag-of-words approach of word2vec embeddings, with a subtraction of main component of the PCA of all corpus embeddings.
The custom training of the embeddings on the corpus allowed us to (approximately) create bilingual embeddings by training them on a bilingual corpus (French and English)

Much better word embeddings have been developed since this project, mainly fasttext multilingual embeddings, and transformer-based pretrained embeddings like BERT.

# To do

- Have returned objects of the *query_object()* function as attribute of the *query_object()* model itself, instead of a list of objects.
- Group all model files in a single *create_index()* or *query_object()* pickle file
- Better, clearer separation of usage of *widx* vs *widx_ord* - it's confusing in the code when this happens
- Some code is reused through query object (especially the query treatment)
- Some code is reused between *create_index()* and *query_object()*, and both classes should be merged
- In *create_index()*, There is a total of 7 passes over the corpus, each pass running *text_treatment.format_txt()* and *cleanpassage()* which takes a considerable amount of time. *text_treatment.format_txt()* could be included in the *pdf_treatment()* class instead, and *cleanpassage()* could be ran only once and then saved on drive for the other passes on the corpus (though unclean passages are needed for *query_object.print_closest_passages()*
- *query_object.print_closest_passages()* has never really been optimized at all in terms of speed and memory usage

# Dependancies

The pdf2text function depends on the [xpdf-tools](http://www.xpdfreader.com/download.html), which need to be downloaded and refered through the *exec_path* attribute of the *pdf_treatment()* class.

```
gensim = 3.8.3
googletrans = 2.4.0
nltk = 3.4.5
numpy = 1.18.5
scikit-learn = 0.23.1
unidecode = 1.1.1
```
