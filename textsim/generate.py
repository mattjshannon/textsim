#!/usr/bin/env python3
"""

generate.py

Comparing three of my scientific publications to
Frankenstein and Pride and Prejudice with gensim.

NOTE:

Frankenstein and Pride and Prejudice texts are care of:
Project Gutenberg. (n.d.). Retrieved September 4, 2019, from www.gutenberg.org.
"""

import re
from collections import defaultdict
from pkg_resources import resource_filename

import gensim
from ipdb import set_trace as st
from gensim import corpora
from gensim.utils import simple_preprocess


def convert_txt_to_dict(txt_path, save_path=None):
    '''Convert a txt file to a gensim Dictionary.'''
    corpus_dict = corpora.Dictionary(
        simple_preprocess(line, deacc=True) for line in open(txt_path)
    )

    if save_path is None:
        save_path = txt_path.replace('.txt', '.dict')

    corpus_dict.save(save_path)
    print('Saved: {}'.format(save_path))

    return

def generate_preprocessed_dict(txt_path, dict_path):
    '''Return a saved gensim Dictionary from disk; create if doesn't exist.'''
    try:
        dict0 = corpora.Dictionary.load(dict_path)
    except FileNotFoundError as e:
        print('File not found, thus creating...')
        convert_txt_to_dict(txt_path, save_path=dict_path)
    else:
        print('Saved: {}'.format(dict_path))

    return dict0


def generate_processed_corpus(path):
    # Form corpus.
    try:
        with open(path, 'r') as file:
            document = file.read()
    except Exception as e:
        raise(e)

    # Create a set of frequent words / from tutorial
    stoplist = set('for a of the and to in'.split(' '))

    # Lowercase each document, split it by white space and filter out stopwords
    texts = [word for word in document.lower().split() if word not in stoplist]

    # Count word frequencies
    frequency = defaultdict(int)
    for token in texts:
        frequency[token] += 1

    # Only keep words that appear more than once
    processed_corpus = [[token for token in texts if frequency[token] > 1]]

    return processed_corpus


books = ['frankenstein.txt', 'pride_and_prejudice.txt']
papers = ['paper_2015.txt', 'paper_2016.txt', 'paper_2018.txt']

all_books = []
all_papers = []

# Iterate over all documents and create dictionaries.
for document in (*books, *papers):
    file_path = resource_filename('textsim', 'resources/' + document)
    save_path = file_path.replace('.txt', '.dict')
    regex = re.compile('[\W_]+', re.UNICODE)

    try:
        with open(file_path, 'r') as file:
            data = file.read()
            cleaned_data = simple_preprocess(regex.sub(' ', data))

            if document in books:
                all_books.append(cleaned_data)
            elif document in papers:
                all_papers.append(cleaned_data)

            corpus_dict = corpora.Dictionary([cleaned_data])
            corpus_dict.save(save_path)
            print('Saved: {}'.format(save_path))

    except Exception as e:
        raise(e)



model1 = gensim.models.Word2Vec(
    all_books, size=150, window=10, min_count=2, workers=10, iter=10
)

model2 = gensim.models.Word2Vec(
    all_papers, size=150, window=10, min_count=2, workers=10, iter=10
)

# print(model.wv.most_similar(positive='molecules', topn=20))

# model_path = resource_filename('textsim', 'resources/' + 'test_model.bin')
# model2.save(model_path)

# model2.wv.save_word2vec_format(model_path, binary=True)



model1.wv.save_word2vec_format('test1.txt')
model2.wv.save_word2vec_format('test2.txt')

# # Generate and save corpora Dictionaries.
# fdict = generate_preprocessed_dict(txt_path=frank, dict_path=frank.replace('.txt', '.dict'))
# pdict = generate_preprocessed_dict(txt_path=pride, dict_path=pride.replace('.txt', '.dict'))

# # Generate and save processed paper corpuses.
# # for path in (p1, p2, p3):
# for path in (frank, pride):
#     corp = generate_processed_corpus(path=path)

#     fcorpus = [fdict.doc2bow(text) for text in corp]
#     pcorpus = [pdict.doc2bow(text) for text in corp]

#     fpath = path.replace('.txt', '.frank_mm')
#     ppath = path.replace('.txt', '.pride_mm')

#     corpora.MmCorpus.serialize(fpath, fcorpus)
#     corpora.MmCorpus.serialize(ppath, pcorpus)

#     print('Saved: {}'.format(fpath))
#     print('Saved: {}'.format(ppath))
