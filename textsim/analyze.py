#!/usr/bin/env python3
"""

analyze.py

Comparing three of my scientific publications to
Frankenstein and Pride and Prejudice with gensim.

NOTE:

Frankenstein and Pride and Prejudice texts are care of:
Project Gutenberg. (n.d.). Retrieved September 4, 2019, from www.gutenberg.org.
"""

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from collections import defaultdict
from pkg_resources import resource_filename

import gensim
from ipdb import set_trace as st
from gensim import corpora, models, similarities
from gensim.utils import simple_preprocess


# Basic paths needed.
frank = resource_filename('textsim', 'resources/frankenstein.dict')
pride = resource_filename('textsim', 'resources/pride_and_prejudice.dict')
p1f = resource_filename('textsim', 'resources/paper_2015.frank_mm')
# p2 = resource_filename('textsim', 'resources/paper_2016.txt')
# p3 = resource_filename('textsim', 'resources/paper_2018.txt')
test = resource_filename('textsim', 'resources/pride_and_prejudice.frank_mm')

dictionary = corpora.Dictionary.load(frank)
corpus = corpora.MmCorpus(test)

lsi = models.LsiModel(corpus, id2word=dictionary)#, num_topics=2)



doc = "visited"
vec_bow = dictionary.doc2bow(doc.lower().split())
vec_lsi = lsi[vec_bow] # convert the query to LSI space
print(vec_lsi)


