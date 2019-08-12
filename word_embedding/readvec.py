import numpy as np
import pandas as pd
from underthesea import word_tokenize
import re, os
from nltk.corpus import stopwords
import fasttext
from gensim.models import Word2Vec
import multiprocessing
import gensim.downloader as api

# model = Word2Vec.load('metmoi3.wv',mmap='r')
from gensim.models import KeyedVectors
mo = KeyedVectors.load("metmoi3.wv", mmap='r')
m1 = mo.wv
m2 = m1.get_vector('ngon')
m3 = m1.vocab
print(dir(m1))
print([i for i,j in m3.items()])
# print([j for i,j in m3.items()])
