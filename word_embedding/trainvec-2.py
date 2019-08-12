import numpy as np
import pandas as pd
from underthesea import word_tokenize
import re, os
from nltk.corpus import stopwords
import fasttext
from gensim.models import Word2Vec
import multiprocessing
import gensim.downloader as api

def clean_str(string):
    string = string.replace(':)', 'ngon')
    string = string.replace(':(', 'tệ')
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    # string = re.sub(r"\s{2,}", " ", string)
    cleanr = re.compile('<.*?>')
    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    return string.strip().lower()


if __name__ == '__main__':
    data =  pd.read_excel('../dataset/bigfile-train6389.xlsx')
    data = data[['text','sentiment']]
    data[data.index.notnull()]
    data['text'] = data['text'].apply(lambda x: str(x).lower())
    data['text'] = data['text'].apply(lambda x: clean_str(x))
    
    text = []
    no_stop_words=[]
    for row in data['text'].values:
        word_list = word_tokenize(row,format="text").split(" ")
        stop_words = set(stopwords.words('vie123'))
        no_stop_words = [w for w in word_list if not w in stop_words]
        stop_words = set(stopwords.words('english'))
        no_stop_words = [w for w in word_list if not w in stop_words]
        text.append(no_stop_words)
    train_data = text
    # print(train_data)
    print(len(train_data))
    # print(train_data[:5])
    DIM = 150
    model = Word2Vec(
            train_data,
            size = DIM,
            window=2,
            min_count=2,
            negative=5,
            sg = 0,
            iter=10, workers=multiprocessing.cpu_count(), alpha=0.065, min_alpha=0.065)
    model.save("metmoicb-150.wv")
    # word_vecs = model.wv
    # print(dir(word_vecs))
    # print(word_vecs.get_vector())
    # result = word_vecs.similar_by_word("ngon")
    # print('same Ngon:', result[:10])
    # result = word_vecs.similar_by_word("tệ")
    # print('same Tệ:', result[:10])
    # result = word_vecs.similar_by_word("dịch_vụ")
    # print('same Dịch vụ:', result[:10])