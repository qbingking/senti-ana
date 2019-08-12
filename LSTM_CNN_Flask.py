from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
import os
import sys
import numpy as np
import pandas as pd
import re, os
import random
from flask import Flask, jsonify, render_template, request
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.models import Model
from keras import optimizers
from keras.layers import Input, LSTM, Bidirectional, Embedding, Dense, Convolution1D, Lambda, MaxPooling1D, Conv1D, Dropout, GlobalMaxPooling1D, Flatten
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from tqdm import tqdm
from underthesea import word_tokenize
from keras.regularizers import l2
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from keras.backend import clear_session

def main():
    current_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(current_dir, '..'))
    current_dir = current_dir if current_dir is not '' else '.'

    app = Flask(__name__)
    app.config.from_object(__name__)  # load config from this file , flaskr.py

    # Load default config and override config from an environment variable
    app.config.from_envvar('FLASKR_SETTINGS', silent=True)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    seed = 42
    np.random.seed(seed)

    filename='Models-sentiment/LSTMCNN-200-e10-smax.h5'
    epochs = 10
    word_embedding_dim = embed_dim = 200
    prew2v = "word_embedding/metmoisg-200.wv"

    datasetpath = './dataset/bigfile-train.xlsx'
    batch_size = 40
    max_fatures = 10000
    max_sequence_length = 50 # số từ tối đa / câu

    pre_trained_wv = True


    from keras import backend as K


    def clean_str(string):
        string = string.replace(':)', 'ngon')
        string = string.replace(':(', 'tệ')
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        # string = re.sub(r"\s{2,}", " ", string)
        cleanr = re.compile('<.*?>')
        string = re.sub(r'\d+', '', string)
        string = re.sub(cleanr, '', string)
        string = re.sub("'", '', string)
        string = re.sub(r'\W+', ' ', string)
        return string.strip().lower()

    def prepare_data(data):
        data = data[['text','sentiment']]
        data[data.index.notnull()]
        data['text'] = data['text'].apply(lambda x: str(x).lower())
        data['text'] = data['text'].apply(lambda x: clean_str(x))
        text = []
        for row in data['text'].values:
            word_list = word_tokenize(row,format="text").split(" ")
            stop_words = set(stopwords.words('vie123'))
            # print(stopwords)
            no_stop_words = [w for w in word_list if not w in stop_words]
            stop_words = set(stopwords.words('english'))
            no_stop_words = [w for w in word_list if not w in stop_words]
            # no_stop_words = [w for w in word_list if 1 == 1] # giữ stopword
            no_stop_words = " ".join(no_stop_words)
            word_tokenize(no_stop_words, format="text")
            text.append(no_stop_words) # ['one two three funny girl sale man r']
        tokenizer = Tokenizer(num_words=max_fatures, filters='!"#$%&*+,-./;<=>?@[\\]^`{|}~\t\n')
        tokenizer.fit_on_texts(text)
        X = tokenizer.texts_to_sequences(text)
        X = pad_sequences(X, maxlen= max_sequence_length)
        word_index = tokenizer.word_index
        Y = pd.get_dummies(data['sentiment']).values
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20)
        X_test, X_val, Y_test, Y_val = train_test_split(X_test,Y_test, test_size = 0.30)
        return X_train, X_test, X_val, Y_train, Y_test, Y_val, word_index, tokenizer

    def load_pre_trained_wv(word_index, num_words, word_embedding_dim):
        embeddings_index = {}
        prevec = KeyedVectors.load(prew2v, mmap='r')
        m1 = prevec.wv
        m3 = m1.vocab
        f_words = [i for i,j in m3.items()]
        print("<<< :::Load PreTrain Model W2V::: >>>")
        for aword in f_words:
            embeddings_index[aword] = np.asarray(m1.get_vector(aword), dtype='float32')
        print('%s word vectors.' % len(embeddings_index))
        embedding_matrix = np.zeros((num_words, word_embedding_dim))
        for word, i in word_index.items():
            if i >= max_fatures:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def model():
        if pre_trained_wv is True:
            print(":: USE PRETRAINED W2V ::")
            num_words = min(max_fatures, len(word_index) + 1)
            weights_embedding_matrix = load_pre_trained_wv(word_index, num_words, word_embedding_dim)
            print("num_words :: ",num_words)
            print("word_index + 1 :: ",len(word_index) + 1)
            input_shape = (max_sequence_length,) # 100
            model_input = Input(shape=input_shape, name="input", dtype='int32')
           
            model_lstm_cnn = Sequential()
            e = Embedding(num_words, embed_dim, weights=[weights_embedding_matrix], input_length=max_sequence_length, trainable=True)
            model_lstm_cnn.add(e)
            model_lstm_cnn.add(LSTM(embed_dim,
                                    dropout = 0.2,
                                    recurrent_dropout = 0.2,
                                    return_sequences=True))
            model_lstm_cnn.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
            model_lstm_cnn.add(GlobalMaxPooling1D())
            model_lstm_cnn.add(Dropout(0.3))
            model_lstm_cnn.add(Dense(100, activation='relu',
                                          kernel_regularizer= l2(0.005),
                                          activity_regularizer= l2(0.005)))
            model_lstm_cnn.add(Dense(3, activation='softmax'))
            return model_lstm_cnn

    def predict(new_text):
        Model = model()
        Model.compile(loss ='binary_crossentropy', optimizer='Adam', metrics = ['accuracy'])
        new_text = new_text.lower()
        new_text = clean_str(new_text)
        new_text = word_tokenize(new_text, format = "text")
        new_text = [new_text]
        Model.load_weights('./{}'.format(filename))
        history= Model.load_weights('./{}'.format(filename))
        sentiments = tokenizer.texts_to_sequences(new_text)
        sentiments = pad_sequences(sentiments, maxlen=max_sequence_length, dtype='int32', value=0)
        sentiments = Model.predict(sentiments,batch_size=1,verbose = 2)
        sentiments= sentiments[0]
        clear_session()
        return sentiments

    data = pd.read_excel(datasetpath)
    X_train, X_test, X_val, Y_train, Y_test, Y_val,  word_index, tokenizer = prepare_data(data)

    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)

    # @app.route('/')
    # def home():
    #     return render_template('home.html')

    @app.route('/about')
    def about():
        return 'About Us'

    @app.route('/', methods=['POST', 'GET'])
    def demo():
        if request.method == 'POST':
            if 'sentence' not in request.form:
                flash('No sentence post')
                redirect(request.url)
            elif request.form['sentence'] == '':
                flash('No sentence')
                redirect(request.url)
            else:
                sent = request.form['sentence']
                sentiments= predict(sent)
                if(np.argmax(sentiments) == 2):
                    sentiments="Tích cực "
                elif (np.argmax(sentiments) == 0):
                    sentiments="Tiêu Cực"
                elif (np.argmax(sentiments) == 1):
                    sentiments="Trung tính"
                return render_template('demo_result.html', sentence=sent, sentiments=sentiments)
        return render_template('demo.html')

    @app.errorhandler(404)
    def not_found(error):
        return make_response(jsonify({'error': 'Not found'}), 404)

    app.run(debug=False)


if __name__ == '__main__':
    main()
