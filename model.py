# Import custom helper libraries
import os, sys, re, csv, codecs
import pickle

# Maths modules
import numpy as np
import pandas as pd
from numpy import exp
from numpy.core.fromnumeric import repeat, shape  # noqa: F401,W0611
from scipy.stats import f_oneway

# Viz modules
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
%matplotlib inline

# Render for export
import plotly.io as pio
pio.renderers.default = "notebook"
# import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)  
import plotly.figure_factory as ff

#Sklearn modules
from sklearn import metrics
from sklearn.metrics import (ConfusionMatrixDisplay,PrecisionRecallDisplay,RocCurveDisplay,)
from sklearn.metrics import (confusion_matrix, roc_auc_score, average_precision_score, classification_report)
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)
from sklearn.base import ClassifierMixin, is_classifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# System modules
import random
import contractions
import re
import time
from collections import Counter
from collections import defaultdict
from unidecode import unidecode
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union
import gc
from random import shuffle
import itertools

# ML modules
from tqdm import tqdm
tqdm.pandas()

# NLTK modules
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Keras modules
import keras
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, BatchNormalization, TimeDistributed, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D,Bidirectional
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.initializers import Constant
# from keras.layers import (LSTM, Embedding, BatchNormalization, Dense, TimeDistributed, Dropout, Bidirectional, Flatten, GlobalMaxPool1D)
# from keras.optimizers import Adam

# Tensoflow modules
from tensorflow.keras.callbacks import EarlyStopping

# Gensim
import gensim.models.keyedvectors as word2vec

# Load data from CSV
df = pd.read_csv(r"C:\\Users\\ezequ\\proyectos\\openclassrooms\\Projet_7\\data\\raw\\sentiment140_16000_tweets.csv",
                 names=["target", "text"], encoding='latin-1')

# Drop useless raw
df = df.iloc[1: , :]

#TEXT PREPROCESSING
def text_cleaning(text, ponct, only_letters, numbers):
    text = text.lower()
    text = unidecode(text)
    ponctuation = "[^!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]"
    number = "[^0-9]"
    letters = "[^a-zA-Z ]"
    if ponct == 1:
        text = re.sub(ponctuation, '', text)
    if only_letters == 1:
        text = re.sub(letters, '', text)
    if numbers == 1:
        text = re.sub(number, '', text)
    return text

# Let's put the text in lower case.
df["new_text"] = df["text"].str.lower()

# Let's remove the punctuation.
df['new_text'] = df.progress_apply(lambda x: text_cleaning(x['text'], 0, 1, 0),axis=1)

# We can separate the text into word lists => each word unit is a tokens
df['words'] = df.progress_apply(lambda x: word_tokenize(x['new_text']),axis=1)

# Let's count the number of words per comment
df['nb_words'] = df.progress_apply(lambda x: len(x['words']),axis=1)

nltk.download('stopwords')
sw_nltk = stopwords.words('english')
keep_words = []
new_sw_nltk = [word for word in sw_nltk if word not in keep_words]
new_sw_nltk.extend(['th','pm', 's', 'er', 'paris', 'rst', 'st', 'am', 'us'])
pat = r'\b(?:{})\b'.format('|'.join(new_sw_nltk))
cleaning = df['new_text'].str.replace(pat, '')
df['new_words'] = cleaning.progress_apply(lambda x: nltk.word_tokenize(x))
df['new_text'] = cleaning

# The process of classifying words into their parts of speech and labeling 
# them accordingly is known as part-of-speech tagging, POS-tagging, or simply tagging. 

def word_pos_tagger(list_words):
    pos_tagged_text = nltk.pos_tag(list_words)
    return pos_tagged_text

all_reviews = df["new_text"].str.cat(sep=' ')
description_words = word_pos_tagger(nltk.word_tokenize(all_reviews))
list_keep = []
list_excl = ['IN', 'DT', 'CD', 'CC', 'RP', 'WDT', 'EX', 'MD', 'NNP', 'WDT', 'UH', 'WRB', 
'WP', 'WP$', 'PDT', 'PRP$', 'EX', 'POS', 'SYM', 'TO', 'NNPS']
for word, tag in description_words:
    if tag not in list_excl:
        list_keep.append(tag)
        
df["text_tokens_pos_tagged"] =  df["new_text"].progress_apply(lambda x: nltk.word_tokenize(x))
df["text_tokens_pos_tagged"] =  df["text_tokens_pos_tagged"].progress_apply(lambda x: nltk.pos_tag(x))

list_nouns = ["NN", "NNS"]
df["words_subjects"] =  df["text_tokens_pos_tagged"].progress_apply(lambda x: [y for y, tag in x if tag in list_nouns])

# The join() method takes all items in an iterable and joins them into one string.
df["words_subjects"] =  df["words_subjects"].progress_apply(lambda x: " ".join(x))

def stemming_text(word):
    stemmer = SnowballStemmer(language='english')
    return stemmer.stem(word)

df["words_subjects_st"] = df["words_subjects"].progress_apply(lambda x: stemming_text(x))

#label enconder
le = LabelEncoder()
le.fit(df['target'])

# VECTORIZATION
df['target_encoded'] = le.transform(df['target'])

list_classes = ["target_encoded"]
y = df[list_classes].values
embed_size=0

list_sentences_train = df["words_subjects_st"].values
list_sentences_test = df["words_subjects_st"].values
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list_sentences_train)
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(list_sentences_train)
word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index)  + 1

def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(list_sentences_train, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))
padded_sentences = pad_sequences(embed(list_sentences_train), length_long_sentence, padding='post')

maxlen = 300
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

def loadEmbeddingMatrix(typeToLoad):
    """
    Args:
        typeToLoad: word_embedding type
    Returns:
        Embedding_matrix.
    """
    if(typeToLoad=="word2vec"):
        model = word2vec.KeyedVectors.load_word2vec_format('C:\Program Files (x86)\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin',binary=True,limit=100000)
        embed_size = 300
        
        embeddings_index = dict()
        for word in model.key_to_index:
            embeddings_index[word] = model.word_vec(word)
        print('Loaded %s word vectors.' % len(embeddings_index))

        gc.collect()
        all_embs = np.hstack(list(embeddings_index.values()))
        emb_mean,emb_std = all_embs.mean(), all_embs.std()

        nb_words = len(tokenizer.word_index)
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        gc.collect()

        EMBEDDING_DIM = 300
        embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
        embeddedCount = 0
        for word, i in word_index.items():
            i-=1
            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
            embedding_vector = embeddings_index.get(word)
            #and store inside the embedding matrix that we will train later on.
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')

        del(embeddings_index)
        gc.collect()
        #finally, return the embedding matrix
        return embedding_matrix
    
embedding_matrix_word2vec = loadEmbeddingMatrix('word2vec')

maxlen=300
inp = Input(shape=(maxlen, ))

EMBEDDING_DIM = 300

x_word2vec = Embedding(vocab_size,
          EMBEDDING_DIM,
          weights=[embedding_matrix_word2vec],
          input_length=length_long_sentence,
          trainable=False)(inp)

# Split data into train and test sets
# set aside 20% of train and test data for evaluation

X_train, X_test, y_train, y_test = train_test_split(X_t, y,
    test_size=0.2, shuffle = True, random_state = 42)

# Use the same function above for the validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
    test_size=0.1, random_state= 42) # 0.25 x 0.8 = 0.2

# Model
def lstm_1():
    model = Sequential()
    
    model.add(Embedding(
        input_dim=embedding_matrix_word2vec.shape[0], 
        output_dim=embedding_matrix_word2vec.shape[1], 
        weights = [embedding_matrix_word2vec], 
        input_length=maxlen
    ))
    
    model.add(Bidirectional(LSTM(
        length_long_sentence, 
        return_sequences = True, 
        recurrent_dropout=0.2
    )))
    
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = lstm_1()

#fit the model with train set
batch_size = 32
epochs = 10
hist = model.fit(X_train,y_train, batch_size=batch_size, epochs=epochs,  validation_data= [X_val, y_val], callbacks=[EarlyStopping(monitor="val_loss", patience=3),], workers=4,use_multiprocessing=True,)

# Save the model to disk
pickle.dump(model, open('model.pkl', 'wb'))

# # Loading the model to compare results
# model = pickle.load(open('model.pkl', 'rb'))
# print(model.predict([[2, 9, 6]]))
