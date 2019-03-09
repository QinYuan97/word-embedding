# -*- coding: utf-8 -*-

#%%
import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf

from util import *
#%%
file = os.path.abspath('text8.zip')
words, total_words = read_data(file)
#%%
vocabulary_size = 10000

words_freq = build_words_freq(words, vocabulary_size)
#%%
import logging
import os
from gensim.models import word2vec
#%%
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='log.txt')

sentences = word2vec.Text8Corpus('text8')
#%%
pre_sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

size = 150
epochs = 10
model = word2vec.Word2Vec(pre_sentences, min_count=0, size=size)
#%%
model.build_vocab_from_freq(words_freq, update=True)
#%%
from gensim.models import KeyedVectors

pre_dims = 100
vectors = KeyedVectors.load(('vectors_%ddims_%depochs.kv' % (pre_dims, epochs)), mmap='r')
#%%
vectors.init_sims(replace = True)
#%%
for word in vectors.vocab:
    if word in model.wv.vocab:
        model.wv.vectors[model.wv.vocab[word].index][0:100] = vectors[word]
#%%
model.train(sentences, total_words=total_words, epochs=epochs)
#%%
model.wv.evaluate_word_analogies('questions-words.txt')
#%%
model.save('dimension_%dbatch.model' % (epochs))