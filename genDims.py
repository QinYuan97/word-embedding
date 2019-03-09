# -*- coding: utf-8 -*-


import os
import tensorflow as tf
import numpy as np

from util import *
#%%
file = os.path.abspath('text8.zip')
words, total_words = read_data(file)
#%%
vocabulary_size = 2000

words_freq = build_words_freq(words, vocabulary_size)
#%%
import logging
import os
from gensim.models import word2vec
from gensim.models import KeyedVectors

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='log.txt')

sentences = word2vec.Text8Corpus('text8')

pre_sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
size = 100
model = word2vec.Word2Vec(pre_sentences, min_count=0, size=size)
#%%
epochs = 10
model.build_vocab_from_freq(words_freq, update=True)

model.train(sentences, total_words=total_words, epochs=epochs)
#%%
model.wv.evaluate_word_analogies('questions-words.txt')
#%%
model.wv.save('vectors_%ddims_%depochs.kv' % (size, epochs))