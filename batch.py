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
vocabulary_size = 2000

words_freq_2000 = build_words_freq(words, vocabulary_size)
#%%
import logging
import os
from gensim.models import word2vec
#%%
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.Text8Corpus('text8')
#%%
pre_sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

size = 100
epochs = 10
model = word2vec.Word2Vec(pre_sentences, min_count=0, size=size)
#%%
model.build_vocab_from_freq(words_freq_2000, update=True)
#%%
model.train(sentences, total_words=total_words, epochs=epochs)
#%%
vocabulary_size = 10000
words_freq_10000 = build_words_freq(words, vocabulary_size)
model.build_vocab_from_freq(words_freq_10000, update=True)
#%%
model.train(sentences, total_words=total_words, epochs=epochs)
#%%
#model = word2vec.Word2Vec.load('results/batch_%depochs.model' % (epochs))

#%%
model.wv.evaluate_word_analogies('questions-words.txt')
#%%
model.save('results/batch_%ddims_%depochs.model' % (size, epochs))
#%%
print('First 2000 words count: ' , (total_words - words_freq_2000['UNK'] ))
print('First 10000 words count: ' , (total_words - words_freq_10000['UNK']))
