# -*- coding: utf-8 -*-
#%%
import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf

#%%
def read_data(filename):    
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()    
    return data

file = os.path.abspath('text8.zip')
words = read_data(file)
print('Data size', len(words))
total_words = len(words)

#%%
vocabulary_size = 10000

def build_words_freq(words, size):
    count = [['UNK', -1]]    
    # 词汇频数统计
    count.extend(collections.Counter(words).most_common(size-1))
    dictionary = dict()    
    # 存入字典
    for word, _ in count:
        dictionary[word] = len(dictionary)

    unk_count = 0 
    for word in words:        
        if word not in dictionary:
            unk_count += 1
      
    count[0][1] = unk_count    
    words_freq = dict()
    for i in range(len(count)):
        words_freq[count[i][0]] = count[i][1]
    return words_freq

words_freq = build_words_freq(words, vocabulary_size)
#%%
import logging
import os
from gensim.models import word2vec
from gensim.models import KeyedVectors
#%%
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='log.txt')
sentences = word2vec.Text8Corpus('text8') 

pre_sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
model = word2vec.Word2Vec(pre_sentences, min_count=0, size=100)

#%%
model.build_vocab_from_freq(words_freq, update=True)

#%%
epochs = 30
model.train(sentences, total_words=total_words, epochs=epochs)
#%%
model = word2vec.Word2Vec.load('results/origin_%depochs.model' % (epochs))
#%%
model.wv.most_similar(positive=["woman","king"],negative=["man"],topn=1)
#%%
model.wv.most_similar(['girl','father'],['boy'],topn=3)
#%%
model.wv.most_similar(["boy"], topn=3)
#%%
model.wv.similarity("boy", "girl")
#%%
model.wv.doesnt_match("blue god red black white".split())
#%%
model.wv.evaluate_word_analogies('questions-words.txt')

#%%
#vectors = KeyedVectors.load('vectors_100dims_5epochs.kv', mmap='r')

#model.save('results/origin_%depochs.model' % (epochs))
