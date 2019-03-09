# -*- coding: utf-8 -*-

#%%
import collections
import os
import zipfile
import numpy as np
import logging
import os
from gensim.models import word2vec

from util import *
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename='log.txt')
#%%
file = os.path.abspath('text8.zip')
words, total_words = read_data(file)
sentences = word2vec.Text8Corpus('text8')
#%%
words_freq_2000 = build_words_freq(words, 2000)
words_freq_10000 = build_words_freq(words, 10000)

pre_size = 100
epochs = 5
#%%
model = word2vec.Word2Vec.load('results/batch_%ddims_%depochs.model' % (pre_size, epochs))
#%%
#model.wv.most_similar(["girl"], topn=15)

#%%
# 将后 8000 词用2000词中与其最相近的替换
for word in words_freq_10000:
    if(word not in words_freq_2000 and word in model.wv.vocab):
        most_list = model.wv.most_similar([word], topn=15)
        for i in range(len(most_list)):
            simi_word = most_list[i][0]
            if(simi_word in words_freq_2000):
                #print(word + " " + simi_word)
                model.wv.vectors[model.wv.vocab[word].index] = model.wv[simi_word]
                break
#%%
pre_wv = model.wv
#%%
pre_sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]

new_size = 150
model = word2vec.Word2Vec(pre_sentences, min_count=0, size=new_size)

model.build_vocab_from_freq(words_freq_10000, update=True)
#%%
for word in model.wv.vocab:
    if word in pre_wv.vocab:
        model.wv.vectors[model.wv.vocab[word].index][0:pre_size] = pre_wv[word]
#%% 训练，前100维不变
epochs = 10
for i in range(epochs):
    model.train(sentences, total_words=total_words, epochs=1)
    for word in model.wv.vocab:
        if word in pre_wv.vocab:
            model.wv.vectors[model.wv.vocab[word].index][0:pre_size] = pre_wv[word]
#%%
model.wv.evaluate_word_analogies('questions-words.txt')