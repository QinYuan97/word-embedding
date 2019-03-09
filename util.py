# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:16:25 2019

@author: qinyuan
"""

def test():
    print('hello')
#%% 
def read_data(filename):    
    '''读取训练集
    返回分割的words和总词数
    '''
    import zipfile
    import tensorflow as tf
    
    with zipfile.ZipFile(filename) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        total_words = len(words)
        print('Data size', total_words)
    return words, total_words
#%%
def build_words_freq(words, size):
    '''建立相应大小的词频字典
    '''
    import collections
    
    count = [['UNK', -1]]    
    count.extend(collections.Counter(words).most_common(size-1))
    dictionary = dict()    
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
#%%