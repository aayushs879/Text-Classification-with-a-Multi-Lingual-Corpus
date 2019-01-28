#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
encor = np.load('encor.npy').item()
escor = np.load('escor.npy').item()
frcor = np.load('frcor.npy').item()
esidf = np.load('esidf.npy').item()
fridf = np.load('fridf.npy').item()
enidf = np.load('enidf.npy').item()
idx = list(np.loadtxt('idx.txt'))
length = len(np.loadtxt('word_features.txt'))
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'word-vecs/'))


# In[3]:


from gensim.models import KeyedVectors
esmodel = KeyedVectors.load_word2vec_format('word-vecs/wiki.es/wiki.es.vec', binary = False) #fastText spanish model
frmodel = KeyedVectors.load_word2vec_format('word-vecs/wiki.fr/wiki.fr.vec', binary = False) #fasttext french model
enmodel = KeyedVectors.load_word2vec_format('word-vecs/glove.6B/glove.6B.300d.w2vformat.txt', binary = False) #Glove model for global vector representation in english by stanford


# In[ ]:
# i used tfidf weighted average to obtain the final vector of a sentence

X_es = np.zeros((len(escor.keys()), 300))
from textblob import TextBlob
i = 0
for key in escor:
    words = escor[key].split(' ')
    ngrams = TextBlob(escor[key]).ngrams(3) + TextBlob(escor[key]).ngrams(2)
    for wordlist in ngrams:
        words.append(' '.join([wordlist[i] for i in range(len(wordlist))]))
    count = 0
    vec = np.zeros(300)
    temp = 0
    for word in words:
        try:
            vec += esidf[word] * esmodel[word]
            count += 1
        except :
            temp +=1
    #print('{p}% of words empty in {key}'.format(p = temp*100/len(words), key = key))
    if count != 0:
        vec = vec/count
    X_es[i] = vec.reshape((1, 300))
    i +=1
    if i%200 ==0:
        print('Completed upto index', i)
    


# In[ ]:


np.savetxt('X_es.txt', X_es)


# In[4]:


X_fr = np.zeros((len(frcor.keys()), 300))
from textblob import TextBlob
i = 0
for key in frcor:
    words = frcor[key].split(' ')
    ngrams = TextBlob(frcor[key]).ngrams(3) + TextBlob(frcor[key]).ngrams(2)
    for wordlist in ngrams:
        words.append(' '.join([wordlist[i] for i in range(len(wordlist))]))
    count = 0
    vec = np.zeros(300)
    temp = 0
    for word in words:
        try:
            vec += fridf[word] * frmodel[word]
            count += 1
        except :
            temp +=1
    #print('{p}% of words empty in {key}'.format(p = temp*100/len(words), key = key))
    if count != 0:
        vec = vec/count
    X_fr[i] = vec.reshape((1, 300))
    i +=1
    if i%200 ==0:
        print('Completed upto index', i)


# In[5]:


np.savetxt('X_fr.txt', X_fr)


# In[ ]:


X_en = np.zeros((len(encor.keys()), 300))
from textblob import TextBlob
i = 0
for key in encor:
    words = encor[key].split(' ')
    ngrams = TextBlob(encor[key]).ngrams(3) + TextBlob(encor[key]).ngrams(2)
    for wordlist in ngrams:
        words.append(' '.join([wordlist[i] for i in range(len(wordlist))]))
    count = 0
    vec = np.zeros(300)
    temp = 0
    for word in words:
        try:
            vec += enidf[word] * enmodel[word]
            count += 1
        except :
            temp +=1
    print('{p}% of words empty in {key}'.format(p = round(temp*100/len(words), 3), key = key))
    if count != 0:
        vec = vec/count
    X_en[i] = vec.reshape((1, 300))
    i +=1
    if i%200 ==0:
        print('Completed upto index', i)


# In[ ]:


np.savetxt('X_en.txt', X_en)






