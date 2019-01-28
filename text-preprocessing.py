#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from textblob import TextBlob
import re
import unidecode
from contractions import expandContractions


# In[2]:


text = pd.read_csv('train.csv').drop(['Complaint-Status'], axis = 1)
text = text.append(pd.read_csv('test.csv'), ignore_index = True)
'''text['word count'] = text['Consumer-complaint-summary'].apply(lambda x : len(str(x).split(' ')))
text['char_len'] = text['Consumer-complaint-summary'].apply(lambda x : len(str(x)))

def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

text['avg_word'] = text['Consumer-complaint-summary'].apply(lambda x: avg_word(str(x)))
word_features = text.iloc[:, -3:].values
np.savetxt('word_features.txt', word_features)'''
text['Consumer-Complaint-summary'] = text['Consumer-complaint-summary'].apply(lambda x: expandContractions(x))
text['Consumer-complaint-summary'] = text['Consumer-complaint-summary'].apply(lambda x: re.sub('[~`!@#$%^&*():;"{}_/?><\|.,`0-9]', '', x.replace('-', ' ')))
#text['Consumer-complaint-summary'] = text['Consumer-complaint-summary'].apply(lambda x: unidecode.unidecode(x))
text = text['Consumer-complaint-summary'].iloc[:].values


# In[3]:


# detecting the corresponding languages of summary



"""!pip install langdetect
from langdetect import detect
languages = []
for i in range(len(text)):
  languages.append(detect(text[i].lower()))
idx = []
for i in range(len(l)):
    if (l[i] != 'en') and (l[i] != 'fr') and (l[i] != 'es'):
        idx.append(i)
        
for i in idx:
    print(i, text[i])
  
pd.DataFrame(languages, index = None).to_csv(os.path.join(path, 'languages.csv'), header = None, index = None)
"""
languages = pd.read_csv('languages.csv')
idx = np.loadtxt('idx.txt')
fr = {}
en = {}
es = {}
for i in range(len(languages.one)):
    if i not in idx:
        if languages.one[i] == 'fr':
            fr.update({i:text[i]})
        elif languages.one[i] == 'es':
            es.update({i:text[i]})
        else:
            en.update({i:text[i]})


# In[ ]:





# In[4]:


# langdetect detected more than three languages but we will consider these as main as their counts were substantially large, rest we shall append to english language
# now we make a dictionary of the statements corresponding to a particular language whilst preserving their indices

senti = np.zeros((len(text), 1))
from textblob import Word
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
encor = {}
frcor = {}
escor = {}

fr_stem = SnowballStemmer('french')
es_stem = SnowballStemmer('spanish')
en_stem = SnowballStemmer('english')

swfr = set(stopwords.words('french'))
swen = set(stopwords.words('english'))
swes = set(stopwords.words('spanish'))


#making three different corpora for languages
for key in fr:
    words = nltk.word_tokenize(fr[key].lower())
    words = filter(lambda t: not t.startswith('xx'), words)
    words = filter(lambda t: not t.startswith('yy'), words)
    words = [Word(word).lemmatize() for word in words if not word in swfr]
    sentence = ' '.join(words)
    senti[int(key)] = TextBlob(sentence).sentiment[0]
    frcor.update({key:unidecode.unidecode(sentence)})


for key in en:
    words = nltk.word_tokenize(en[key].lower())
    words = filter(lambda t: not t.startswith('xx'), words)
    words = filter(lambda t: not t.startswith('yy'), words)
    words = [Word(word).lemmatize() for word in words if not word in swen]
    sentence = ' '.join(words)
    senti[int(key)] = TextBlob(sentence).sentiment[0]
    encor.update({key:unidecode.unidecode(sentence)})


for key in es:
    words = nltk.word_tokenize(es[key].lower())
    words = filter(lambda t: not t.startswith('xx'), words)
    words = filter(lambda t: not t.startswith('yy'), words)
    words = [Word(word).lemmatize() for word in words if not word in swes]
    sentence = ' '.join(words)
    senti[int(key)] = TextBlob(sentence).sentiment[0]
    escor.update({key:unidecode.unidecode(sentence)})


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
entv = TfidfVectorizer(ngram_range = (1, 3), max_features = 3000)
estv = TfidfVectorizer(ngram_range = (1, 3), max_features = 1000)
frtv = TfidfVectorizer(ngram_range = (1, 3), max_features = 1000)
entv.fit_transform(encor.values())
estv.fit_transform(escor.values())
frtv.fit_transform(frcor.values())


# In[6]:


esidf = dict(zip(estv.get_feature_names(), estv.idf_))
enidf = dict(zip(entv.get_feature_names(), entv.idf_))
fridf = dict(zip(frtv.get_feature_names(), frtv.idf_))


# In[7]:


np.save('encor.npy', encor)
np.save('frcor.npy', frcor)
np.save('escor.npy', escor)
np.save('esidf.npy', esidf)
np.save('fridf.npy', fridf)
np.save('enidf.npy', enidf)









