#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import math
import pandas as pd
import numpy as np
from collections import Counter


# In[2]:


# punctuation list setup
import string
punct_list = []
for punct in string.punctuation:
    punct_list.append(punct)


# ### Corpus download

# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/bshmueli/108-nlp/master/reuters.csv")


# ### Stopwords 

# In[4]:


stopwords = pd.read_csv("https://raw.githubusercontent.com/bshmueli/108-nlp/master/stopwords.txt")
stopwords = [row for row in stopwords['i']]


# In[5]:


def get_corpus(df):
#     print("Dataset size", len(df))
#     print("Dataset columns", df.columns)
    corpus = df.content.to_list()
    return corpus


# In[6]:


def tokenize(document):
    words = document.split(' ')
    # Convert all tokens to lowercase
    words = [word.lower() for word in words]
    # Remove stopwords
    words = [w for w in words if not w in stopwords] 
    return words


# In[7]:


def rm_punctuation(token):
    clean = []
    for word in token:
        word = (re.split('\W+', word))
        x = list(filter(None,word))
        clean.append(x)
    w = []
    for item in clean:
        for x in item:
            w.append(x)
    return w


# In[8]:


def get_vocab(corpus):
    vocabulary = Counter()
    for document in corpus:
        tokens = tokenize(document)
        vocabulary.update(tokens)
    return vocabulary


# In[9]:


def get_cleaned_corpus():
    corpus = []
    for i in range(len(df)):
        corpus.append(rm_punctuation(tokenize(get_corpus(df)[i])))
    return corpus


# In[10]:


all_corpus = get_cleaned_corpus()


# In[11]:


full_corpus = [corpus[0] for corpus in all_corpus]
vocab = get_vocab(full_corpus).most_common(1000)


# In[12]:


words_dict = {}
for i in range(len(vocab)):
    words_dict[vocab[i][0]] = vocab[i][1]


# In[14]:


# tf
tf = []
for token, freq in vocab:
    vb = []
    for i in range(len(all_corpus)):
        if token in all_corpus[i]:
            vb.append(1)
        else:
            vb.append(0)
    tf.append(vb)
    
# df_x
df_x = []
for x in range(len(vocab)):
    count = 0
    for y in range(len(all_corpus)):
        if tf[x][y] != 0:
            count += 1
    df_x.append(count)
    
# w
w = np.zeros((len(vocab), len(all_corpus)))
for x in range(len(vocab)):
    for y in range(len(all_corpus)):
        w[x, y] = tf[x][y] * math.log(len(all_corpus) / df_x[x])


# In[15]:


s = {}
count=0
for i in range(len(vocab)):
    s[vocab[i][0]] = count
    count += 1


# In[16]:


doc_vec = []
for i in range(len(get_corpus(df))): 
    g = np.zeros(len(get_corpus(df)))
    count = 0
    for item in rm_punctuation(tokenize(get_corpus(df)[i])): # sum the vector of the words
        if item in s:
            g += w[s[item]]
            count += 1
    doc_vec.append(g/count)


# In[18]:


def doc2vec(doc):
    words = tokenize(doc)
    return [1 if token in words else 0 for token, freq in vocab]


# In[19]:


def cosine_similarity(vec_a, vec_b):
    assert len(vec_a) == len(vec_b)
    if sum(vec_a) == 0 or sum(vec_b) == 0:
        return 0
    a_b = sum(i[0] * i[1] for i in zip(vec_a, vec_b))
    a_2 = sum([i*i for i in vec_a])
    b_2 = sum([i*i for i in vec_b])
    return a_b/(math.sqrt(a_2) * math.sqrt(b_2))


# In[20]:


def doc_similarity(doc_a, doc_b):
    return cosine_similarity(doc2vec(doc_a), doc2vec(doc_b))


# In[21]:


def k_similar(seed_id, k=5):
    seed_doc = corpus[seed_id]
    print(' > "{}"'.format(seed_doc))
    
    similarities = [cosine_similarity(doc_vec[seed_id], doc) for id, doc in enumerate(doc_vec)]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i])[-k:]
    nerest = [[corpus[id], similarities[id]] for id in top_indices]
    print()
    for story in reversed(nerest):
        print('* "{}" ({})'.format(story[0], story[1]))


# In[22]:


corpus = get_corpus(df)
full_corpus = [corpus[0] for corpus in all_corpus]
vocab = get_vocab(full_corpus).most_common(1000)
k_similar(10, 5)

