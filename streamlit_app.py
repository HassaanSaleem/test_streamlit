import streamlit as st

#!/usr/bin/env python
# coding: utf-8

# In[1]:

pip install sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import pandas as pd
import numpy as np
import nltk
import re

import gensim
from gensim.models import Word2Vec
import gensim.downloader as api

import operator
import pickle
import joblib


# In[2]:


def remove_special_characters(text):

    text = re.sub(r"(?<!\d)[.,;:'!](?!\d)", "", text, 0)
    
    return text.replace(r"[^A-Za-z0-9]","");

def get_wordnet_pos(word, tag):

    if tag.startswith('J'):
        return word, wordnet.ADJ
    elif tag.startswith('V'):
        return word, wordnet.VERB
    elif tag.startswith('N'):
        return word, wordnet.NOUN
    elif tag.startswith('R'):
        return word, wordnet.ADV
    else:
        return word, wordnet.VERB

def lemmatize_text(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    text = nltk.pos_tag(nltk.word_tokenize(text))
    text = [get_wordnet_pos(word[0], word[1]) for word in text]
    
    return [lemmatizer.lemmatize(w, v) for w, v in text]

def preprocessing(text):
    text = text.lower()
    text = remove_special_characters(text)

    StopWords = stopwords.words("english")
    
    tokenize_text = ' '.join([word for word in nltk.word_tokenize(text) if word not in StopWords])

    lemm_text = lemmatize_text(tokenize_text)
    
    return ' '.join([word for word in lemm_text])


# In[3]:


data = pd.read_csv('news_headlines.csv')


# ### TfidfVectorizer

# In[4]:


vectorizer = pickle.load(open('tfidf.pickle', 'rb'))


# In[5]:


X = vectorizer.fit_transform(data.headlines)


# In[6]:


y = vectorizer.transform(["raza rabbani"])


# In[7]:


cosine_similarities = cosine_similarity(X,y)
cosine_similarities = {x: cosine_similarities[x].item() for x in range(len(cosine_similarities))}
cosine_similarities = sorted(cosine_similarities.items(), key=operator.itemgetter(1), reverse=True)


# In[8]:


res = []
for ind, value in cosine_similarities:
    if value > 0:
        res.append(data.headlines[ind])


# In[9]:


st.text(res[0])



# ## Word2Vec

# In[46]:


tk_data = p_data.apply(nltk.word_tokenize)


# In[47]:


# Create CBOW model
model1 = gensim.models.Word2Vec(tk_data, min_count = 1,
                              vector_size = 300, window = 5)


# In[48]:


ind = 0
word2vec_x = []
for news in p_data:
    tokenized_news = nltk.word_tokenize(news)
    vector = []
    for words in tokenized_news:
        try:
            vector.append(model1.wv.get_vector(words))
        except:
            pass
    word2vec_x.append(np.add.reduce(vector).reshape(1, -1))


# In[49]:


y = 'raza rabbani'

tokenized_y = nltk.word_tokenize(y)
vector = []
for words in tokenized_y:
    try:
        embd = model1.wv.get_vector(words)
        vector.append(embd)
    except:
        pass
vector_y = np.add.reduce(vector).reshape(1, -1)



cosine_similarities = []
for v_x in word2vec_x:
    cosine_similarities.append(cosine_similarity(v_x,vector_y))
cosine_similarities = {x: cosine_similarities[x].item() for x in range(len(cosine_similarities))}
cosine_similarities = sorted(cosine_similarities.items(), key=operator.itemgetter(1), reverse=True)


res = []
for ind, value in cosine_similarities:
    if value > 0:
        res.append(data.headlines[ind])

 

st.title("Pakistani News...")


query = st.text_input("Enter Your Query", "Type Here ...")
 
# display the name when the submit button is clicked
# .title() is used to get the input text string
if(st.button('Submit')):
    result = name.title()
    st.success(result)
