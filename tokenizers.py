#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:26:26 2019

@author: root
"""
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

import string

def porterStemmer_stopWRemove( text ):
    ps = PorterStemmer()
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,20}\b').tokenize(text)]
    return [ps.stem(f.lower()) for f in tokens if f and f.lower() not in stopwords.words('english')]

def wordNetLemmatizer_stopWRemove( text ):
    lm = WordNetLemmatizer()
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,20}\b').tokenize(text)]
    return [lm.lemmatize(f.lower()) for f in tokens if f and f.lower() not in stopwords.words('english')]

def tokenize_simple( text):
        #   no punctuation & starts with a letter & between 3-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return [f.lower() for f in tokens if f and f.lower() not in stopwords.words('english')]