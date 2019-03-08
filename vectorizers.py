#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:37:39 2019

@author: root

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def tfidf_vectorizer( iterable_tokens ):
    analyzer_ = lambda x: x
    # minimum document frequency
    min_docu_freq = 1
    
    vect = TfidfVectorizer(analyzer=analyzer_, min_df=min_docu_freq).fit(iterable_tokens[0])
    
    word_index = vect.vocabulary_
    Xencoded = vect.transform(iterable_tokens[1])
    
    return word_index, Xencoded
    
def count_vectorizer( iterable_tokens ):

    vect = CountVectorizer(lowercase=False).fit(iterable_tokens[0])
    
    word_index = vect.vocabulary_
    Xencoded = vect.transform(iterable_tokens[1])
    
    return word_index, Xencoded