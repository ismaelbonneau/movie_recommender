# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

import codecs
import string
import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import numpy as np


languages = ['english', 'french', 'spanish', 'russian', 'italian', 'portuguese', 'german', 'turkish', 'arabic']

def quickLanguageDetector(text):
	"""
	detecte le langage d'un texte.
	marche de façon très simple: regarde combien de stopwords
	d'un langage donné le texte contient.
	moins précis qu'un calcul de stopwords sur le texte mais plus rapide
	nécessite une entrée en utf-8
	"""
	tokens_set = wordpunct_tokenize(text)
	tokens_set = set([token.lower() for token in tokens_set])

	response = np.array([0,0,0,0,0,0,0,0,0])
	i = 0
	for language in languages:
		stopwords_set = set(stopwords.words(language))
		response[i] = len(tokens_set.intersection(stopwords_set))
		i+=1
	return languages[np.argmax(response)]


def languageDetector(text):
	"""
	detecte le langage d'un texte.
	compare les stopwords du texte donné en entrée
	avec les stopwords de différentes langues.
	nécessite une entrée en utf-8
	"""

	tokens = [token.lower() for token in wordpunct_tokenize(text) if token not in string.punctuation ]
	fdist = nltk.FreqDist(tokens)

	print(fdist.most_common(30))
	response = np.array([0,0,0,0,0,0,0,0,0])
	i = 0
	for language in languages:
		stpwords = stopwords.words(language)
		text_stopwords_set = set()
		for word, frequence in fdist.most_common(len(stpwords)):
			text_stopwords_set.add(word)
		stopwords_set = set(stopwords.words(language))
		response[i] = len(text_stopwords_set.intersection(stopwords_set))
		i+=1
	return languages[np.argmax(response)]	




