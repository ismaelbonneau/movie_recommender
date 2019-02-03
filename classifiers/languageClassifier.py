# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

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



