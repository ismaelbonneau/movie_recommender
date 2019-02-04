1# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

import string
import nltk
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
import numpy as np


languages = ['english', 'french', 'spanish', 'russian', 'italian', 'portuguese', 'german', 'turkish', 'arabic']


def languageDetector(text):
	"""
	detecte le langage d'un texte.
	compare les stopwords du texte donné en entrée
	avec les stopwords de différentes langues.
	nécessite une entrée en utf-8
	"""

	tokens = [token.lower() for token in wordpunct_tokenize(text) if token not in string.punctuation ]
	fdist = nltk.FreqDist(tokens)

	response = []
	for language in languages:
		stpwords = stopwords.words(language) 
		text_stopwords_set = set()
		for word, frequence in fdist.most_common(len(stpwords)):
			text_stopwords_set.add(word)
		stopwords_set = set(stpwords)
		response.append(len(text_stopwords_set.intersection(stopwords_set)) / len(stopwords_set))

	return languages[np.argmax(np.array(response))]	

