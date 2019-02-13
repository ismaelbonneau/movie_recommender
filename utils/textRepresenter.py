# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from collections import Counter




class textRepresenter:

	"""

	"""

	def __init__(self, fromfile=False, lemmatise=True, frecDict=False):
		"""
		"""

		self.fromfile = fromfile
		self.lemmatise = lemmatise
		self.frecDict = frecDict

	def transform(self, input_):

		if self.fromfile:
			with open(input_, "r", encoding="utf-8") as f:
				text = (f.read()).lower()
		else:
			text = word_tokenize(input_.lower()) #convertir le texte en minuscule
		if not lemmatise:
			return text
		#sinon on lemmatise le texte
		tag_map = defaultdict(lambda : wn.NOUN)

		final_words = []
		word_Lemmatized = WordNetLemmatizer()
		for word, tag in pos_tag(text): #lemmatiser les mots et ne pas retenir les stopwords et la ponctuation 
			if word not in stopwords.words('english') and word.isalpha():
				word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
				final_words.append(word_Final)
		if not frecDict:
			return final_words #retourner une liste de mots
		else:
			return dict(Counter(final_words)) #retourner un dictionnaire du nombre d'occurences des mots
