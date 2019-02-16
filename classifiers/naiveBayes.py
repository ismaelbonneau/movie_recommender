# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

from collections import Counter
from utils.featureExtraction import wordLemmatizer
from sklearn import preprocessing, naive_bayes, metrics
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import operator
import matplotlib.pyplot as plt

class NaiveBayes:
	"""

	"""

	def __init__(self):

		self.classifier = naive_bayes.MultinomialNB()
		self.encoder = preprocessing.LabelEncoder()


	def fit(self, X_train, y_train, X_test, y_test):
		"""
		Entraine le classifieur et evalue sur les donnees de test
		"""
		#encoding des vecteurs d'expectation 

		train_y = self.encoder.fit_transform(y_train)
		test_y = self.encoder.fit_transform(y_test)
		#entrainement du classifieur
		self.classifier.fit(X_train, train_y)
		#evaluation
		self.predictions = self.classifier.predict(X_test)
		accuracy = metrics.accuracy_score(self.predictions, test_y)

		print("trained on {} samples, validated on {}.".format(X_train.shape[0], X_test.shape[0]))
		print("accuracy: ", accuracy)

		return self.predictions, accuracy


	def getErrors(self, labels, series):

		#renommage pour affichage plus propre
		nomseries = [" ".join(x.split("_")[1:]) for x in series]
		prout = [nomseries[i] for i in labels]
		prout = dict(Counter(prout))
		#au cas où
		self.predictions = np.array(self.predictions)
		labels = np.array(labels)
		#les épisodes pour lesquels le classieur s'est trompé
		erreurs = self.predictions != labels
		errors_dict = {}
		for i, v in enumerate(erreurs):
			if v:
				nom = nomseries[labels[i]]
				if nom not in errors_dict:
					errors_dict[nom] = 1
				else:
					errors_dict[nom] += 1

		#pour exprimer en pourcentage d'épisodes mal classifés
		for x in errors_dict:
			errors_dict[x] /= prout[x]
		errors_dict = sorted(errors_dict.items(), key=operator.itemgetter(1), reverse=True)
		names = [x[0] for x in errors_dict]
		errors = [x[1] for x in errors_dict]
		#bar chart du nombre d'erreur pour chaque série
		plt.figure(figsize=(20,10))
		plt.ylabel("pourcentage de mauvaise classification")
		plt.title("pourcentage de mauvaise classification pour chaque serie")
		plt.bar(range(len(errors_dict)), errors, width=0.35)
		plt.xticks(range(len(errors_dict)), names, rotation='vertical')
		plt.show()

		return errors_dict
