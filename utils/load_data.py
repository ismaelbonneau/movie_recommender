# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara


#fichier contenant les fonctions de chargement des données

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split



def first(iterable, condition = lambda x: True):
	"""
    Returns the first item in the `iterable` that
    satisfies the `condition`.

    thanks to Caridorc 
    https://stackoverflow.com/questions/2361426/get-the-first-item-from-an-iterable-that-matches-a-condition/2361899s
    """
	return next(x for x in iterable if condition(x))

def getMostImportantSeries(path, byepisodes=True):
	"""
	retourne les séries ordonnées par leur nombre d'épisode décroissant
	"""
	listseries = glob.glob(path+os.sep+"*")

	series = []
	count = []
	for serie in listseries:

		series.append(os.path.basename(serie))
		if byepisodes:
			#compter le nombre d'épisodes de la série
			count.append(len(glob.glob(serie+os.sep+"*"+os.sep+"*.lines")))
		else:
			#compter le nombre de saisons de la série
			count.append(len(glob.glob(serie+os.sep+"*")))

	#trier les 2 listes sur la base du nombre de saisons/d'épisodess
	count, series = (list(t) for t in zip(*sorted(zip(count, series))))

	#retourner les listes triées dans l'ordre décroissant
	return series[::-1], count[::-1]


def load_data(path, series=[], nbclass=10, random=True, split=True, ratio=0.8):
	"""

	"""
	if series == []:
		#si aucune série n'est spécifiée, on tire des séries au hasard
		listseries, count = getMostImportantSeries(path)
		borne = count.index(first(count, lambda i: i < 10))
		del count

		series_index = np.random.choice(borne - 1, nbclass, replace=False)
		classe = 0
		X = []
		Y = []
		for i in series_index:
			for ep in glob.glob(path+os.sep+listseries[i]+os.sep+"*"+os.sep+"*.lines"):
				with open(ep, "r", encoding="utf-8") as f:
					X.append(f.read())
					Y.append(classe)
			classe += 1

		if split:
			return [listseries[i] for i in series_index], train_test_split(X, Y, test_size=(1. - ratio))
		else:
			return [listseries[i] for i in series_index], (X, Y)
	else:
		X = []
		Y = []
		i = 0
		for serie in series:
			#print(os.path.basename(serie))
			for saison in os.listdir(os.path.join(path, serie)):
				for ep in os.listdir(os.path.join(path+os.sep+serie,saison)):
					with open(os.path.join(path+os.sep+serie,saison+os.sep+ep), "r", encoding="utf-8") as f:
						X.append(f.read())
					Y.append(i)
			i += 1

		if split:
			return train_test_split(X, Y, test_size=(1. - ratio))

		else:
			return X, Y



