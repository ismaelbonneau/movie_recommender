# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara


#fichier contenant les fonctions de chargement des données

import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split



def getMostImportantSeries(path, byepisodes=True):
	"""
	retourne les séries ordonnées par leur nombre d'épisode décroissant
	"""
	listseries = glob.glob(path)

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


def load_data(path, series=[], random=True, split=True, ratio=0.8):
	"""

	"""
	if series == []:
		#TODO
		pass

	else:
		X = []
		Y = []
		i = 0
		for serie in series:
			#print(os.path.basename(serie))
			for saison in os.listdir(os.path.join(path, serie)):
				for ep in os.listdir(os.path.join(path+os.sep+serie,saison)):
					with open(os.path.join(path+os.sep+serie,saison+os.sep+ep), encoding="utf-8") as f:
						X.append(f.read())
					Y.append(i)
			i += 1

		if split:
			return train_test_split(X, Y, test_size=(1. - ratio))

		else:
			return X, Y



countepisodes, series = getMostImportantSeries("../dataset/*")
print(countepisodes[-1])
print(series[-1])