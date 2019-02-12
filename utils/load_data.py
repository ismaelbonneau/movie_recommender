# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara


#fichier contenant les fonctions de chargement des données


import os
from sklearn.model_selection import train_test_split

def load_data(path, series=[], random=True, split=True, ratio=0.8):
	
	if series != []:
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
			return train_test_split(X, Y, test_size=0.2)

		else:
			return X, Y
	else:
		#TODO
		pass