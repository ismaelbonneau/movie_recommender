# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara

import sys
import os
sys.path.append(os.path.abspath('../classifiers'))

import glob
import codecs
import languageClassifier

codecs.register_error("strict", codecs.ignore_errors)

#dataset path
path = "/home/ismael/Documents/movie_recommender/dataset"


pasAnglais = []
#parcours de toutes les séries
for serie in os.listdir(path):
	
	for season in os.listdir(os.path.join(path,serie)):
		if os.path.isdir(os.path.join(path,serie+os.sep+season)):

			episodes = glob.glob(os.path.join(path,serie+os.sep+season+os.sep+"*.lines"))
			for episode in episodes:
				with codecs.open(episode, "r", "utf-8") as file:
					text = file.read()
					if languageClassifier.languageDetector(text) != "english":
						pasAnglais.append(episode)
				
print('nombre de séries pas en anglais: ', len(pasAnglais))

with open("pasAnglais.txt", "w") as f:
	for serie in pasAnglais:
		f.write(serie+'\n')