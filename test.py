# !/usr/bin/python
# -*- coding: utf-8 -

import os
from utils.load_data import load_data
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from classifiers.naiveBayes import NaiveBayes
from utils.featureExtraction import wordLemmatizer



path = "dataset"

listeseries = ["2567_House", "2956_Criminal_Minds", "1262_Lost", "175_The_Walking_Dead", "76_Breaking_Bad", "196_Smallville",
"217_Game_of_Thrones", "1906_The_Vampire_Diaries", "3280_Peaky_Blinders", "1704_Rick_and_Morty", "1039_Narcos",
              "1845_Sherlock_(2010)", "1701_Outlander", "3314_Shameless", "818_Gomorra_(2014)", "413_Dexter",
              "2123_Sense8", "121_Suits", "2469_The_Simpsons", "1718_South_Park", "3259_Stargate_SG-1"]

fe = wordLemmatizer()

for serie in os.listdir(path):
	if os.path.basename(serie) not in listeseries:
		for saison in os.listdir(os.path.join(path, serie)):
			for ep in os.listdir(os.path.join(path+os.sep+serie,saison)):
				fname = os.path.join(path+os.sep+serie+os.sep+saison, ep)
				with open(fname, "r", encoding="utf-8") as f:
					x = fe.transform(f.read())
				with open(path+os.sep+serie+os.sep+saison+os.sep+os.path.basename(os.path.splitext(ep)[0])+".tokens", "w") as t:
					t.write(x)
	print(os.path.basename(serie), " done.")

"""

listeseries, (X_train, X_test, y_train, y_test) = load_data(path, nbclass=30, random=False, split=True, ratio=0.8)


#conversion des données sous forme vectorielle - nombre d'occurence
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stopwords.words('english'), strip_accents='ascii')
count_vect.fit(X_train + X_test)

xtrain_count = count_vect.transform(X_train)
xtest_count = count_vect.transform(X_test)


print("countVector model:")

NB = NaiveBayes()
accuracy, predictions = NB.fit(xtrain_count, y_train, xtest_count, y_test)
NB.getErrors(y_test, listeseries)

"""
