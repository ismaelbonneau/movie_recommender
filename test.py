# !/usr/bin/python
# -*- coding: utf-8 -

from utils.load_data import load_data
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.corpus import stopwords
from classifiers.naiveBayes import NaiveBayes

path = "dataset"

listeseries = ["2567_House", "2956_Criminal_Minds", "1262_Lost", "175_The_Walking_Dead", "76_Breaking_Bad", "196_Smallville",
"217_Game_of_Thrones", "1906_The_Vampire_Diaries"]


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