# !/usr/bin/python
# -*- coding: utf-8 -

from collections import Counter
from utils.featureExtraction import wordLemmatizer
from utils.load_data import load_data
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import preprocessing, linear_model, naive_bayes, metrics, svm
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def train_model(classifier, X_train, y_train, X_test, y_test):
    
	classifier.fit(X_train, y_train)

	predictions = classifier.predict(X_test)
    
	accuracy = metrics.accuracy_score(predictions, y_test)

	print("trained on {} samples, validated on {}.".format(X_train.shape[0], X_test.shape[0]))
	print("accuracy: ", accuracy)

	return accuracy, predictions

def getErrors(predictions, labels, series):
	erreurs = predictions != labels

	errors_dict = []
	for i, v in enumerate(erreurs):
		if v:
			errors_dict.append(series[y_test[i]])
	return dict(Counter(errors_dict))

path = "dataset"

listeseries = ["2567_House", "2956_Criminal_Minds", "1262_Lost", "175_The_Walking_Dead", "76_Breaking_Bad", "196_Smallville",
"217_Game_of_Thrones", "1906_The_Vampire_Diaries"]


listeseries, (X_train, X_test, y_train, y_test) = load_data(path, nbclass=70, random=False, split=True, ratio=0.8)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stopwords.words('english'), strip_accents='ascii')
count_vect.fit(X_train + X_test)

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(y_train)
test_y = encoder.fit_transform(y_test)

xtrain_count = count_vect.transform(X_train)
xtest_count = count_vect.transform(X_test)

print("countVector model:")
accuracy, predictions = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xtest_count, test_y)

predictions = np.array(predictions)
y_test = np.array(y_test)

errors_dict = getErrors(predictions, y_test, listeseries)

import pprint

pp = pprint.PrettyPrinter(indent=4)

#pp.pprint(errors_dict)


tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(X_train + X_test)

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(X_train)
xtest_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)

print("tf-idf n-gram model:")
accuracy, predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xtest_tfidf_ngram, test_y)

predictions = np.array(predictions)
y_test = np.array(y_test)

errors_dict = getErrors(predictions, y_test, listeseries)
#pp.pprint(errors_dict)


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(X_train + X_test)

xtrain_tfidf =  tfidf_vect.transform(X_train)
xtest_tfidf =  tfidf_vect.transform(X_test)

print("tf-idf word model:")
accuracy, predictions = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xtest_tfidf, test_y)

predictions = np.array(predictions)
y_test = np.array(y_test)

errors_dict = getErrors(predictions, y_test, listeseries)
#pp.pprint(errors_dict)