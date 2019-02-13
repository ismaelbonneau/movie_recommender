# !/usr/bin/python
# -*- coding: utf-8 -

from utils.featureExtraction import wordLemmatizer
from utils.load_data import load_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, linear_model, naive_bayes, metrics, svm
import numpy as np

from nltk.corpus import stopwords


def train_model(classifier, feature_vector_train, label, feature_vector_valid, test_y):
    
	classifier.fit(feature_vector_train, label)

	predictions = classifier.predict(feature_vector_valid)
    
	return metrics.accuracy_score(predictions, test_y)

path = "dataset"

listeseries = ["2567_House", "2956_Criminal_Minds", "1262_Lost", "175_The_Walking_Dead", "76_Breaking_Bad", "196_Smallville",
"217_Game_of_Thrones", "1906_The_Vampire_Diaries"]


series, (X_train, X_test, y_train, y_test) = load_data(path, nbclass=40, split=True, ratio=0.8)

count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words=stopwords.words('english'), strip_accents='ascii')
count_vect.fit(X_train + X_test)

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(y_train)
test_y = encoder.fit_transform(y_test)

xtrain_count = count_vect.transform(X_train)
xtest_count = count_vect.transform(X_test)

print(len(X_train), xtrain_count.shape)
print(len(X_test), xtest_count.shape)


bayes = naive_bayes.MultinomialNB()
bayes.fit(xtrain_count, train_y)
predictions = bayes.predict(xtest_count)

accuracy = metrics.accuracy_score(predictions, test_y)
print("accuracy: "accuracy)

predictions = np.array(predictions)
y_test = np.array(y_test)

erreurs = predictions != y_test
