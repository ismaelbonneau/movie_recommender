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
import glob


class NEextractor():
    """
    classe permettant d'extraire les personnages d'une série
    """

    def __init__(self):
        """
        """
        pass
    def extract(self, path_to_dataset, series, includeAll=True, globalcutoff=30, individualcutoff=10):
        """
        réalise l'extraction
        globalcutoff: nombre de mots à retenir parmi les mots ayant un df de 1 en considerant tout le corpus
        individualcutoff: nombre de mots à retenir parmi les mots ayant un df de 1 en considerant 2 docs
        """
        #chargement du corpus
        customstopwords = stopwords.words('english') + ["yes","yeah","hmm","hey","nah","oh","uh","okay","okey","ye","hey","na","ca","ok","got","come","back","up"]
        corpus = []
        for serie in series:
            listepisodes = glob.glob(path_to_dataset+"/"+serie+"/*/*.tokens") #texte lemmatisé!
            text = ""
            for episode in listepisodes:
                with open(episode, "r", encoding="utf-8") as file:
                    text += " " + file.read()
            corpus.append(text)

            
        vectorizer = TfidfVectorizer(lowercase=True, 
                             binary=False,
                             analyzer='word',
                             use_idf = True,
                             stop_words=customstopwords,
                             max_df = 1)

        X = vectorizer.fit_transform(corpus)
        terms = vectorizer.get_feature_names()

        words_per_serie = {}

        for serie in range(len(corpus)):
            feature_index = X[serie, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [X[serie, x] for x in feature_index])
            doc_terms = []
            for word, score in [(terms[i], score) for (i, score) in tfidf_scores]:
                doc_terms.append((word, score))
            important_terms = [word for word, score in sorted(doc_terms, key=lambda x: x[1], reverse=True)][:globalcutoff]
            words_per_serie[series[serie]] = important_terms

        for serie in range(len(corpus)):
            vectorizer = TfidfVectorizer(lowercase=True,binary=False,analyzer='word',use_idf=True,stop_words=customstopwords,max_df=1)
            X = vectorizer.fit_transform([corpus[serie], corpus[(serie+1)%(len(corpus))], corpus[(serie+2)%(len(corpus))]])
            terms = vectorizer.get_feature_names()
            
            feature_index = X[0, :].nonzero()[1]
            tfidf_scores = zip(feature_index, [X[0, x] for x in feature_index])
            doc_terms = []
            for word, score in [(terms[i], score) for (i, score) in tfidf_scores]:
                doc_terms.append((word, score))
            important_terms = [word for word, score in sorted(doc_terms, key=lambda x: x[1], reverse=True)][:individualcutoff]
            words_per_serie[series[serie]] = list(set(words_per_serie[series[serie]]) | set(important_terms))

        self.words = words_per_serie
        return words_per_serie

    def save(self, filename):
    	"""
		enregistrer le fichier de personnages sur le disque
    	"""
    	with open(filename, "w", encoding="utf-8") as file:
    		for serie in self.words:
    			for mot in self.words[serie]:
    				file.write(mot+"\n")


class wordLemmatizer():
	"""
	classe permettant de lemmatiser un texte
	"""

	def __init__(self, fromfile=False):

		self.fromfile = fromfile

	def transform(self, input_):
		if self.fromfile:
			with open(input_, "r", encoding="utf-8") as f:
				text = (f.read()).lower()
		else:
			text = word_tokenize(input_.lower()) #convertir le texte en minuscule
		tag_map = defaultdict(lambda : wn.NOUN)
		final_words = []
		word_Lemmatized = WordNetLemmatizer()
		for word, tag in pos_tag(text): #lemmatiser les mots et ne pas retenir les stopwords et la ponctuation 
			if word not in stopwords.words('english') and word.isalpha():
				word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
				final_words.append(word_Final)
		return " ".join(final_words)

class wordFreqDict():
	"""
	classe permettant de lemmatiser un texte et de retourner un modèle BOW
	"""

	def __init__(self, fromfile=False, lemmatise=True, frecDict=True):
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