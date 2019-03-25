
import glob
import numpy as np
from nltk.corpus import stopwords
#using https://github.com/amueller/word_cloud

from utils.load_data import getMostImportantSeries
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.tokenize import RegexpTokenizer
import string
nltk_stopw = stopwords.words('english')
import pickle
import gensim
from sklearn.metrics.pairwise import cosine_similarity
import tokenizers as tkns
import vectorizers as vcts


class CustomSim():
    
    def __init__(self, dataPath, nbSeries, w2vPath, alreadyTokenized = False, stopWords = stopwords.words('english')):
        self.path = dataPath
        self.w2vPath = w2vPath
        self.nbSeries = nbSeries
        seriesNames, _ = getMostImportantSeries(self.path, byepisodes=True)
        self.series = seriesNames[:self.nbSeries]
        self.nltk_stopw = stopWords
        
        self.tokenizer = self.tokenize_ex
        self.alreadyTokenized = alreadyTokenized
        self.vectorizer = self.tfidf_vectorizer
        self.combiner = self.sparseMultiply
        self.similarityMeasurer = cosine_similarity
        
    def setTokenizer(self, tokenizer):
        """ TOKENIZE FUNCTION SIGNATURE """
        # in : du texte brut ( un long str )
        # out : une liste de tokens ( liste de str )
        # cette fonction s'occupe des stop words, stemming, lemmatizing ... 
        # bref tout ce qu'il faut faire comme normalisation du texte
        self.tokenizer = tokenizer
        
    def loadW2VModel(self, filename_model_load):
        self.word_vector_model = gensim.models.Word2Vec.load(self.w2vPath+filename_model_load)
        print("Model Loaded :")
        print("\tVec size = "+str(self.word_vector_model.vector_size))
        self.vec_size = self.word_vector_model.vector_size
        print("\tWord Window = "+str(self.word_vector_model.window))
        print("\tWord min count = "+str(self.word_vector_model.vocabulary.min_count))
        print("\tTotal words = "+str(self.word_vector_model.corpus_total_words))
      
    def build_corpus_tokens(self,corpusFormat):
        for serie in self.series:
            listepisodes = glob.glob(self.path+"/"+serie+"/*/*"+corpusFormat)
            serie_tokens = []
            for episode in listepisodes:
                with open(episode, "r", encoding="utf-8") as file:
                    tokenized = self.tokenizer(file.read())
                    serie_tokens += tokenized
            yield np.array(serie_tokens)
            
    def fitW2VModel(self, vec_size, word_window, word_count_min, filename_model_save, corpusFormat = ".lines"):
        if( not self.alreadyTokenized ):
            st = SavingTokenizer(self.path, self.series, corpusFormat, self.tokenizer)
            st.build_tokens()
        
        series_yeilded = LoadingTokenizer(self.path, self.series, corpusFormat, self.tokenizer)
        
        self.vec_size = vec_size
        self.word_vector_model = gensim.models.Word2Vec(series_yeilded, size=vec_size, window=word_window, min_count=word_count_min)
        self.word_vector_model.save(self.w2vPath+filename_model_save)
        
    def setVectorizer(self, vectorizer):
        """ Custom Vectorizer """
        # pour faire un constructeur de vecteurs documents :
        # in : liste contenant N fois le meme itérateur sur les séries ( [ iterateur, iterateur , ... ])
        # out :
        #  word_index :
        #      A mapping of terms to feature indices. 
        #      ( dictionnaire : clé = mot, valeur = indice dans la colonne du vecteur )
        #  Xencoded : 
        #      document-term matrix ( matrice : lignes = docs, colonnes = mots de word_index)
        #      en gros chaque case[i,j] correspond au nombre ( int ? float ? ) qu'on associe au mot j dans le document i
        
        # Après il suffit de remplacer les variables word_index et Xencoded par les versions désirées
        self.vectorizer = vectorizer
        
    def getEmbeddingMatrix (self, word_index, embeddings):
        embedding_matrix = np.zeros((len(word_index)+1, self.vec_size))
        for word, i in word_index.items():
            if embeddings.wv.__contains__(word):
                embedding_matrix[i] = embeddings.wv.__getitem__(word)
        return embedding_matrix
    
    def setCombination( self, combiner ):
        """ Custom Doc Representation """
        # Il suffit de trouver une combinaison entre les données qu'on a ( tf-idf, tf, embedding de mots ... )
        # il n'y a pas de limites soyez créatifs xD
        # out : construire une matrice ou chaque ligne correspond à un document, 
        #       et tel que la ligne est un vecteur de taille vec_size
        #       il faut bien sure faire attention à préserver l'ordre des documents

        self.combiner = combiner
        
    def setSimilarityMeasure( self, similarityMeasurer ):
        """ Custom similarity measure """
        # in : une matrice ou chaque ligne correspond à un vecteur représentant un document ( l'ordre des documents preservé )
        # out : une matrice ou chaque ligne correspond à la similarité du document actuel à tous les autres documents
        self.similarityMeasurer = similarityMeasurer
        
    def buildVectors(self, fileSaveName, corpusFormat = ".lines", nbIterators = 2):
        l = [LoadingTokenizer(self.path, self.series, corpusFormat, self.tokenizer, numpify = True) for i in range(nbIterators)]
        self.word_index, self.Xencoded = self.vectorizer(l)
        with open(self.w2vPath+fileSaveName+"WI", "wb") as f:
            pickle.dump(self.word_index, f)
        with open(self.w2vPath+fileSaveName+"Xe", "wb") as f:
            pickle.dump(self.Xencoded, f)
        
    def loadVectors(self, fileLoadName):
        with open(self.w2vPath+fileLoadName+"WI", "rb") as f:
            self.word_index = pickle.load(f)
        with open(self.w2vPath+fileLoadName+"Xe", "rb") as f:
            self.Xencoded = pickle.load(f)
        
    def combineVectEmbeddings(self, fileSaveName):
        # création de la matrice embedding à partir du model word2vec
        embedding_matrix = self.getEmbeddingMatrix(self.word_index, self.word_vector_model)
        # produit matriciel entre cette matrice et les tf-idf
        self.Xcombined = self.combiner( self.Xencoded, embedding_matrix)
        with open(self.w2vPath+fileSaveName, "wb") as f:
            pickle.dump(self.Xcombined, f)
        
    def calculateSimilarities( self, fileSaveName ):
        self.sim = self.similarityMeasurer(self.Xcombined)
        np.save(self.w2vPath+fileSaveName+'.npy', self.sim)
        return self.sim
    
    def tokenize_ex(self, text):
        #   no punctuation & starts with a letter & between 3-15 characters in length
        tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
        return [f.lower() for f in tokens if f and f.lower() not in self.nltk_stopw]
    
    def tfidf_vectorizer(self, iterable_tokens ):
        analyzer_ = lambda x: x
        # minimum document frequency
        min_docu_freq = 1
        
        vect = TfidfVectorizer(analyzer=analyzer_, min_df=min_docu_freq).fit(iterable_tokens[0])
        
        word_index = vect.vocabulary_
        Xencoded = vect.transform(iterable_tokens[1])
        
        return word_index, Xencoded
    
    def sparseMultiply (self, sparseX, embedding_matrix):
        denseZ = []
        for row in sparseX:
            newRow = np.zeros(self.vec_size)
            for nonzeroLocation, value in list(zip(row.indices, row.data)):
                newRow = newRow + value * embedding_matrix[nonzeroLocation]
            denseZ.append(newRow)
        denseZ = np.array([np.array(xi) for xi in denseZ])
        return denseZ

class WordTrainer(object):
    def __init__(self, path, series, corpusFormat, tokenizer):
        self.path = path
        self.series = series
        self.corpusFormat = corpusFormat
        self.tokenizer = tokenizer
    def __iter__(self):
        for serie in self.series:
            listepisodes = glob.glob(self.path+"/"+serie+"/*/*"+self.corpusFormat)
            for episode in listepisodes:
                with open(episode, "r", encoding="utf-8") as file:
                    tokenized = self.tokenizer(file.read())
                    yield tokenized

class SavingTokenizer(object):
    def __init__(self, path, series, corpusFormat, tokenizer):
        self.path = path
        self.series = series
        self.corpusFormat = corpusFormat
        self.tokenizer = tokenizer
    def build_tokens(self):
        for serie in self.series:
            listepisodes = glob.glob(self.path+"/"+serie+"/*/*"+self.corpusFormat)
            serie_tokens = []
            for episode in listepisodes:
                with open(episode, "r", encoding="utf-8") as file:
                    tokenized = self.tokenizer(file.read())
                    serie_tokens += tokenized
#            tmp = np.array(serie_tokens)
#            print("saving in"+self.path+'/'+serie+"/"+self.tokenizer.__name__+'.npy')
#            np.save(self.path+'/'+serie+"/"+self.tokenizer.__name__+'.npy', tmp)
            with open(self.path+'/'+serie+"/"+self.tokenizer.__name__, "wb") as f:
                pickle.dump(serie_tokens, f)

class LoadingTokenizer(object):
    def __init__(self, path, series, corpusFormat, tokenizer, numpify = False ):
        self.path = path
        self.series = series
        self.corpusFormat = corpusFormat
        self.tokenizer = tokenizer
        self.numpify = numpify
    def __iter__(self):
        for serie in self.series:
#            print("loading "+self.path+'/'+serie+"/"+self.tokenizer.__name__)
            with open(self.path+'/'+serie+"/"+self.tokenizer.__name__, "rb") as f:
                tmp = pickle.load(f)
#            tmp = np.load(self.path+'/'+serie+"/"+self.tokenizer.__name__+'.npy')
            if self.numpify:
                yield np.array(tmp)
            else:
                yield tmp