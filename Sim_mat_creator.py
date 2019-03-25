from nltk.corpus import stopwords
#using https://github.com/amueller/word_cloud

nltk_stopw = stopwords.words('english')
from sklearn.metrics.pairwise import cosine_similarity
import tokenizers as tkns
import vectorizers as vcts
import CustomSimilarity as cs

""" The variations """
# variation 1 :
#     nbSeries = 100
#     vec_size = 50
#     word_window = 8
#     word_count_min = 5
#     tokenizer = tokenize_simple
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 2 :
#     nbSeries = 100
#     vec_size = 100
#     word_window = 8
#     word_count_min = 5
#     tokenizer = tokenize_simple
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 3 :
#     nbSeries = 100
#     vec_size = 200
#     word_window = 8
#     word_count_min = 5
#     tokenizer = tokenize_simple
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 4 :
#     nbSeries = 100
#     vec_size = 50
#     word_window = 8
#     word_count_min = 5
#     tokenizer = porterStemmer_stopWRemove
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 5 :
#     nbSeries = 100
#     vec_size = 100
#     word_window = 8
#     word_count_min = 5
#     tokenizer = porterStemmer_stopWRemove
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 6 :
#     nbSeries = 100
#     vec_size = 200
#     word_window = 8
#     word_count_min = 5
#     tokenizer = porterStemmer_stopWRemove
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 7 :
#     nbSeries = 100
#     vec_size = 50
#     word_window = 8
#     word_count_min = 5
#     tokenizer = wordNetLemmatizer_stopWRemove
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 8 :
#     nbSeries = 100
#     vec_size = 100
#     word_window = 8
#     word_count_min = 5
#     tokenizer = wordNetLemmatizer_stopWRemove
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 9 :
#     nbSeries = 100
#     vec_size = 200
#     word_window = 8
#     word_count_min = 5
#     tokenizer = wordNetLemmatizer_stopWRemove
#     vectorizer = tf-idf ( min df = 1 )
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 10 :
#     nbSeries = 100
#     vec_size = 50
#     word_window = 8
#     word_count_min = 5
#     tokenizer = tokenize_simple
#     vectorizer = count_vectorizer
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 11 :
#     nbSeries = 100
#     vec_size = 100
#     word_window = 8
#     word_count_min = 5
#     tokenizer = tokenize_simple
#     vectorizer = count_vectorizer
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# variation 12 :
#     nbSeries = 100
#     vec_size = 200
#     word_window = 8
#     word_count_min = 5
#     tokenizer = tokenize_simple
#     vectorizer = count_vectorizer
#     combiner = multiplication ( multiply each word's tf-idf with the word's embedding, then summing all vectors )
#     similarityMeasure = cosine_similarity

# this is just for SAVING, do not try to load with this it will just overwrite
variation = 12
import time

start = time.time()

""" THE PIPELINE """        

""" initializing parameters """

# dataset path
path = "/root/Documents/PLDAC/data"
# repertoir pour stocker les calculs de word2vec et les fichiers resultats
w2vPath = "/root/Documents/PLDAC/Word2VecData/"
# nombre de series que l'on veut étudier
nbSeries = 100
# taille du vecteur embedding des mots
vec_size = 200

# nb de mots voisins pris en comptes lors de l'embedding
word_window = 8
# ignorer les mot ayant une occurence inférieur à ça dans tout le corpus
word_count_min = 5
# nom du fichier de sauvegarde du model word2vec
filename_model_save = "w2v_model_"+str(variation)
# set to True if the tokenization used has already been calculated
alreadyTokenized = True

print("Initializing")
obj = cs.CustomSim(path, nbSeries, w2vPath, alreadyTokenized = alreadyTokenized)
# désignation du tokenizer
obj.setTokenizer(tkns.tokenize_simple)
# désignation du vectorizer
obj.setVectorizer(vcts.count_vectorizer)
# désignation du combiner
obj.setCombination(obj.combiner)
# désignation de la mesure de similarité
obj.setSimilarityMeasure(cosine_similarity)


""" Calculations """

# soit construire le model, ou charger un model déja calculé
print("Word2Vec Calculations")
obj.fitW2VModel(vec_size, word_window, word_count_min, filename_model_save)
#print("loading W2V Model")
#obj.loadW2VModel(filename_model_save)

# soit construire les vecteurs, ou charger des vecteurs déja calculés
#print("Building Vectors")
#fileSaveName = "Vectors_"+str(variation)+"_"
#obj.buildVectors(fileSaveName)
print("Loading Vectors")
fileSaveName = "Vectors_10_"
obj.loadVectors(fileSaveName)

print("Combining")
fileSaveName = "Combined_"+str(variation)+"_"
obj.combineVectEmbeddings(fileSaveName)

print("Calculating similarities")
fileSaveName = "sim_"+str(variation)+"_"
sim = obj.calculateSimilarities(fileSaveName)
series = obj.series


""" Showing results """

# Affichage de la matrice de similarités
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

plt.figure(figsize=(10,10))
plt.imshow(sim, cmap='YlGn', interpolation='nearest')
plt.xticks(range(len(series)), [" ".join(x.split("_")[1:]) for x in series], rotation='vertical')
plt.yticks(range(len(series)), [" ".join(x.split("_")[1:]) for x in series])
plt.show()

end = time.time()

print("Time")
print(end-start)