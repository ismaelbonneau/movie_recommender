import pandas as pd
import gensim
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
import pickle
import random
class HybridSimilarities():
    
    def __init__(self, ratings, titles, simMat, keep):
        self.ratings = ratings
        self.titles = titles
        self.simMat = simMat
        self.keep = keep
        
    def buildClusters(self, new):
        result = []
        tt = new.T
        for i in range(len(new)):
            notes = dict()
            for note in range(0,22):
                notes[note] = []
            for name, rating in new.iloc[i].dropna().iteritems():
                name = str(name)
                notes[rating].append(name)
                notes[rating+1].append(name)
                notes[rating-1].append(name)
            result += list(filter(None,notes.values()))
        return result
      
      
    def train(self,collaboratif = 1, similarities = 1, vecSize = 50, window = 1000, iter = 100, workers = 4, min_count = 1):
        
        cl = self.buildClusters(self.ratings)
        
        model = gensim.models.Word2Vec(cl, size=50, window=1000, iter=100, workers=4, min_count=1)
        
        simMat = self.simMat
        keep = self.keep
        keep = keep.loc[sorted(list(map(int,list(model.wv.vocab.keys()))))]
        keep['count'] = list(range(len(keep)))
        self.newKeep = keep
        keeps = keep['index'].values
        simMat = simMat[keeps][:,keeps]
        self.contentSims = simMat
        series = keep['seriesname'].values
        
        length  = len(keep)
        vectors = []
        for serie in keep.index:
            #print(titles.iloc[serie].title)
            vec = model.wv.get_vector(str(serie))
            vectors.append(vec)

        from sklearn.metrics.pairwise import cosine_similarity
        colabSims = ((cosine_similarity(vectors)+1)/2)
        self.colabSims = colabSims
        
        hybrid = (np.array(self.colabSims)*collaboratif + np.array(self.contentSims)*similarities) / (collaboratif+similarities)
        self.hybrid = hybrid
        
        return hybrid
      
    
    def calibrate(self, collaboratif, similarities):
        hybrid = (np.array(self.colabSims)*collaboratif + np.array(self.contentSims)*similarities) / (collaboratif+similarities)
        self.hybrid = hybrid
        
    def recommend(self, serieId, k = 10 ):
        
        keep = self.newKeep
        hybrid = self.hybrid
        
        l = np.argsort(hybrid[serieId])[::-1][:k]
        k = keep.iloc[l]
        res = pd.concat([k, pd.DataFrame(df.iloc[k.index].mean(axis=1), columns=['Rating'])], axis=1, sort=False)
        
        return res
      
    def recommendUser(self, username, minRate = 15, k = 5):
        userSeries = showUserOriginal(self.ratings.T, username, minRate = minRate)
        df_all = pd.DataFrame(columns=['Unnamed: 0', 'seriesname', 'title',	'index',	'count',	'Rating', 'score', 'weightedScore'])
        for id_ in userSeries.index:
            simId = self.showSeries().loc[id_]['count']
            reco = self.recommend(simId,k+1)
            l = reco['count'].values
            tmp = self.hybrid[simId][l]
            score = pd.concat([reco, pd.DataFrame(tmp, columns=['score'], index=reco.index)], axis=1, sort=False)
            score['weightedScore'] = score['score']*userSeries.loc[id_]['rate']
            score.drop(columns=['score','Rating'])
            score = score[1:]
            df_all = pd.concat([df_all, score], axis=0, sort=False)

        grouped = df_all.groupby(['Unnamed: 0', 'seriesname', 'title',	'index',	'count'], as_index=False).max()
        grouped.index = grouped['Unnamed: 0']
        for p in userSeries.index:
            grouped.drop(p, inplace=True, errors='ignore')    
        grouped = grouped.sort_values(by='weightedScore', ascending=False, )[:10]
        return grouped

    def showSeries(self):
        
        return self.newKeep
      
    def showUserOriginal(self, df, user, minRate = 0):
        kiko = df[user].dropna()
        kaka = titles.loc[kiko.index]
        kaka['rate'] = kiko
        return kaka[kaka.rate >= minRate]