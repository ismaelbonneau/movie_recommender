import pandas as pd
import gensim
import numpy as np
from sklearn.metrics import mean_squared_error
import warnings
import pickle
import random
warnings.simplefilter(action='ignore', category=FutureWarning)

class Item2Vec:
  
    def __init__(self, userratings_fn, namesConvert_fn, titles_fn, keep_fn, simMat_fn ):
      
        # Dataframe : line 'serie', column 'user', cell 'Rating/10 or NaN'
        df = pd.read_csv(userratings_fn).drop('Unnamed: 0', axis='columns')
        # Dataframe to convert between the different namings of series
        nameConvert = pd.read_csv(namesConvert_fn)
        # Series titles ordered by the same order as in the ratings csv
        titles = pd.read_csv(titles_fn)
        
        # Loading the similarity matrix
        simMat = np.load(simMat_fn)
        
        # Keep.csv contains the lines/columns to keep from the similarity matrix
        # keep only the ones that match the series we have in userratings.csv
        keep = pd.read_csv(keep_fn)
        keeps = keep['index'].values
        simMat = simMat[keeps][:,keeps]
        series = keep['seriesname'].values
        

        
        # SELFS !
        self.df = df
        self.titles = titles
        self.simMat = simMat
     
    def __init__(self, df, titles, simMat ):
      
        # SELFS !
        self.df = df
        self.titles = titles
        self.simMat = simMat
        
    # Corpus building
    # both functions take a dataframe : line 'serie', column 'user', cell 'rate'

    # Builds each document as the usernames ordered from the highest rating one to the lowest rating one for each serie
    # ie:
    # we get nbSeries document
    # each document is the sequence of all the users that rated it, ordered from the ones who liked it to the the ones who hated it
    #           user C     user D     user A      user P
    # serie X     10         8          8          3
    # the document will be [ 'C', 'D', 'A', 'P' ]
    def buildSequences(self, new):     
        result = dict()
        dfs = dict()
        for index, row in new.iterrows():
            x = row.dropna()
            y = pd.Series.argsort(x)
            p = np.array(x.axes).ravel()
            cols = np.flip(p[y]).tolist()
            result[index] = cols
            dfs[index] = x[cols].values
        return result, dfs


    # Builds each document as the cluster of users that gave the same rating to the same serie
    # we get nbSeries*10 maxiumum documents ( since there is 10 grades to give )
    # each document is the usernames of those who gave the exact same rating to a given serie
    #           user C     user D     user A      user P
    # serie X     10         8          8          3
    # serie Y      2         10         2          2
    # docs:
    # [ 'C' ] who gave 10 to serie X
    # [ 'D', 'A'] who gave 8 to serie X
    # [ 'P' ] who gave 3 to serie X
    # [ 'C', 'A', 'P' ] who gave 2 to serie Y
    # [ 'D' ] who gave 10 to serie Y
    # so we treat each serie alone and cluster its users based on their ratings

    def buildClusters(self, new):
        result = []
        tt = new.T
        for i in range(len(new)):
            result += list(map(list, list(tt.groupby([i]).groups.values())))
        return result
    
    # Same but groups with an interval of +1 and -1 too
    def buildClusters2(self, new):
        result = []
        tt = new.T
        for i in range(len(new)):
            notes = dict()
            for note in range(0,22):
                notes[note] = []
            for name, rating in new.iloc[i].dropna().iteritems():
                notes[rating].append(name)
                notes[rating+1].append(name)
                notes[rating-1].append(name)
            result += list(filter(None,notes.values()))
        return result
      
    
    # User similarity calculations
    # both functions input :
    #    model : w2v model
    #    topn : how many users to consider as similar
    #    df : the dataframe containing the ratings
    #    cross : how many series in common we want 
    # output :
    #    dictionary : key 'user', value 'list of users'


    # Returns for each user, the list of users that are 'similar' to it
    # this version calculates how many series the two users have rated in common and 
    # considers two users similar if they have at least topn series in common
    def getSims(self, model,topn,df,cross):
        result = dict()
        users = model.wv.vocab
        yes = 0
        bla = 0
        for user in users:
            bla += 1
            print(bla,len(users))
            rows = model.wv.similar_by_word(user,topn=len(users))
            tmp = []
            count = 0
            for row,_ in rows:
                m = df[[row,user]]
                xx = m.replace({0:np.nan})
                xx = xx.dropna()
                if len(xx)>=cross:
                    tmp.append(row)
                    count += 1
                if count >= topn:
                    break
            if len(tmp) < topn:
                yes += 1
            result[user] = tmp
        print(yes)
        return result

    # Returns for each user, the list of users that are 'similar' to it
    # cross and df are useless here
    def getSimsNormal(self, model,topn, df, cross):
      result = dict()
      users = model.wv.vocab
      for user in users:
        rows = model.wv.similar_by_word(user,topn=topn)
        tt = list(map(list, zip(*rows)))
        u = tt[0]
        p = tt[1]
        result[user] = (u,np.array(p))
      return result      

    
    # predicts the ratings of a user based on its similar users
    # input:
    #     user : which user to predict
    #     sims : users similarity dictionary
    #     df : ratings dataframe
    #     dropnan : show only intersection between predicted and known if True
    # output:
    #     dataframe containing two columns 'Predicted' and 'Original',
    #     showing in each line the two ratings we know or predicted for each serie
    def predictUser(self, user, dropnan = True, weighted = False, showTitle = False):
        sims = self.sims
        df = self.df
        titles = self.titles
        others = df[sims[user][0]]
        others = others.replace({0:np.nan})
        
        size = len(sims[user][1])
        if weighted:
            w = sims[user][1]
        else:
            w = np.ones(size)
            
        preds = []
        for row in others.values:
            indices = np.argwhere(~np.isnan(row)).ravel()
            if indices.size == 0:
                preds.append(np.NaN)
            else:
                preds.append( np.average(row[indices], weights=w[indices]))
        xx = pd.DataFrame({'Original': df[user].values}, index = df.index)        
        
        xx['Predicted'] = preds
        xx = xx.replace({0:np.nan})
        if dropnan:
            xx = xx.dropna()
        if showTitle:
            xx['Title'] = titles.loc[xx.index]
            xx = xx[['Title', 'Predicted', 'Original']]
        return xx    
      
    # MSE in a two columns dataframe
    def msePredicted(self, df):
        A = df['Predicted'].T
        B = df['Original'].T
        return mean_squared_error(A,B)

    # Calculates the MSE for given user similarities
    # output:
    #    totalMSE
    #    skipped : number of users without measurable predictions
    #    lengths : number of predictions made for each user
    def mseAllPrediction(self, weighted = False ):
        df = self.df
        sims = self.sims
        predictUser = self.predictUser
        msePredicted = self.msePredicted
        
        users = list(sims.keys())
        lengths = []
        total = 0
        skipped = 0
        for user in users:
            pred = predictUser(user, weighted = weighted)
            lengths.append(len(pred))
            if pred.empty:
                skipped +=1
            else:
                total += msePredicted(pred)
        return total/len(users), skipped, lengths    
      
      
    # modify the ratings dataframe to add this new ratings
    # for 'user', concerning 'serie', give the rating 'note', in the dataframe 'df'
    # the serie name is given as the IMDB title of the serie, the function
    # converts and finds the corresponding line in the dataframe using 'titles'
    def rate(self, user,serie,note):
        titles = self.titles
        df = self.df
        hoh = np.where( titles.values.ravel() == serie)[0][0]
        print("Before")
        print(df.loc[hoh,[user]])
        df.loc[hoh,[user]] = note
        print("After")
        print(df.loc[hoh,[user]])    
        
        
    # Recommends a list of series (ordered) to a specific user
    # input:
    #    user : which user to recommend to
    #    sims : user similarity dictionary
    #    df : ratings dataframe
    #    titles : to show clean serie titles
    #    itemSim : series similarity matrix ( prebuilt, imported and filtered beforehand )
    #    k : how many series we want to recommend
    #    simOnly : recommend only based on serie similarities ( most similar to most liked serie first,
    #              then if k most similar to best one have been watched we move to second most liked one ..)
    #    similOdds : probability of recommending by serie similarity instead of collaborative filtering
    #    coldUntil : how many ratings a user needs to make to stop considering it cold start 
    #                we recommend the best rated series in the beginning
    #    minNbRates : how many ratings required for a serie to be considered in best rated series
    # output:
    #    dataframe : line 'serie', columns 'Predicted rating' & 'serie title'
    #                series are ordered from best rated to lowest rated
    #                when recommending using similarities the prediction rating is not considered 
    #                ( so it makes the order seem false ) but its ordered by serie similarity insead
    def recommend(self, user, k, simOnly=False, similOdds=0.0, coldUntil=2, minNbRates = 1000, weighted = False):
        sims = self.sims
        df = self.df
        titles = self.titles
        itemSim = self.simMat
        predictUser = self.predictUser
        
        if df[user].count() < coldUntil:
            yuy = df[df.count(axis=1)>minNbRates]
            res = titles.loc[yuy.mean(axis=1).sort_values(ascending=False).index[:k]]
            res['Predicted'] = np.NaN
            return res[['Predicted','title']]
        xx = predictUser(user, dropnan = False, weighted = weighted)
        bestRate = xx['Original'].max()
        simAscending = False
        if bestRate < 6:
            simAscending = True
        xx['title'] = titles['title']
        hih = xx.dropna(subset=['Predicted'])
        res = hih.drop(list(hih.dropna().index))[['Predicted','title']].sort_values(by=['Predicted'],ascending=False)
        takeSims = False
        if random.uniform(0, 1)<=similOdds:
            takeSims = True
        if res.empty or simOnly or takeSims:
            print("Using Similarities")
            seen = xx.dropna(subset=['Original'])
            preferred = list(seen.sort_values(by=['Original'],ascending=simAscending).index)
            for current in range(len(itemSim)):
                s = itemSim[preferred[current]]
                if not simAscending:
                    s = -s
                proposed = (s).argsort()[1:k+1]
                b = list(proposed)
                a = set(seen.index)
                for e in a:
                    if e in b:
                        b.remove(e)
                res = xx.iloc[b][['Predicted','title']]
                if not res.empty:
                    return res
        return res        
    
    # Shows what are the ratings that this user gave originally
    def showUserOriginal(self, user):
        df = self.df
        titles = self.titles
        kiko = df[user].dropna()
        kaka = titles.loc[kiko.index]
        kaka['rate'] = kiko
        return kaka    
      
    # The real modafaka that calculates everything to recommend later
    # input:
    #     df : ratings dataframe
    #     vec_size : w2v vector size
    #     window : how many 'users' to consider as neighbours when calculating
    #     sim : how many users to consider as similar ( corresponds to topn in other functions )
    #     iters : w2v iterations
    #     min_count : minumum appearance of a user in the corpus 
    #                 ( which means that min_count should be < cold_until in recommend )
    #     clustering : build rating clusters or user sequences to build the corpus
    # output:
    #     sims : user similarity dictionary
    #     t : MSE for all the predictions based on this model
    #     s : number of users that didn't have any error measurable prediction
    #         ( set of predicted ratings INTER set of originally given ratings = EMPTY )
    #     lengths : the number of series predicted for each user
    #     poi : for each user, number of ratings and number of predictions made
    #     bed : ratio nb prediction / nb ratings for each user
    def train(self, vec_size = 500, window = 4, sim = 200, iters = 200, min_count = 1, clustering = False, weighted = False):
        df = self.df
        buildSequences = self.buildSequences
        buildClusters = self.buildClusters2
        getSims = self.getSims
        getSimsNormal = self.getSimsNormal
        mseAllPrediction = self.mseAllPrediction
        
        if clustering:
            corpus = buildClusters(df)
        else:
            seqs, _ = buildSequences(df)
            corpus = list(seqs.values())
        model = gensim.models.Word2Vec(corpus, size=vec_size, window=window, iter=iters, workers=4, min_count=min_count)
        self.w2v = model
        self.users = np.array(list(model.wv.vocab.keys()))
        sims = getSimsNormal(model,sim,df,0)
        self.sims = sims
        t,s,lengths = mseAllPrediction(weighted = weighted)
        print("MSE , Zero predictions")
        print(t,s)
        lengths = pd.Series(lengths)
        print(lengths.describe())
        counts = df.count()

        poi = pd.DataFrame(counts).T
        poi = poi[list(sims.keys())]
        lengths.index = list(sims.keys())
        poi = poi.append(lengths,ignore_index=True)
        bed = poi.iloc[1]/poi.iloc[0]
        print("Prediction percentage")
        print(bed.describe())

        self.sims = sims
        return sims, t, s, lengths, poi, bed
    
    def varySims(self, sim, weighted = False):
        df = self.df
        df = self.df
        buildSequences = self.buildSequences
        buildClusters = self.buildClusters2
        getSims = self.getSims
        getSimsNormal = self.getSimsNormal
        mseAllPrediction = self.mseAllPrediction
        model = self.w2v
        sims = getSimsNormal(model,sim,df,0)
        self.sims = sims
        t,s,lengths = mseAllPrediction(weighted = weighted)
        print("MSE , Zero predictions")
        print(t,s)
        lengths = pd.Series(lengths)
        print(lengths.describe())
        counts = df.count()

        poi = pd.DataFrame(counts).T
        poi = poi[list(sims.keys())]
        lengths.index = list(sims.keys())
        poi = poi.append(lengths,ignore_index=True)
        bed = poi.iloc[1]/poi.iloc[0]
        print("Prediction percentage")
        print(bed.describe())        
