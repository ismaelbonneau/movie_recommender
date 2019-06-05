# !/usr/bin/python
# -*- coding: utf-8 -*

# Ismael Bonneau & Issam Benamara


#Permet d'evaluer nos moèles de recommendation
#a l'aide des mesures de RI, precision/rappel/F1/reciprocal rank

import numpy as np
from scipy.stats import ttest_ind
from difflib import SequenceMatcher

def mse_mae(df, test, pred, baseline):
    mse = []
    msebaseline = []
    mae = []
    maebaseline = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if test[i,j]:
                mse.append((df.values[i,j] - pred[i,j])**2)
                mae.append(np.abs(df.values[i,j] - pred[i,j]))

                msebaseline.append((df.values[i,j] - baseline[i,j])**2)
                maebaseline.append(np.abs(df.values[i,j] - baseline[i,j]))

    print("mse: ", np.mean(mse))
    print("mae: ", np.mean(mae))
    print("mse baseline: ", np.mean(msebaseline))
    print("mae baseline: ", np.mean(maebaseline))


class EvalIRmodel:

	def __init__(self):
		self.Beta = 2.0

	def buildPertinents(self, series_df):
	    everything = series_df[["seriesname","genres"]].values

	    ms = []
	    pertinents = []
	    skippable = []
	    forbidden = ['ction','Anim','ster','tion']
	    for c , one in enumerate(everything):
	        string1 = one[1]
	        tmp = []
	        for two in everything:
	            string2 = two[1]
	            nom = two[0]
	            match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))
	            mots = string1[match.a: match.a + match.size].split("-")
	            yes = False
	            for m in mots:
	                if ((len(m)<4) or (m in forbidden)):
	                    continue
	                yes = True
	                break
	            if yes:
	                tmp.append(nom)
	        if len(tmp) == 0:
	            skippable.append(c)
	            continue
	        pertinents.append(tmp)
	    return pertinents, skippable

	def evaluate_one(self,retrieved, pertinent, k=5):
		"""
		calcule la précision, le rappel, le score F1 et le reciprocal rank
		"""

		evaluations = {"precision": 0, "recall": 0, "F1": 0, "reciprocal rank": 0}
		intersection = set(retrieved[:k]) & set(pertinent) 
		if len(retrieved) == 0:
			precision = 0
		else:
			precision = len(intersection)/len(retrieved[:k])
        
		if len(pertinent) == 0:
			rappel = 1.
		else:
			rappel = len(intersection)/len(pertinent)
        
		if rappel == 0 and precision == 0:
			F = 0.
		else:
			F = (1 + self.Beta**2)*((precision * rappel)/((self.Beta**2)*precision + rappel))

		evaluations["precision"] = precision
		evaluations["recall"] = rappel
		evaluations["F1"] = F
		#evaluations["reciprocal rank"] = 1 + min([retrieved.index(pertinent) for pertinent in intersection]) #+1 pour faire démarrer les index à 1
		return evaluations

	def evaluate(self, retrieved, pertinent, k=5):

		evaluations = {"precision": [], "recall": [], "F1": []}
		
		for i in range(len(retrieved)):
			evals = self.evaluate_one(retrieved[i], pertinent[i], k=k)

			evaluations["precision"].append(evals["precision"])
			evaluations["recall"].append(evals["recall"])
			evaluations["F1"].append(evals["F1"])
			#evaluations["reciprocal rank"].append(evals["reciprocal rank"])

		evaluations["precision"] = {"mean": np.array(evaluations["precision"]).mean(), "std": np.array(evaluations["precision"]).std()}
		evaluations["recall"] = {"mean": np.array(evaluations["recall"]).mean(), "std": np.array(evaluations["recall"]).std()}
		evaluations["F1"] = {"mean": np.array(evaluations["F1"]).mean(), "std": np.array(evaluations["F1"]).std()}
		#evaluations["reciprocal rank"] = {"mean": np.array(valuations["reciprocal rank"]).mean(), "std": np.array(valuations["reciprocal rank"]).std()}
		return evaluations
		