# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:50:09 2019

@author:

  _____                               _ 
  \_   \ ___  _ __ ___    __ _   ___ | |
   / /\// __|| '_ ` _ \  / _` | / _ \| |
/\/ /_  \__ \| | | | | || (_| ||  __/| |
\____/  |___/|_| |_| |_| \__,_| \___||_|
                                        

"""

import time
import numpy as np
import tensorflow as tf
from scipy.sparse import dok_matrix, csr_matrix
from sklearn.decomposition import FastICA, PCA


class NMF:

	def __init__(self, k, init="random", solver="adam"):

		#number of latent factors
		self.k = k
		#methode d'initialisation
		if solver not in ["sgd", "adam", "rmsprop"]:
			raise ValueError(str(solver) + " is not a correct value for parameter solver. Valid ones are 'sgd', 'adam', 'rmsprop'")
		if init not in ["random", "pca", "ica"]:
			raise ValueError(str(init) + " is not a correct value for parameter init. Valid ones are 'random', 'pca', 'ica'")
		self.init = init
		self.solver = solver

	def run(self, df, train, test, nbite, reg, alpha=0.001, verbose=True):
		"""
		data: pandas dataframe
		nbite: nombre d'iterations
		reg: regularization parameter
		alpha: learning rate
		"""

		#moyenne item
		baseline = tf.constant(np.tile(np.array(df.mean(axis=0)),(df.shape[1],1)))
		shape = df.shape

		#constante: la matrice R à reconstituer entièrement
		R = tf.constant(df.values)
		#variable tensorflow masque
		mask_tf_train = tf.Variable(train)
		mask_tf_test = tf.Variable(test)

		#variables tensorflow

		if self.init == "random":
			#U et I initialisés selon une loi normale et normalisés en divisant par k
			U = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(shape[0], self.k)).astype(np.float64)), name="U")
			I = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(self.k, shape[1])).astype(np.float64)), name="I")
		if self.init == "ica":
			matrix = dok_matrix(df.shape, dtype=np.float64)

			for i in range(df.shape[0]):
			    for j in range(df.shape[1]):
			        if not np.isnan(df.values[i,j]):
			            matrix[i, j] = df.values[i,j]

			ica = FastICA(n_components=self.k)

			U = tf.Variable(np.abs(ica.fit_transform(matrix.toarray())), name="U")
			I = tf.Variable(np.abs(ica.components_), name="I")

		if self.init == "pca":
			matrix = dok_matrix(df.shape, dtype=np.float64)

			for i in range(df.shape[0]):
			    for j in range(df.shape[1]):
			        if not np.isnan(df.values[i,j]):
			            matrix[i, j] = df.values[i,j]

			pca = PCA(n_components=self.k)

			U = tf.Variable(np.abs(pca.fit_transform(matrix.toarray())), name="U")
			I = tf.Variable(np.abs(pca.components_), name="I")			


		R_pred = tf.matmul(U, I) #embeddings

		#beta: paramètre de regularization
		beta = tf.constant(reg, dtype=tf.float64, name="beta")
		#regularization L1
		regularizer = beta * (tf.reduce_sum(U) + tf.reduce_sum(I))

		#cout de l'algo NMF, norme matricielle de R - R_pred
		cost = tf.reduce_sum(tf.square(tf.boolean_mask(R, mask_tf_train) - tf.boolean_mask(R_pred, mask_tf_train)))
		cost += regularizer

		#contraintes de non-négativité de U et I
		clip_U = U.assign(tf.maximum(tf.zeros_like(U), U))
		clip_I = I.assign(tf.maximum(tf.zeros_like(I), I))
		clip = tf.group(clip_U, clip_I)

		#erreur MSE train
		mse_train = tf.reduce_mean(tf.square(tf.boolean_mask(R_pred, mask_tf_train) - tf.boolean_mask(R, mask_tf_train)), name="mse_train")
		mse_test = tf.reduce_mean(tf.square(tf.boolean_mask(R_pred, mask_tf_test) - tf.boolean_mask(R, mask_tf_test)), name="mse_test")

		#baseline MSE test
		baselineMSE = tf.reduce_mean(tf.square(tf.boolean_mask(baseline, mask_tf_test) - tf.boolean_mask(R, mask_tf_test)))

		global_step = tf.Variable(0, trainable=False)

		if self.solver == "adam":
			optimizer = tf.train.AdamOptimizer(alpha).minimize(cost, global_step=global_step)
		elif self.solver == "sgd":
			optimizer = tf.train.GradientDescentOptimizer(alpha).minimize(cost, global_step=global_step)
		else:
			#optimiseur rmsprop
			optimizer = tf.train.RMSPropOptimizer(alpha).minimize(cost, global_step=global_step)

		costs = []
		mses_train = []
		mses_test = []

		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		for i in range(nbite):
		    sess.run(optimizer)
		    sess.run(clip)
		    if i%100==0:
		        prout = sess.run(cost)
		        lol = sess.run(mse_train)
		        mdr = sess.run(mse_test)
		        if verbose:
			        print("cost: %f" % prout)
			        print("mse train: %f" % lol)
			        print("mse test: %f" % mdr)
			        print("***************")
		        costs.append((i, prout))
		        mses_train.append((i, lol))
		        mses_test.append((i, mdr))
		        
		            
		learnt_U = sess.run(U)
		learnt_I = sess.run(I)
		msebaseline = sess.run(baselineMSE)
		if verbose:
			print("baseline: ", msebaseline)
		sess.close()

		return learnt_U, learnt_I, {"mse_train": mses_train, "mse_test": mses_test, "cost": costs}


class SVDpp:

	def __init__(self, k):

		#number of latent factors
		self.k = k


	def run(self, df, train, test, nbite, reg, alpha=0.001, verbose=True):
		"""
		data: pandas dataframe
		nbite: nombre d'iterations
		reg: regularization parameter
		alpha: learning rate
		"""

		#moyenne item: baseline
		baseline = tf.constant(np.tile(np.array(df.mean(axis=0)),(df.shape[1],1)), dtype=tf.float32)

		shape = df.shape

		#constante: la matrice R à reconstituer entièrement
		R = tf.constant(df.values, dtype=tf.float32)

		#biais global 
		b = tf.Variable(df.mean().mean(), dtype=tf.float32, trainable=False)

		#biais user
		b_U = tf.Variable((df.mean().mean() - df.mean(axis=1).values.reshape(-1,1)), dtype=tf.float32, trainable=False, name="bU")

		#biais item
		b_I = tf.Variable((df.mean().mean() - df.mean(axis=0).values.reshape(-1,1).T), dtype=tf.float32, trainable=False, name="bI")

		#variable tensorflow masque
		mask_tf_train = tf.Variable(train)
		mask_tf_test = tf.Variable(test)

		#variables tensorflow
		#U et I initialisés selon une loi normale et normalisés en divisant par k
		U = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(shape[0], self.k)).astype(np.float32)), name="U")
		I = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(self.k, shape[1])).astype(np.float32)), name="I")

		#mean + item and user deviation from mean + embeddings
		R_pred = b + b_U + b_I + tf.matmul(U, I)

		#beta: paramètre de regularization
		beta = tf.constant(reg, dtype=tf.float32, name="beta")
		regularizer = beta * (tf.reduce_sum(tf.square(U)) + tf.reduce_sum(tf.square(I)))

		#cout de l'algo NMF, norme matricielle de R - R_pred
		cost = tf.reduce_sum(tf.square(tf.boolean_mask(R, mask_tf_train) - tf.boolean_mask(R_pred, mask_tf_train)))
		cost += regularizer

		#erreur MSE train
		mse_train = tf.reduce_mean(tf.square(tf.boolean_mask(R_pred, mask_tf_train) - tf.boolean_mask(R, mask_tf_train)), name="mse_train")
		mse_test = tf.reduce_mean(tf.square(tf.boolean_mask(R_pred, mask_tf_test) - tf.boolean_mask(R, mask_tf_test)), name="mse_test")

		#baseline MSE test
		baselineMSE = tf.reduce_mean(tf.square(tf.boolean_mask(baseline, mask_tf_test) - tf.boolean_mask(R, mask_tf_test)))

		global_step = tf.Variable(0, dtype=tf.float32, trainable=False)
		#learning rate decay
		learning_rate = tf.train.exponential_decay(alpha, global_step, nbite, 0.98, staircase=True)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

		costs = []
		mses_train = []
		mses_test = []

		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		for i in range(nbite):
		    sess.run(optimizer)
		    if i%100==0:
		        cst = sess.run(cost)
		        msetrain = sess.run(mse_train)
		        msetest = sess.run(mse_test)
		        if verbose:
			        print("cost: %f" % cst)
			        print("mse train: %f" % msetrain)
			        print("mse test: %f" % msetest)
			        print("***************")
		        costs.append((i, cst))
		        mses_train.append((i, msetrain))
		        mses_test.append((i, msetest))
		        
		            
		learnt_U = sess.run(U)
		learnt_I = sess.run(I)
		final_b_U = sess.run(b_U)
		final_b_I = sess.run(b_U)
		final_b = sess.run(b)

		msebaseline = sess.run(baselineMSE)
		if verbose:
			print("baseline: ", msebaseline)
		sess.close()

		return learnt_U, learnt_I, final_b_U, final_b_I, final_b, {"mse_train": mses_train, "mse_test": mses_test, "cost": costs}


class triNMF:

	def __init__(self, k):

		#number of latent factors
		self.k = k

	def run(self, df, sim, train, test, steps, reg, alpha=0.001, verbose=True)
		#moyenne item
		baseline = tf.constant(np.tile(np.array(df.mean(axis=0)),(df.shape[1],1)))

		shape = df.shape

		#constante: la matrice R à reconstituer entièrement
		R = tf.constant(df.values) #divisée par 10 pour obtenir des notes entre 0 et 1
		#constante: la matrice S de similarités (sans valeurs manquantes) à approximer par QxI
		S = tf.constant(sim, dtype=tf.float64)

		#variable tensorflow masque
		mask_tf_train = tf.Variable(train)
		mask_tf_test = tf.Variable(test)

		#variables tensorflow
		#U et I initialisés selon une loi normale d'écart type 1/k
		U = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(shape[0], self.k)).astype(np.float64)), name="U")
		I = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(self.k, shape[1])).astype(np.float64)), name="I")
		Q = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(shape[1], self.k)).astype(np.float64)), name="Q", trainable=False)

		R_pred = tf.matmul(U, I) #embeddings

		#beta: paramètre de regularization
		beta = tf.constant(reg, dtype=tf.float64, name="beta")
		regularizer = beta * (tf.square(tf.reduce_sum(S - tf.matmul(Q, I))))

		#cout de l'algo NMF, norme matricielle de R - R_pred
		cost = tf.reduce_sum(tf.square(tf.boolean_mask(R, mask_tf_train) - tf.boolean_mask(R_pred, mask_tf_train)))
		cost += regularizer

		#contraintes de non-négativité de U et I
		clip_U = U.assign(tf.maximum(tf.zeros_like(U), U))
		clip_I = I.assign(tf.maximum(tf.zeros_like(I), I))
		clip = tf.group(clip_U, clip_I)

		#erreur MSE train
		mse_train = tf.reduce_mean(tf.square(tf.boolean_mask(R_pred, mask_tf_train) - tf.boolean_mask(R, mask_tf_train)), name="mse_train")
		mse_test = tf.reduce_mean(tf.square(tf.boolean_mask(R_pred, mask_tf_test) - tf.boolean_mask(R, mask_tf_test)), name="mse_test")
		#mse similarités
		mse_sim = tf.reduce_mean(tf.square(S - tf.matmul(Q, I)))

		#baseline MSE test
		baselineMSE = tf.reduce_mean(tf.square(tf.boolean_mask(baseline, mask_tf_test) - tf.boolean_mask(R, mask_tf_test)))

		global_step = tf.Variable(0, trainable=False)

		optimizer = tf.train.AdamOptimizer(alpha).minimize(cost, global_step=global_step)

		costs = []
		mses_train = []
		mses_test = []
		mses_sim = []

		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		for i in range(steps):
		    sess.run(optimizer)
		    sess.run(clip)
		    if i%100==0:
		        cst = sess.run(cost)
		        if verbose:
		        	print(cst)
		        msetrain = sess.run(mse_train)
		        msetest = sess.run(mse_test)
		        msesim = sess.run(mse_sim)
		        costs.append((i, cst))
		        mses_train.append((i, msetrain))
		        mses_test.append((i, msetest))
		        mses_sim.append((i, msesim))
		            
		learnt_U = sess.run(U)
		learnt_I = sess.run(I)
		msebaseline = sess.run(baselineMSE)
		sess.close()

		if verbose:
			print("baseline: ", msebaseline)

	return learnt_U, learnt_I, final_b_U, final_b_I, final_b, {"mse_train": mses_train, "mse_test": mses_test, "cost": costs}

