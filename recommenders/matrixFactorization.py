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


class NMF:

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

		#moyenne item
		baseline = tf.constant(np.tile(np.array(df.mean(axis=0)),(1423,1)))
		shape = df.shape

		#constante: la matrice R à reconstituer entièrement
		R = tf.constant(df.values / 10) #divisée par 10 pour obtenir des notes entre 0 et 1

		#variable tensorflow masque
		mask_tf_train = tf.Variable(train)
		mask_tf_test = tf.Variable(test)

		#variables tensorflow
		#U et I initialisés selon une loi normale et normalisés en divisant par k
		U = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(shape[0], self.k)).astype(np.float64)), name="U")
		I = tf.Variable(np.abs(np.random.normal(scale=1./self.k, size=(self.k, shape[1])).astype(np.float64)), name="I")

		R_pred = tf.matmul(U, I) #embeddings

		#beta: paramètre de regularization
		beta = tf.constant(reg, dtype=tf.float64, name="beta")
		regularizer = beta * (tf.reduce_sum(tf.square(U)) + tf.reduce_sum(tf.square(I)))

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
		learning_rate = tf.train.exponential_decay(alpha, global_step, nbite, 0.98, staircase=True)

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost, global_step=global_step)

		costs = []
		mses_train = []
		mses_test = []

		sess = tf.Session()
		sess.run(tf.initialize_all_variables())
		for i in range(nbite):
		    sess.run(optimizer)
		    sess.run(clip)
		    if i%1000==0:
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
		baseline = tf.constant(np.tile(np.array(df.mean(axis=0)),(1423,1)) / 10.)

		shape = df.shape

		#constante: la matrice R à reconstituer entièrement
		R = tf.constant(df.values / 10) #divisée par 10 pour obtenir des notes entre 0 et 1

		#biais global 
		b = tf.Variable(df.mean().mean() / 10., dtype=tf.float32, trainable=False)

		#biais user
		b_U = tf.Variable((df.mean().mean() - df.mean(axis=1).values.reshape(-1,1)) / 10., dtype=tf.float32, trainable=False, name="bU")

		#biais item
		b_I = tf.Variable((df.mean().mean() - df.mean(axis=0).values.reshape(-1,1).T) / 10., dtype=tf.float32, trainable=False, name="bI")

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

		global_step = tf.Variable(0, trainable=False)
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
		    sess.run(clip)
		    if i%1000==0:
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