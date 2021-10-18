import os
import json
import numpy as np
import pandas as pd
import gzip
import argparse
from collections import Counter
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import multivariate_normal
import tensorflow as tf
import numpy.random as npr

class LemmaTokenizer:
	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if str.isalpha(t)]


def load_simulated_data(N=1000, sigma_true=0.5, rho=0):
	input_dim = 7
	latent_dim = 2
	cov = np.eye(latent_dim)
	for k in range(latent_dim):
		for j in range(k):
			cov[k, j] = rho
			cov[j, k] = rho

	z = multivariate_normal.rvs(mean=np.zeros(latent_dim), cov=cov, size=N)
	x = np.zeros((N, input_dim))

	x[:, 0] = 1 * z[:, 0]
	x[:, 1] = 2 * z[:, 0]
	x[:, 2] = 3 * (z[:, 0] ** 2)
	x[:, 3] = 4 * z[:, 1]
	x[:, 4] = 5 * z[:, 1]
	x[:, 5] = 6 * np.sin(z[:, 1])
	x[:, 6] = 7 * z[:, 0] * z[:, 1]

	x = x + (sigma_true) * np.random.randn(N, input_dim)
	return x, z

def process_text(docs, max_features=None):
	stop = stopwords.words('english')
	vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1, 1), stop_words=stop, max_df=0.9,
								 min_df=0.0007, max_features=max_features)
	counts = vectorizer.fit_transform(docs).toarray()
	term_total =counts.sum(axis=1)
	valid = (term_total > 1)
	counts = counts[valid, :]
	vocab = vectorizer.get_feature_names()
	vocab = np.array(vocab)
	return counts, vocab


def load_peerread(data_file):
	df = pd.read_csv(data_file)
	docs = df['abstract_text'].values
	matrix, features = process_text(docs)
	return matrix, features

def load_peerread_small(data_file):
	df = pd.read_csv(data_file)
	docs = df['abstract_text'].values
	matrix, features = process_text(docs,max_features=500)
	return matrix, features


def load_movielens(data_file, n_movies=3000, n_users=20000, seed=42):
	np.random.seed(seed)

	ratings = pd.read_csv(data_file + 'movielens/ratings.csv')
	movies = pd.read_csv(data_file + 'movielens/movies.csv')

	ratings = ratings[ratings.rating >= 4]
	ratings.rating = 1

	# keep only users that have rated more than 20 movies
	ratings = ratings[ratings.groupby('userId')['userId'].transform('size') >= 20]

	# keep only top n_movies rated movies
	keep = ratings.movieId.value_counts().nlargest(n_movies).index
	ratings = ratings[ratings.movieId.isin(keep)]

	ratings_mat = ratings.pivot(index='userId', columns='movieId', values='rating')
	ind = np.random.choice(ratings_mat.index, n_users, replace=False)
	ratings_mat = ratings_mat.loc[ind]

	ratings_mat = ratings_mat.fillna(0)

	movieIds = ratings_mat.columns

	movies = movies.set_index('movieId')
	titles = movies.loc[movieIds]

	matrix = ratings_mat.to_numpy()
	titles = titles['title'].to_numpy()

	return matrix, titles


def load_movielens_small(data_file, n_movies=300, n_users=100000, seed=42):
	np.random.seed(seed)

	ratings = pd.read_csv(data_file + 'movielens/ratings.csv')
	movies = pd.read_csv(data_file + 'movielens/movies.csv')

	ratings = ratings[ratings.rating >= 4]
	ratings.rating = 1

	# keep only users that have rated more than 20 movies
	ratings = ratings[ratings.groupby('userId')['userId'].transform('size') >= 20]

	# keep only top n_movies rated movies
	keep = ratings.movieId.value_counts().nlargest(n_movies).index
	ratings = ratings[ratings.movieId.isin(keep)]

	ratings_mat = ratings.pivot(index='userId', columns='movieId', values='rating')
	ind = np.random.choice(ratings_mat.index, n_users, replace=False)
	ratings_mat = ratings_mat.loc[ind]

	ratings_mat = ratings_mat.fillna(0)

	movieIds = ratings_mat.columns

	movies = movies.set_index('movieId')
	titles = movies.loc[movieIds]

	matrix = ratings_mat.to_numpy()
	titles = titles['title'].to_numpy()

	return matrix, titles


def load_zeisel(data_file):

	x = pd.read_csv(data_file + 'zeisel/Y_quantile.txt', delimiter=" ", header = None)
	gene_info = pd.read_csv(data_file + 'zeisel/gene_info.txt', delimiter=" ", header = None)

	x = x.to_numpy()

	# get top 558 most variable genes (same as scVI, Lopez et al 2018)
	x_std = np.std(x, axis=0)
	ind = (-x_std).argsort()[:558]
	x = x[:, ind]

	gene_info = gene_info.to_numpy()
	gene_info = gene_info[ind, :]

	return x, gene_info