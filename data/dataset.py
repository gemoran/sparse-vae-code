import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
import pandas as pd
import gzip
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import sys
from collections import Counter
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.stats import chi2
from data.process_data import load_peerread, load_simulated_data, load_movielens, load_movielens_small, load_peerread_small, load_zeisel

class IntervenedDataset(Dataset):
	def __init__(self, processed_data_file):
		arr = np.load(processed_data_file)
		self.factors = arr['theta_sim']
		self.tr_data = arr['sim_obs']
		self.te_data = arr['orig_obs']
		self.metadata = arr['features']
		self.num_splits = 2
		self.tr_normalized_data = None
		self.te_normalized_data = None

	def get_sigma_prior(self):
		# set up sigmas prior
		sigmas_init = self.tr_data.std(axis=0)
		sig_quant = 0.9
		sig_df = 3

		sig_est = np.quantile(sigmas_init, q=0.05)
		if sig_est==0:
			sig_est = 1e-3

		q_chi = chi2.ppf(1-sig_quant, sig_df)
		sig_scale = sig_est * sig_est * q_chi / sig_df
		return sig_scale, sigmas_init

	def normalize_columns(self):
		self.tr_normalized_data = self.tr_data / self.tr_data.sum(axis=1)[:, np.newaxis]
		self.te_normalized_data = self.te_data / self.te_data.sum(axis=1)[:, np.newaxis]
		
	def __getitem__(self, idx):
		datadict = {
			'data': torch.tensor(self.tr_data[idx, :], dtype=torch.float)
		}
		if self.tr_normalized_data is not None:
			datadict.update({'normalized_data': torch.tensor(self.tr_normalized_data[idx, :], dtype=torch.float)})

		return datadict

	def __len__(self):
		return self.tr_data.shape[0]

	def get_num_features(self):
		return self.tr_data.shape[1]



class LemmaTokenizer:
	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def __call__(self, doc):
		return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if str.isalpha(t)]

class BaseDataset(Dataset):
	def __init__(self, dataset_name, data_file=None, processed_data_file=None, is_discrete_data=False, make_data_from_scratch=False, **kwargs):
		super(Dataset, self).__init__()

		self.dataset_name = dataset_name
		self.data_file = data_file
		self.processed_data_file = processed_data_file
		self.is_discrete_data = is_discrete_data
		self.make_data_from_scratch = make_data_from_scratch

		self.parse_args(**kwargs)
		self.process_dataset()

	def parse_args(self, **kwargs):
		self.simulated_sigma_true = float(kwargs.get('sigma_true', 0.5))
		self.simulated_N = int(kwargs.get('num_samples', 1000))
		self.simulated_rho = float(kwargs.get('rho', 0))
		self.min_year = int(kwargs.get('min_year', 2010))
		self.max_year = int(kwargs.get('max_year', 2016))
		self.subsample = int(kwargs.get('subsample', 20000))
		self.text_attr_key = kwargs.get('text_attr', 'reviewText')

	def load_data_from_raw(self):
		if self.dataset_name == 'peerread':
			data, metadata = load_peerread(self.data_file)
		elif self.dataset_name == 'peerread_small':
			data, metadata = load_peerread_small(self.data_file)
		elif self.dataset_name == 'simulated':
			data, metadata = load_simulated_data(N=self.simulated_N, sigma_true=self.simulated_sigma_true, rho = self.simulated_rho)
		elif self.dataset_name == 'movielens':
			data, metadata = load_movielens(self.data_file)
		elif self.dataset_name == 'movielens_small':
			data, metadata = load_movielens_small(self.data_file)
		elif self.dataset_name == 'zeisel':
			data, metadata = load_zeisel(self.data_file)
		return data, metadata

	def load_processed_data(self):
		arrays = np.load(self.processed_data_file, allow_pickle=True)
		data = arrays['data']
		metadata = arrays['metadata']
		return data, metadata

	def get_num_features(self):
		return self.data.shape[1]

	def process_dataset(self):
		if os.path.exists(self.processed_data_file) and (not self.make_data_from_scratch):
			data, metadata = self.load_processed_data()
		else:
			data, metadata = self.load_data_from_raw()
			np.savez_compressed(self.processed_data_file, data=data, metadata=metadata)

		self.data = data
		self.metadata = metadata
		self.normalized_data = None

	def assign_splits(self, num_splits=10, seed=42):
		np.random.seed(seed)
		num_docs = self.data.shape[0]
		if num_splits > 1:
			self.splits = np.random.randint(0, high=num_splits, size=num_docs)
		else:
			self.splits=None
		self.num_splits=num_splits

	def split_data(self, fold):
		if self.num_splits > 1:
			tr_indices = np.where(self.splits != fold)[0]
			te_indices = np.where(self.splits == fold)[0]

			self.tr_data = self.data[tr_indices, :]
			self.te_data = self.data[te_indices, :]
			if self.normalized_data is not None:
				self.tr_normalized_data = self.normalized_data[tr_indices, :]
				self.te_normalized_data = self.normalized_data[te_indices, :]

			if (self.dataset_name=='simulated'):
				self.tr_metadata = self.metadata[tr_indices, :]
				self.te_metadata = self.metadata[te_indices, :]
		else:
			self.tr_data = self.data
			self.te_data = None

	def center_columns(self):
		self.data = (self.data - self.data.mean(axis=0))/self.data.std(axis=0)

	def normalize_columns(self):
		self.normalized_data = self.data / self.data.sum(axis=1)[:, np.newaxis]

	def get_sigma_prior(self):
		# set up sigmas prior
		sigmas_init = self.data.std(axis=0)
		sig_quant = 0.9
		sig_df = 3

		sig_est = np.quantile(sigmas_init, q=0.05)
		if sig_est==0:
			sig_est = 1e-3

		q_chi = chi2.ppf(1-sig_quant, sig_df)
		sig_scale = sig_est * sig_est * q_chi / sig_df
		return sig_scale, sigmas_init


	def __getitem__(self, idx):
		datadict = {
			'data': torch.tensor(self.tr_data[idx, :], dtype=torch.float)
		}

		if self.normalized_data is not None:
			datadict.update({'normalized_data': torch.tensor(self.tr_normalized_data[idx, :], dtype=torch.float)})
		return datadict

	def __len__(self):
		return self.tr_data.shape[0]