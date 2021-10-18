from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge
import sys
from scipy.special import expit, softmax
import itertools as it
from scipy.sparse import csr_matrix
import pandas as pd
import bottleneck as bn
import scipy
from six.moves import range
from sklearn import ensemble
from evaluation.dci import compute_dci

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Evaluator():
	def __init__(self, model, dataset, is_discrete=False, model_name=None):
		self.model = model
		self.dataset = dataset
		self.is_discrete = is_discrete
		self.model_name = model_name

		self.cache_model_params()
		# self.produce_theta()

	def cache_model_params(self):
		self.model.eval()
		with torch.no_grad():
			self.W = self.model.get_generator_mask().cpu().detach().numpy()

	def visualize_mask_as_topics(self, num_words=10, format_pretty=False):
		print("---"*60)
		n_components = self.W.shape[1]
		vocab = self.dataset.metadata
		for k in range(n_components):
			w = self.W[:,k]
			w = np.abs(w)
			top_words = (-w).argsort()[:num_words]

			if not format_pretty:
				topic_words = [(vocab[t], w[t]) for t in top_words]
				print('Topic {}: {}'.format(k, topic_words))
			else:
				topic_words = [vocab[t] for t in top_words]
				# topics_list.append(topic_words)
				print('Topic {}: {}'.format(k, ' '.join(topic_words)))
		print("---"*60)

	def visualize_topics(self, num_words=10, format_pretty=False):
		vocab = self.dataset.metadata
		n_components = self.W.shape[1]
		identity_matrix = np.identity(n_components)
		self.model.eval()
		with torch.no_grad():
			for k in range(n_components):
				z_k = np.tile(identity_matrix[k,:], (100, 1))
				latent_factor = torch.tensor(z_k, dtype=torch.float, device=device)
				x_predicted = self.model.decode(latent_factor).cpu().detach().numpy()
				x_predicted = x_predicted.mean(axis=0)
				top_words = (-x_predicted).argsort()[:num_words]

				if not format_pretty:
					topic_words = [(vocab[t], x_predicted[t]) for t in top_words]
					print('Topic {}: {}'.format(k, topic_words))
				else:
					topic_words = [vocab[t] for t in top_words]
					# topics_list.append(topic_words)
					print('Topic {}: {}'.format(k, ' '.join(topic_words)))
		print("---"*60)

	def evaluate_heldout_nll(self):
		if self.dataset.num_splits > 1:
			if self.is_discrete:
				test_features = self.dataset.te_normalized_data
			else:
				test_features = self.dataset.te_data
			te_input = torch.tensor(test_features, dtype=torch.float).to(device)
			te_target = torch.tensor(self.dataset.te_data, dtype=torch.float).to(device)

			if self.model_name == 'vsc':
				self.model.eval()
				with torch.no_grad():
					x_mean, z, z_mean, z_log_var, z_log_gamma = self.model(te_input)
					x_loss = self.model.reconstruction_loss(x_mean, te_target).cpu().detach().numpy()

			else:
				self.model.eval()
				with torch.no_grad():
					x_mean, z, z_mean, z_log_var = self.model(te_input)
					x_loss = self.model.reconstruction_loss(x_mean, te_target).cpu().detach().numpy()

		else:
			if self.is_discrete:
				train_features = self.dataset.tr_normalized_data
			else:
				train_features = self.dataset.tr_data
			tr_input = torch.tensor(train_features, dtype=torch.float).to(device)
			tr_target = torch.tensor(self.dataset.tr_data, dtype=torch.float).to(device)

			if self.model_name == 'vsc':
				self.model.eval()
				with torch.no_grad():
					x_mean, z, z_mean, z_log_var, z_log_gamma = self.model(tr_input)
					x_loss = self.model.reconstruction_loss(x_mean, tr_target).cpu().detach().numpy()

			else:
				self.model.eval()
				with torch.no_grad():
					x_mean, z, z_mean, z_log_var = self.model(tr_input)
					x_loss = self.model.reconstruction_loss(x_mean, tr_target).cpu().detach().numpy()

			print("No test data -- using training data...")

		return x_loss

	def recall_at_R(self, R=10):

		if self.dataset.num_splits > 1:
			if self.is_discrete:
				test_features = self.dataset.te_normalized_data
			else:
				test_features = self.dataset.te_data
			te_input = torch.tensor(test_features, dtype=torch.float).to(device)
		else:
			if self.is_discrete:
				test_features = self.dataset.tr_normalized_data
			else:
				test_features = self.dataset.tr_data
			te_input = torch.tensor(test_features, dtype=torch.float).to(device)
			print("No test data -- using training data...")

		self.model.eval()
		with torch.no_grad():
			if self.model_name == 'vsc':
				x_mean, z, z_mean, z_log_var, z_log_gamma = self.model(te_input)
			else:
				x_mean, z, z_mean, z_log_var = self.model(te_input)


		x_mean = x_mean.cpu().detach().numpy()

		N = x_mean.shape[0]

		idx = bn.argpartition(-x_mean, R, axis=1)
		x_mean_binary = np.zeros_like(x_mean, dtype=bool)
		x_mean_binary[np.arange(N)[:, np.newaxis], idx[:, :R]] = True

		x_true_binary = np.asarray(test_features > 0)
		tmp = (np.logical_and(x_true_binary, x_mean_binary).sum(axis=1)).astype(np.float32)
		recall = tmp / np.minimum(R, x_true_binary.sum(axis=1))

		recall = np.mean(recall)

		return recall

	def evaluate_dci(self, seed=42,
						batch_size=16):

		"""Computes the DCI scores

        Args:
          data: class BaseDataset
          true_factors:  true latent factors that generated data
          model: Trained model that gets representations
          random_seed: Numpy random seed used for randomness.
          batch_size: Batch size for sampling.

        Returns:
          Dictionary with average disentanglement score, completeness and
            informativeness (train and test).
        """

		np.random.seed(seed)

		if self.dataset.num_splits>1:
			train_features = self.dataset.tr_data
			test_features = self.dataset.te_data
			train_factors = self.dataset.tr_metadata
			test_factors = self.dataset.te_metadata
		else:
			train_features = self.dataset.tr_data
			test_features = self.dataset.tr_data
			train_factors = self.dataset.tr_metadata
			test_factors = self.dataset.tr_metadata

		num_train = train_features.shape[0]
		num_test = test_features.shape[0]

		batch_ind_tr = np.random.choice(range(num_train), batch_size, replace=False)
		batch_ind_te = np.random.choice(range(num_test), batch_size, replace=False)

		batch_train_features = train_features[batch_ind_tr, :]
		batch_test_features = test_features[batch_ind_te, :]

		batch_train_factors = train_factors[batch_ind_tr, :]
		batch_test_factors = test_factors[batch_ind_te, :]

		# codes are estimated factors (in the terminology of DCI)
		if self.model_name == 'vsc':
			self.model.eval()
			with torch.no_grad():
				train_codes, _, _ = self.model.encode(torch.tensor(batch_train_features, dtype=torch.float).to(device))
				test_codes, _, _ = self.model.encode(torch.tensor(batch_test_features, dtype=torch.float).to(device))

		else:
			self.model.eval()
			with torch.no_grad():
				train_codes, _ = self.model.encode(torch.tensor(batch_train_features, dtype=torch.float).to(device))
				test_codes, _ = self.model.encode(torch.tensor(batch_test_features, dtype=torch.float).to(device))


		train_codes = np.transpose(train_codes.detach().cpu().numpy())
		test_codes = np.transpose(test_codes.detach().cpu().numpy())

		train_factors = np.transpose(batch_train_factors)
		test_factors = np.transpose(batch_test_factors)

		scores = compute_dci(train_codes, train_factors, test_codes, test_factors)
		return scores


