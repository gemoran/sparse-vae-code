from model.models import SparseVAESpikeSlab, VAE, VSC
from model.model_trainer import ModelTrainer
from evaluation.evaluator import Evaluator
import sys
import os
import numpy as np
from data.dataset import BaseDataset
import torch
from torch.utils.data import DataLoader
from absl import flags
from absl import app

TEXT_SETTINGS = {'peerread', 'peerread_small'}

def main(argv):
	print("Running model", FLAGS.model, '..' * 20)
	print(FLAGS.data)

	if FLAGS.pretraining_file == "":
		pretraining_file = '../dat/proc/' + FLAGS.data + '_pretraining.npz'
	else:
		pretraining_file = FLAGS.pretraining_file

	if FLAGS.procfile == "":
		os.makedirs('../dat/proc/', exist_ok=True)
		proc_file = '../dat/proc/' + FLAGS.data + '_proc.npz'
	else:
		proc_file = FLAGS.procfile

	if FLAGS.pretrained:
		array = np.load(pretraining_file)
		beta_pretrained = array['beta'].T
	else:
		beta_pretrained = None

	make_data_from_scratch = (FLAGS.data == 'simulated')
	dataset = BaseDataset(FLAGS.data, data_file=FLAGS.datafile, 
							processed_data_file=proc_file, 
							is_discrete_data=FLAGS.is_discrete, 
							make_data_from_scratch=make_data_from_scratch)

	if FLAGS.is_discrete:
		dataset.normalize_columns()
		sigma_prior = 1.
		sigma_init = None
	else:
		sigma_prior, sigma_init = dataset.get_sigma_prior()

	dataset.assign_splits(num_splits=FLAGS.num_folds)
	dataset.split_data(fold=FLAGS.split)

	train_params = {'batch_size': FLAGS.batch_size,
					'shuffle': True,
					'num_workers': 0
					}
	training_dataloader = DataLoader(dataset, **train_params)
	input_dim = dataset.get_num_features()

	#### Setting the a and b prior parameters to be either the passed args or some function of input dims and n_components
	a = 1.0
	b = input_dim

	if FLAGS.model == 'spikeslab':

		model = SparseVAESpikeSlab(FLAGS.batch_size,
			input_dim,
			FLAGS.n_components,
			hidden_dim=FLAGS.hidden_dim,
			loss_type=FLAGS.loss_type,
			sigma_prior_scale=sigma_prior,
			sigmas_init=sigma_init,
			lambda0=FLAGS.lambda0,
			lambda1=FLAGS.lambda1,
			W_init=beta_pretrained,
			a=a,
			b=b,
			row_normalize=FLAGS.row_norm)


	elif FLAGS.model == 'vae':
		model = VAE(FLAGS.batch_size,
			input_dim,
			FLAGS.n_components,
			hidden_dim=FLAGS.hidden_dim,
			loss_type=FLAGS.loss_type,
			sigma_prior_scale=sigma_prior,
			sigmas_init=sigma_init,
			beta_vae=FLAGS.beta_vae)

	elif FLAGS.model == 'vsc':
		model = VSC(FLAGS.batch_size,
			input_dim,
			FLAGS.n_components,
			hidden_dim=FLAGS.hidden_dim,
			loss_type=FLAGS.loss_type,
			sigma_prior_scale=sigma_prior,
			sigmas_init=sigma_init)


	model_trainer = ModelTrainer(model,
		FLAGS.model,
		is_discrete_data=FLAGS.is_discrete,
		save=FLAGS.save, 
		load=FLAGS.load, 
		model_file=FLAGS.model_file)

	model_trainer.train(training_dataloader, epochs=FLAGS.epochs, lr=FLAGS.lr)

	evaluator = Evaluator(model, dataset, is_discrete=FLAGS.is_discrete, model_name=FLAGS.model)

	heldout_nll = evaluator.evaluate_heldout_nll()
	print("Heldout negative log likelihood:", heldout_nll)

	os.makedirs(FLAGS.outdir, exist_ok=True)
	outfile = os.path.join(FLAGS.outdir, FLAGS.model + '.' + str(FLAGS.split) + '.beta=' + str(FLAGS.beta_vae) + '.row_norm=' + str(FLAGS.row_norm) +  '.l0_l1=' + str((FLAGS.lambda0,FLAGS.lambda1)) + '.dim=' + str(FLAGS.hidden_dim))
	np.save(outfile, np.array([heldout_nll]))

	if FLAGS.data == 'movielens' or FLAGS.data == 'movielens_small':
		heldout_recall = evaluator.recall_at_R()
		print("Heldout recall:", heldout_recall)
		evaluator.visualize_mask_as_topics()

	if FLAGS.data in TEXT_SETTINGS:
		evaluator.visualize_mask_as_topics()

	
if __name__ == '__main__':
	FLAGS = flags.FLAGS
	flags.DEFINE_string('model', 'spikeslab', "variant of VAE model to fit from [spikeslab, vae, vsc]")
	flags.DEFINE_string("datafile", "", "path to file if using raw data files.")
	flags.DEFINE_string("procfile", "", "path to file for processed data.")
	flags.DEFINE_string("pretraining_file", "", "path to pretrained data.")
	flags.DEFINE_string("data", "simulated", "name of setting to use from [simulated, peerread_small, movielens_small, zeisel]")
	flags.DEFINE_string("outdir", "../out/", "directory to which to write outputs.")
	flags.DEFINE_string("model_file", "svae_simulated", "file where model is saved.")
	flags.DEFINE_string("loss_type", "mse", "loss to use from [mse, binary, categorical]")

	flags.DEFINE_integer("n_components", 5, "number of latent components to use.")
	flags.DEFINE_integer("batch_size", 100, "batch size to use in training.")
	flags.DEFINE_integer("epochs", 100, "number of epochs for training.")
	flags.DEFINE_integer("split", 0, "split to run experiment.")
	flags.DEFINE_integer("num_folds", 2, "number of splits.")
	flags.DEFINE_integer("hidden_dim", 300, "dimension of hidden layers.")

	flags.DEFINE_float("lambda0", 10., "lambda0 regularization parameter for spikeslab.")
	flags.DEFINE_float("lambda1", 1., "lambda1 regularization parameter for spikeslab.")
	flags.DEFINE_float("beta_vae", 1, "beta-vae parameter.")
	flags.DEFINE_float("lr", 0.01, "learning rate for Adam")

	flags.DEFINE_boolean("is_discrete", False, "flag that indicates if observed data is continuous or discrete.")
	flags.DEFINE_boolean("pretrained", False, "flag to use pretrained topics or not.")
	flags.DEFINE_boolean("row_norm", False, "flag to row normalize W or not in spike slab model.")
	flags.DEFINE_boolean("save", False, "flag to save model.")
	flags.DEFINE_boolean("load", False, "flag to load saved model.")

	
	app.run(main)