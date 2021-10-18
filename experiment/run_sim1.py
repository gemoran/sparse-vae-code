from model.models import SparseVAESpikeSlab, VAE, VSC
from model.model_trainer import ModelTrainer
from evaluation.evaluator import Evaluator
import sys
import os
import numpy as np
from data.dataset import BaseDataset
import torch
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from absl import flags
from absl import app

def main(argv):
    print("Running simulation study", '..' * 20)

    make_data_from_scratch = (FLAGS.data == 'simulated')
    dataset = BaseDataset(FLAGS.data, data_file=FLAGS.datafile,
                          processed_data_file=FLAGS.procfile,
                          is_discrete_data=FLAGS.is_discrete,
                          make_data_from_scratch=make_data_from_scratch,
                          rho=FLAGS.rho)

    if FLAGS.is_discrete:
        dataset.normalize_columns()
        sigma_prior = 1.
        sigma_init = None
    else:
        sigma_prior, sigma_init = dataset.get_sigma_prior()

    dataset.assign_splits(num_splits=FLAGS.num_folds)
    dataset.split_data(FLAGS.split)

    train_params = {'batch_size': FLAGS.batch_size,
                    'shuffle': True,
                    'num_workers': 0
                    }
    training_dataloader = DataLoader(dataset, **train_params)
    input_dim = dataset.get_num_features()

    np.savetxt(FLAGS.outdir + "x.csv", dataset.te_data, delimiter=',')
    np.savetxt(FLAGS.outdir + "z.csv", dataset.te_metadata, delimiter=',')

    models = ['spikeslab', 'vae', 'vsc']
    priors = ['standard']
    beta_vae = [1, 2, 4, 6, 8, 16]

    for p in priors:
        for m in models:
            for beta in beta_vae:
                print('..' * 5, "prior:", p, ", model:", m, ", beta:", beta, '..' * 5)

                if beta==1:

                    if m=='spikeslab':
                        model = SparseVAESpikeSlab(FLAGS.batch_size, input_dim, FLAGS.n_components, hidden_dim=FLAGS.hidden_dim ,z_prior=p,sigma_prior_scale=sigma_prior,
                                                   sigmas_init=sigma_init, lambda0=FLAGS.lambda0, lambda1=FLAGS.lambda1, row_normalize=FLAGS.row_norm)

                    if m=='vae':
                        model = VAE(FLAGS.batch_size, input_dim, FLAGS.n_components, hidden_dim=FLAGS.hidden_dim,z_prior=p, sigma_prior_scale=sigma_prior,
                            sigmas_init=sigma_init)

                    if m=='vsc':
                        model = VSC(FLAGS.batch_size,
                                    input_dim,
                                    FLAGS.n_components,
                                    hidden_dim=FLAGS.hidden_dim,
                                    sigma_prior_scale=sigma_prior,
                                    sigmas_init=sigma_init)

                    model.a=1
                    model.b=input_dim

                    model_trainer = ModelTrainer(model,
                                                 m,
                                                 is_discrete_data=FLAGS.is_discrete,
                                                 save=FLAGS.save,
                                                 load=FLAGS.load,
                                                 model_file=FLAGS.model_file)

                    model_trainer.train(training_dataloader, epochs=FLAGS.epochs, weight_decay=FLAGS.weight_decay)

                    x_mean, z, z_mean, z_log_var = model(torch.tensor(dataset.te_data, dtype=torch.float))

                    np.savetxt(FLAGS.outdir + m + '_' + p + '_x_fit.csv', x_mean.detach().numpy(), delimiter=',')
                    np.savetxt(FLAGS.outdir + m + '_' + p + '_z_mean.csv', z_mean.detach().numpy(), delimiter=',')

                    evaluator = Evaluator(model, dataset, is_discrete=FLAGS.is_discrete, model_name=m)

                    if not torch.any(torch.isnan(z_mean)):
                        scores = evaluator.evaluate_dci()
                        print("DCI disentanglement:", scores['disentanglement'])
                        dci_out = np.array(scores['disentanglement'], ndmin=1)
                    else:
                        dci_out = np.array(0, ndmin=1)

                    np.savetxt(FLAGS.outdir + m + '_' + p + '_dci.csv', dci_out, delimiter=',')


                    if m == 'spikeslab':
                        np.savetxt(FLAGS.outdir + m + '_' + p +'_W.csv', model.W.detach().numpy(), delimiter=',')
                        thetas = model.thetas
                        np.savetxt(FLAGS.outdir + m + '_' + p +'_thetas.csv', thetas.numpy(), delimiter=',')
                        p_star = model.p_star
                        np.savetxt(FLAGS.outdir + m + '_' + p +'_p_star.csv', p_star.numpy(), delimiter=',')

                else:
                    if m == "vae":

                        model = VAE(FLAGS.batch_size, input_dim, FLAGS.n_components, hidden_dim=FLAGS.hidden_dim,z_prior=p, sigma_prior_scale=sigma_prior,
                            sigmas_init=sigma_init, beta_vae=beta)


                        model_trainer = ModelTrainer(model,
                                                 m,
                                                 is_discrete_data=FLAGS.is_discrete,
                                                 save=FLAGS.save,
                                                 load=FLAGS.load,
                                                 model_file=FLAGS.model_file)

                        model_trainer.train(training_dataloader, epochs=FLAGS.epochs, weight_decay=FLAGS.weight_decay)

                        x_mean, z, z_mean, z_log_var = model(torch.tensor(dataset.te_data, dtype=torch.float))

                        np.savetxt(FLAGS.outdir + "beta_" + str(beta) + '_' + m + '_' + p + '_x_fit.csv', x_mean.detach().numpy(), delimiter=',')
                        np.savetxt(FLAGS.outdir + "beta_" + str(beta) + '_' + m + '_' + p + '_z_mean.csv', z_mean.detach().numpy(), delimiter=',')

                        evaluator = Evaluator(model, dataset, is_discrete=FLAGS.is_discrete)

                        if not torch.any(torch.isnan(z_mean)):
                            scores = evaluator.evaluate_dci()
                            print("DCI disentanglement:", scores['disentanglement'])
                            dci_out = np.array(scores['disentanglement'], ndmin=1)
                        else:
                            dci_out = np.array(0, ndmin=1)

                        np.savetxt(FLAGS.outdir + "beta_" + str(beta) + '_' + m + '_' + p + '_dci.csv', dci_out, delimiter=',')




if __name__ == '__main__':
    FLAGS = flags.FLAGS

    flags.DEFINE_string("datafile", "", "path to file if using raw data files.")
    flags.DEFINE_string("procfile", "", "path to file for processed data.")
    flags.DEFINE_string("pretraining_file", "", "path to pretrained data.")
    flags.DEFINE_string("data", "simulated",
                        "name of setting to use from [simulated]")
    flags.DEFINE_string("outdir", "out/sim1/", "directory to which to write outputs.")
    flags.DEFINE_string("model_file", "svae_simulated", "file where model is saved.")

    flags.DEFINE_integer("n_components", 5, "number of latent components to use.")
    flags.DEFINE_integer("batch_size", 100, "batch size to use in training.")
    flags.DEFINE_integer("epochs", 200, "number of epochs for training.")
    flags.DEFINE_integer("hidden_dim", 50, "dimension of hidden layers.")
    flags.DEFINE_integer("split", 0, "split to run experiment.")
    flags.DEFINE_integer("num_folds", 2, "number of splits.")

    flags.DEFINE_float("lambda0", 10, "lambda0 regularization parameter for spikeslab.")
    flags.DEFINE_float("lambda1", 1, "lambda1 regularization parameter for spikeslab.")
    flags.DEFINE_float("rho", 0, "cov(factors) in simulated data.")
    flags.DEFINE_float("weight_decay", 0, "weight decay for neural network weights.")

    flags.DEFINE_boolean("row_norm", False, "flag to row normalize W or not in spike slab model.")
    flags.DEFINE_boolean("is_discrete", False, "flag that indicates if observed data is continuous or discrete.")
    flags.DEFINE_boolean("pretrained", False, "flag to use pretrained topics or not.")
    flags.DEFINE_boolean("save", False, "flag to save model.")
    flags.DEFINE_boolean("load", False, "flag to load saved model.")

    app.run(main)