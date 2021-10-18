from sklearn.decomposition import NMF, PCA
import numpy as np
import pandas as pd
from data.dataset import BaseDataset
from evaluation.evaluator import Evaluator
from importlib import reload
import itertools as it
import torch
from model.models import SparseVAESpikeSlab, VAE, VSC
import scipy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from model.model_trainer import ModelTrainer
from evaluation.evaluator import Evaluator
from scipy.stats import chi2
import torch.optim as optim


dataset='zeisel'
datafile="../dat/"
out_dir ="../out/"
proc_file = "../dat/proc/" + dataset + "_proc.npz"
dat = BaseDataset(dataset, datafile, proc_file)

dat.assign_splits(num_splits=1)
dat.split_data(1)

n_components=15

x = dat.tr_data
gene_info = dat.metadata
np.savetxt(out_dir + dataset + "/gene_info.csv", gene_info, delimiter=",", fmt='%s')


#  NMF  --------------------------------------

nmf_model = NMF(n_components=n_components)
A_train = nmf_model.fit_transform(x)
H = nmf_model.components_

nmf_loss = np.power(x - np.matmul(A_train, H), 2).mean()

np.savetxt(out_dir + dataset + "/H.csv", H, delimiter=",")
np.savetxt(out_dir + dataset + "/A_train.csv", A_train, delimiter=",")

#-------------------------------------------------------
model_type = 'spikeslab'

input_dim = x.shape[1]
latent_dim = n_components
batch_size = 512
nepoch = 100

hidden_dim=100

lambda1 = 1
lambda0 = 10
a=1
b=input_dim

lr=1e-2

x_np = x
sigmas_init = np.std(x_np, axis=0)
sig_quant = 0.9
sig_df = 3

sig_est = np.quantile(sigmas_init, q=0.05)
if sig_est==0:
    sig_est = 1e-3

q_chi = chi2.ppf(1-sig_quant, sig_df)

sig_scale = sig_est * sig_est * q_chi / sig_df

z_prior='standard'


if model_type == 'spikeslab':
    model = SparseVAESpikeSlab(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim,z_prior=z_prior,sigma_prior_scale=sig_scale,
                                        loss_type='mse', sigmas_init=sigmas_init, lambda0=lambda0, lambda1=lambda1, row_normalize=False)
if model_type == 'vae':
    model = VAE(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim,z_prior=z_prior,sigma_prior_scale=sig_scale,
                                        loss_type='mse', sigmas_init=sigmas_init)
if model_type == 'vsc':
    model = VSC(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim,z_prior=z_prior,sigma_prior_scale=sig_scale,
                                        loss_type='mse', sigmas_init=sigmas_init)


optimizer = optim.Adam(model.parameters(), lr=lr)

dataloader = torch.utils.data.DataLoader(torch.tensor(x, dtype=torch.float), batch_size=batch_size,
                                         shuffle=True)

l = None

for epoch in range(nepoch):
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        rec_loss, kl_loss, reg_loss, sig_loss = model.vae_loss(data)
        loss = rec_loss + kl_loss + sig_loss + reg_loss
        loss.backward()
        optimizer.step()

        if model_type == 'spikeslab':
            p_star = model.p_star.detach()
            thetas = model.thetas.detach()

            for k in range(p_star.shape[1]):
                p_star[:, k] = thetas[k] * torch.exp(- lambda1 * model.W[:, k].abs()) /\
                               (thetas[k] * torch.exp(- lambda1 * model.W[:, k].abs()) + (1-thetas[k]) * torch.exp(-lambda0 * model.W[:, k].abs()))

                thetas[k] = (p_star[:, k].sum() + a - 1) / (a + b + input_dim - 2)


    if epoch % 10 == 0:
        print("Epoch:", epoch, "Total loss:", f"{loss.detach().item():.3}", "Likelihood:", f"{rec_loss.detach().item():.3}",
                "KL loss:", f"{kl_loss.detach().item():.3}", "sig loss:", f"{sig_loss.detach().item():.3}", "W loss:", f"{reg_loss.detach().item():.3}")


torch.save(model.state_dict(), "../out/zeisel/" + model_type)

z_mean, logvar = model.encode(torch.tensor(x, dtype=torch.float))

np.savetxt(out_dir + dataset + "/" + model_type + "_z_mean.csv", z_mean.detach().numpy(), delimiter=",")
np.savetxt(out_dir + dataset + "/" + model_type + "_sigmas.csv", model.log_sigmas.detach().exp().numpy(), delimiter=",")

if model_type=='spikeslab':
    np.savetxt(out_dir + dataset + "/" + model_type + "_W.csv", model.W.detach().numpy(), delimiter=",")
    np.savetxt(out_dir + dataset + "/" + model_type + "_p_star.csv", model.p_star.detach().numpy(), delimiter=",")
    np.savetxt(out_dir + dataset + "/" + model_type + "_thetas.csv", model.thetas.detach().numpy(), delimiter=",")
