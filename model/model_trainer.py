from torch import nn, optim
from torch.nn import functional as F
import torch
import numpy as np
from sklearn import metrics
from sklearn.linear_model import Ridge
import sys
from scipy.special import expit
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer():
    def __init__(self, model, model_name, is_discrete_data=False, save=False, load=False, model_file=None, **kwargs):

        self.model = model
        self.model_name = model_name
        self.save = save
        self.load = load
        self.model_file = model_file
        self.is_discrete_data = is_discrete_data

    def train(self, training_loader, epochs=10, lr=1e-2, weight_decay=0):
        # weight decay 1.2e-6
        self.model.to(device)

        if self.load:
            self.model.load_state_dict(torch.load(self.model_file, map_location=device))
        else:
            self.train_vae_model(training_loader, epochs=epochs, lr=lr, weight_decay=weight_decay)

        if self.save:
            os.makedirs(os.path.dirname(self.model_file), exist_ok=True)

            # if self.model_name == 'spikeslab':
            #     self.model_file += '_lambda0=' + str(self.model.lambda0) + "_lambda1=" + str(self.model.lambda1)
            if self.model_name == 'vae' and self.model.beta_vae != 1:
                self.model_file = self.model_file + '_beta=' + str(self.model.beta_vae)
            torch.save(self.model.state_dict(), self.model_file)




    def train_vae_model(self, training_loader, epochs=10, lr=0.01, weight_decay=1.2e-6):

        if self.model_name == 'vae':
            params_with_l2 = list(self.model.generator.parameters()) + \
                         list(self.model.q_z.parameters()) + list(self.model.z_mean.parameters()) + list(self.model.z_log_var.parameters())
            params_no_l2 = list([self.model.log_sigmas])

        elif self.model_name == 'vsc':
            params_with_l2 = list(self.model.generator.parameters()) + \
                         list(self.model.q_z.parameters()) + list(self.model.z_mean.parameters()) + list(self.model.z_log_var.parameters()) + \
                list(self.model.z_log_gamma.parameters())
            params_no_l2 = list([self.model.log_sigmas])

        else:
            params_with_l2 = list(self.model.generator.parameters()) + list(self.model.column_means.parameters()) + \
                         list(self.model.q_z.parameters()) + list(self.model.z_mean.parameters()) + list(self.model.z_log_var.parameters())
            params_no_l2 = list([self.model.W]) + list([self.model.log_sigmas])


        optimizer_l2 = optim.Adam(params_with_l2, lr=lr, weight_decay=weight_decay)
        optimizer_no_l2 = optim.Adam(params_no_l2, lr=lr)


        for epoch in range(epochs):
            self.model.train()
            for _, data in enumerate(training_loader, 0):

                optimizer_l2.zero_grad()
                optimizer_no_l2.zero_grad()

                features = data['data'].to(device, dtype=torch.float)
                if self.is_discrete_data:
                    normalized_features = data['normalized_data'].to(device, dtype=torch.float)
                    likelihood_loss, kl_loss, w_loss, regularizer_loss = self.model.vae_loss(features, normalized_x=normalized_features)
                else:
                    likelihood_loss, kl_loss, w_loss, regularizer_loss = self.model.vae_loss(features)

                if self.model_name=='vae':
                    kl_loss = self.model.beta_vae * kl_loss

                total_loss = likelihood_loss + kl_loss + regularizer_loss + w_loss

                total_loss.backward()
                optimizer_l2.step()
                optimizer_no_l2.step()

                if (self.model_name == 'spikeslab'):
                    p_star=self.model.p_star.detach()
                    thetas=self.model.thetas.detach()
                    for k in range(p_star.shape[1]):
                        p_star[:, k] = thetas[k] * torch.exp(- self.model.lambda1 * self.model.W[:, k].abs()) / \
                                       (thetas[k] * torch.exp(- self.model.lambda1 * self.model.W[:, k].abs()) + (
                                                   1 - thetas[k]) * torch.exp(-self.model.lambda0 * self.model.W[:, k].abs()))

                        thetas[k] = (p_star[:, k].sum() + self.model.a - 1) / (self.model.a + self.model.b + self.model.input_dim - 2)
                        if thetas[k] < 0:
                            thetas[k] = 1e-10

                l_total = total_loss.item()
                l_like = likelihood_loss.item()
                l_kl = kl_loss.item()
                l_reg = regularizer_loss.item()
                l_w = w_loss.item()


            if epoch % 2 == 0:
                print("Epoch:", epoch, "Total loss:", f"{l_total:.3}", "Likelihood:", f"{l_like:.3}",
                  "KL loss:", f"{l_kl:.3}", "Sigma loss:", f"{l_reg:.3}", "W loss:", f"{l_w:.3}")
                sys.stdout.flush()


