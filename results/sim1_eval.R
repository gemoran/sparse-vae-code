setwd("~/Dropbox/postdoc/research/sparse-vae/src")
source('utils/utils.R')
library(ggpubr)
library(dplyr)
library(stringr)

rep = 25
rhos = c(0, 0.2, 0.4, 0.6, 0.8)
models = c("spikeslab", "vae", 'vsc')
#models = c(models, paste("beta_", c(2, 4, 6, 8, 16), "_vae", sep = ""))
priors = c("standard")

lambda0 = 10
lambda1 = 1

dir = "../out/sim1/"

W_true = matrix(0, nrow = 7, ncol = 5)
W_true[c(1:3), 1] = 1
W_true[c(4:6), 2] = 1
W_true[7, c(1,2)] = 0.5

W_mse = array(NA, dim = c(length(models), length(priors), length(rhos), rep),
              dimnames = list(models, priors, rhos, 1:rep))

W_hamming = array(NA, dim = c(length(models), length(priors), length(rhos), rep),
                  dimnames = list(models, priors, rhos, 1:rep))

X_mse = array(NA, dim = c(length(models), length(priors), length(rhos), rep),
              dimnames = list(models, priors, rhos, 1:rep))

Z_sd = array(0, dim = c(length(models), length(priors), length(rhos), rep),
             dimnames = list(models, priors, rhos, 1:rep))

Z_mse = array(NA, dim = c(length(models), length(priors), length(rhos), rep),
              dimnames = list(models, priors, rhos, 1:rep))

dci = array(NA, dim = c(length(models), length(priors), length(rhos), rep),
            dimnames = list(models, priors, rhos, 1:rep))

for (r in 1:length(rhos)) {
  for (m in models) {
    for (p in priors) {
      for (i in 1:rep) {
        dir_rho = paste(dir, "rho_", r-1, "/rep_", i, "/", sep = '')
        
        if (file.exists(paste(dir_rho, "z.csv", sep = ''))) {
          Z_true = as.matrix(read.csv(paste(dir_rho, "z.csv", sep = ''), header = F))
        }
        if (file.exists(paste(dir_rho, "x.csv", sep = ''))) {
          X = as.matrix(read.csv(paste(dir_rho, "x.csv", sep = ''), header = F))
        }
        
        file_path = paste(dir_rho, m, "_", p, "_", sep = '')
        
        if (file.exists(paste(file_path, "z_mean.csv", sep = ''))) {
          Z = as.matrix(read.csv(paste(file_path, "z_mean.csv", sep = ''), header = F))
        }  else {
          print(paste(file_path, "z_mean.csv", " doesn't exist", sep = ''))
        }
        
        if (file.exists(paste(file_path, "x_fit.csv", sep = ''))) {
          X_fit = as.matrix(read.csv(paste(file_path, "x_fit.csv", sep = ''), header = F))
        }
        
        if (file.exists(paste(file_path, "dci.csv", sep = ''))) {
          dci[m, p, r, i] = as.matrix(read.csv(paste(file_path, "dci.csv", sep = ''), header = F))
        }
        
        X_mse[m, p, r, i] = mean((X-X_fit)^2)
        
        if (!(any(is.nan(Z)))) {
          test_sd = apply(Z, 2, sd)
          
          if (all(test_sd!=0)) {
            
            inds = c(which.max(abs(cor(Z_true[,1], Z))), which.max(abs(cor(Z_true[,2], Z))))
            new_inds = union(inds, 1:5)
            
            Z = Z[, new_inds]
            if (cor(Z_true[,1], Z[,1]) < 0) {
              Z[,1]=-Z[,1]
            }
            if (cor(Z_true[,2], Z[,2]) < 0) {
              Z[,2]=-Z[,2]
            }
            
            Z = t(t(Z) / apply(Z, 2, sd))
            
            Z_mse[m, p, r, i] = mean((Z - cbind(Z_true, matrix(0, nrow = nrow(Z), ncol = 3)))^2)
            
            if (m %in% c("spikeslab", "softmax")) {
              if (file.exists(paste(file_path, "W.csv", sep = ''))) {
                W = as.matrix(read.csv(paste(file_path, "W.csv", sep = ''), header = F))
                W[abs(W) < 1e-2] = 0
              }
              
              if (m == 'spikeslab') {
                if (file.exists(paste(file_path, "p_star.csv", sep = ''))) {
                  p_star = as.matrix(read.csv(paste(file_path, "p_star.csv", sep = ''), header = F))
                  W[p_star < 0.5] = 0
                }
              }
              
              W = abs(W) / apply(abs(W), 1, sum)
              
              W = W[, new_inds]
              
              W_mse[m, p, r, i] = mean((W-W_true)^2)
              
              W_supp = W
              W_true_supp = W_true
              W_supp[W != 0] = 1
              W_true_supp[W_true != 0] = 1
              
              W_hamming[m, p, r, i] = sum(W_supp != W_true_supp)
              
            }
          }
        }
        
      
        
      }
    }
  }
}


x_melt = melt(X_mse)
colnames(x_melt) = c("Method", "prior", "correlation", "rep", "X_MSE")

ylim1 = boxplot.stats(filter(x_melt, Method %in% c('spikeslab', 'vae'),
                             prior!='vampprior', 
                             correlation==0)$X_MSE)$stats[c(1, 5)]

# equations for graph
correlation_labels = character(5)
correlation_labels[1] =  expression(paste(rho, "= 0"))
correlation_labels[2] =  expression(paste(rho, "= 0.2"))
correlation_labels[3] =  expression(paste(rho, "= 0.4"))
correlation_labels[4] =  expression(paste(rho, "= 0.6"))
correlation_labels[5] =  expression(paste(rho, "= 0.8"))

x_melt$correlation = factor(x_melt$correlation, labels=correlation_labels)

x_melt$Method=as.character(x_melt$Method)
x_melt$Method[x_melt$Method=='spikeslab'] = "SparseVAE"
x_melt$Method[x_melt$Method=='vae'] = "VAE"
x_melt$Method[x_melt$Method=='vsc'] = "VSC"


pdf("../doc/fig/sim1/mse_plot.pdf", width = 4.8, height = 1.4)

ggplot(filter(x_melt, Method %in% c('SparseVAE', 'VAE', 'VSC'),prior!='vampprior'), aes(x=Method, y=X_MSE, fill = Method)) + 
  geom_boxplot() +
  facet_wrap(~ correlation, nrow=1, scale='free', labeller = label_parsed) +
  ylab("MSE") +
  xlab("Model") +
  coord_cartesian(ylim = ylim1*1.05) + 
  theme(legend.position = "none") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank()) 

dev.off()

dci_melt = melt(dci)
colnames(dci_melt) = c("Method", "prior", "correlation", "rep", "DCI")
dci_melt$correlation = factor(dci_melt$correlation, labels=correlation_labels)

dci_melt$Method=as.character(dci_melt$Method)
dci_melt$Method[dci_melt$Method=='spikeslab'] = "SparseVAE"
dci_melt$Method[dci_melt$Method=='vae'] = "VAE"
dci_melt$Method[dci_melt$Method=='vsc'] = "VSC"


pdf("../doc/fig/sim1/dci_plot.pdf", width = 6.2, height = 1.4)

ggplot(filter(dci_melt, Method %in% c('SparseVAE', 'VAE', 'VSC'),prior!='vampprior'), aes(x=Method, y=DCI, fill = Method)) + 
  geom_boxplot(outlier.shape=NA) +
  facet_wrap(~ correlation, nrow=1, scale='free', labeller = label_parsed) +
  ylab("DCI") +
  xlab("Model") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank())

dev.off()
