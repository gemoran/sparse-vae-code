source('utils/utils.R')
library(ggpubr)
library(Rtsne)
library(dplyr)

models = c("spikeslab", 'vae', 'vsc')
model_names = c('Sparse VAE', "VAE", "VSC")

dir = '../out/zeisel/'

gene_info = as.matrix(read.csv("../out/zeisel/gene_info.csv", header = F))
cell_info = read.table(file = "../dat/zeisel/meta_data_mRNA.txt", stringsAsFactors = F)

H = as.matrix(read.csv(paste(dir, "H.csv", sep = ''), header = F))
H = t(H)

A_train = as.matrix(read.csv(paste(dir, "A_train.csv", sep = ''), header = F))

tsne_out = Rtsne(A_train)
x_tsne <- as.data.frame(tsne_out$Y)
names(x_tsne) = c("tSNE1", "tSNE2")
x_tsne$Cluster <- as.factor(cell_info$level1class)

g <- ggplot(x_tsne, aes(x = tSNE1, y = tSNE2, color = Cluster)) +
  geom_point(size = 1.25) + 
  scale_color_brewer(palette = "Paired") +
  theme_light() +
  labs(title = "NMF")

pdf('../doc/fig/zeisel/nmf_tsne.pdf', width = 4, height = 2.25)
print(g)
dev.off()

# x_mean = as.matrix(read.csv(paste(model_dir, "_x_mean.csv", sep = ''), header = F))
# x = as.matrix(read.csv(paste(dir, "x.csv", sep = ''), header = F))

for (m in 1:length(models)) {
  model_dir = paste(dir, models[m], sep = '')
  z_mean = as.matrix(read.csv(paste(model_dir, "_z_mean.csv", sep = ''), header = F))
  
  if (models[m] =='spikeslab') {
    W = as.matrix(read.csv(paste(model_dir, "_W.csv", sep = ''), header = F))
    p_star = as.matrix(read.csv(paste(model_dir, "_p_star.csv", sep = ''), header = F))
    thetas = as.matrix(read.csv(paste(model_dir, "_thetas.csv", sep = ''), header = F))
  } else {
    W = matrix(1, nrow = nrow(H), ncol = ncol(H))
  }
  
  sigmas = as.matrix(read.csv(paste(model_dir, "_sigmas.csv", sep = ''), header = F))
  
  # tsne
  z_mean_dat = data.frame(z_mean)
  z_mean_dat$index = 1:nrow(z_mean)
  z_mean_unique = z_mean_dat %>% distinct(across(contains("V")), .keep_all = T)
  z_mean_plot = as.matrix(z_mean_unique[, 1:ncol(z_mean)])
  
  tsne_out = Rtsne(as.matrix(z_mean_unique[, 1:ncol(z_mean)]))
  x_tsne <- as.data.frame(tsne_out$Y)
  names(x_tsne) = c("tSNE1", "tSNE2")
  cell_info =  cell_info[z_mean_unique$index, ]
  x_tsne$Cluster <- as.factor(cell_info$level1class)
  
  g <- ggplot(x_tsne, aes(x = tSNE1, y = tSNE2, color = Cluster)) +
    geom_point(size = 1.25) + 
    scale_color_brewer(palette = "Paired") +
    theme_light() +
    labs(title=model_names[m]) +
    theme(legend.position = "none")
  
  pdf(paste('../fig/zeisel/', models[m], '_tsne.pdf', sep = ''), width = 2.4, height = 2.4)
  print(g)
  dev.off()
}





