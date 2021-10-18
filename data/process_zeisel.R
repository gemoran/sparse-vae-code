# R script to process Zeisel et al (2015) data

library(preprocessCore)

# Read in data
mRNA <- read.table("https://storage.googleapis.com/linnarsson-lab-www-blobs/blobs/cortex/expression_mRNA_17-Aug-2014.txt",
                   header = F, sep = "\t", fill = T, comment.char = "")

mRNA <- t(mRNA)

# Extract meta data and save as separate .txt file
meta_data <- mRNA[-c(1,2), c(1:10)]
colnames(meta_data) <- mRNA[2,1:10]
colnames(meta_data)[2] <- "group"
write.table(meta_data, file = "meta_data_mRNA.txt")

# Extract expression data and save as Y_all.txt
Y_all <- mRNA[-c(1,2),-c(1:11)]
colnames(Y_all) = mRNA[1,-c(1:11)]
rownames(Y_all) = mRNA[-c(1,2), 8]
class(Y_all) = "numeric"
write.table(Y_all, file = "Y_all.txt")

#### now we remove genes as per page 5 of supplementary material of Zeisel et al

# remove genes that have less than 25 molecules in total over all cells
remove1 <- which(apply(Y_all, 2, sum) < 25)
Y1 <- Y_all[,-remove1]

# remove genes which have 5 or fewer other genes which correlate more than rho = 0.2091
Y_corr <- cor(Y1)
rho <- 0.2091    # from supplementary material
remove2 <- which(apply(Y_corr, 2, function(x) sum(x > rho)) < 6)
Y2 <- Y1[,-remove2]


### find 5000 most variable genes using ceftools (https://github.com/linnarsson-lab/ceftools)
## first, convert file to .cef

if (F) {
  tY2 = t(Y2)
  write(file="Y_all.cef", paste("CEF\t0\t1\t1", nrow(tY2), ncol(tY2), "0", sep="\t"))
  write(file="Y_all.cef",paste( c("\tCellID"), paste(colnames(tY2), collapse="\t"), sep='\t'), append=T)
  x = cbind(rep("",nrow(tY2)), tY2)
  write(file="Y_all.cef", paste(c("Gene"), sep="\t"), append=T)
  write.table(file="Y_all.cef", x, append=T, col.names=F, row.names=T, quote=F, sep="\t")
}

## use ceftools to fit noise model: log2(CV) = log2(mean^alpha) + k
if (F) {
  system("< Y_all.cef cef aggregate --noise std | cef sort --by Noise --reverse | cef select --range 1:5000 > Y_top5000.cef")
}
## result: log2(CV) = log2(mean^-0.56) + 0.61

Y_out  <-  read.delim("Y_top5000.cef", sep = "\t",stringsAsFactors = F, skip = 1, header = F)
Y  <-  Y_out[-c(1,2), -c(1, 2, 3)]
Y <- as.matrix(Y)
Y <- apply(Y, 2, as.numeric)

## save unnormalized gene expression
write.table(t(Y), "Y_raw.txt")

Y_raw = Y

# normalize quantiles (average quantile normalization)
Y <- normalize.quantiles(Y_raw)
Y <- t(Y)

write.table(Y, file = "Y_quantile.txt", row.names = F, col.names = F)

# gene and cell info
genes <- Y_out[-c(1,2), 1]
cells <- Y_out[1,-c(1,2,3)]

write.table(genes, file = "gene_info.txt", row.names = F, col.names = F)
write.table(cells, file = "cell_info.txt", row.names = F, col.names = F)


