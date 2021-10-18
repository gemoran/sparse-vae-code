library(ggplot2)
library(reshape2)

# color palette
bic_pal <- colorRampPalette(rev(c('#ca0020','#f4a582','#ffffff','#92c5de','#0571b0')))(25)

# function to plot matrix
plot_matrix <- function(Y, xlab = NULL, ylab = NULL, title = NULL, legend = FALSE) {
  Y <- as.matrix(Y)
  nrows <- nrow(Y)
  ncols <- ncol(Y)
  if (is.null(rownames(Y))) {
    rownames(Y) = factor(1:nrows)
  } else {
    rownames(Y) = factor(rownames(Y))
  }
  if (is.null(colnames(Y))) {
    colnames(Y) = factor(1:ncols)
  } else {
    colnames(Y) = factor(colnames(Y))
  }
  Y_melt <- as.data.frame(matrix(0, nrow = nrows * ncols, ncol = 3))
  Y_melt[, 1] <- factor(rep(rownames(Y), ncols), levels = rownames(Y))
  Y_melt[, 2] <- factor(rep(colnames(Y), each = nrows), levels = colnames(Y))
  Y_melt[, 3] <- as.vector(Y)
  colnames(Y_melt) <- c("X", "Y", "Value")
  lim <- max(abs(Y_melt$Value))
  g_Y <- ggplot(Y_melt, aes(x = Y, y = X)) + geom_tile(aes(fill = Value)) +
    scale_x_discrete(expand = c(0, 0)) + 
    scale_fill_gradientn(colors = bic_pal, limits = c(-lim, lim)) +
    scale_y_discrete(limits = rev(levels(Y_melt$X)), expand = c(0, 0)) +
    theme(title = element_text(face = "plain"), axis.ticks = element_blank()) +
    #theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))+
    theme(axis.line.x = element_blank(), axis.line.y = element_blank()) +
    labs(x = xlab, y = ylab, title = title) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank())+
    theme(panel.background = element_blank()) +
    theme(panel.border = element_rect(colour = "black", fill=NA, size=0.75))+
    theme(axis.text = element_blank())
  if(legend == FALSE) {
    g_Y <- g_Y + theme(legend.position = "none")
  }
  return(g_Y)
}



plot_decoder = function(Y, Y_est, X_est) {
  
  N = nrow(Y)
  G = ncol(Y)
  
  Y = as.matrix(Y)
  Y_est = as.matrix(Y_est)
  X_est = as.matrix(X_est)
  
  b_true = 1:7

  groups = rep(1:G, rep(N, G))
  
  dat = data.frame(y = as.vector(Y), 
                   y_pred = as.vector(Y_est),
                   x = c(rep(X_est[,1], 3), rep(X_est[, 2], 3), X_est[,1] * X_est[,2]),
                   groups = groups)
  
  b = numeric(7)
  
  d = dat[groups == 1,]
  get_line = lm(y ~ x - 1, data = d)
  b[1] = get_line$coefficients
  
  # rescale for plot so b[1] = b_true[1]
  
  dat$x[groups %in% c(1, 2, 3, 7)] <- b[1]/b_true[1] * dat$x[groups %in% c(1, 2, 3, 7)] 
  
  d = dat[groups == 1,]
  get_line = lm(y ~ x - 1, data = d)
  b[1] = round(get_line$coefficients, 1)
  
  
  d = dat[groups == 2,]
  get_line = lm(y ~ x - 1, data = d)
  b[2] = round(get_line$coefficients, 1)
  
  d = dat[groups == 3, ]
  d$x2 <- (d$x)^2
  get_line = lm(y ~ x2 - 1, data = d)
  b[3] = round(get_line$coefficients, 1)
  
  d = dat[groups == 4, ]
  get_line = lm(y ~ x - 1, data = d)
  b[4] = get_line$coefficients
  
  #dat$x[groups %in% c( 4, 5, 6, 7)] <- b[4]/b_true[4] * dat$x[groups %in% c(4, 5, 6, 7)] 
  
  d = dat[groups == 4, ]
  get_line = lm(y ~ x - 1, data = d)
  b[4] = round(get_line$coefficients, 1)
  
  d = dat[groups == 5, ]
  get_line = lm(y ~ x - 1, data = d)
  b[5] = round(get_line$coefficients, 1)
  
  d = dat[groups == 6, ]
  d$x_sin <- sin(d$x)
  get_line = lm(y ~ x_sin - 1, data = d)
  b[6] = round(get_line$coefficients, 1)
  
  d = dat[groups == 7,]
  get_line = lm(y ~ x - 1, data = d)
  b[7] = round(get_line$coefficients, 1)
  
  # equations for graph
  line_labels = character(G)
  line_labels[1] =  paste("x['.1'] == ", b[1], "%.% z[1]")
  line_labels[2] =  paste("x['.2'] == ", b[2], "%.% z[1]")
  line_labels[3] =  paste("x['.3'] == ", b[3], "%.% z[1]^2")
  line_labels[4] =  paste("x['.4'] == ", b[4], "%.% z[2]")
  line_labels[5] =  paste("x['.5'] == ", b[5], "%.% z[2]")
  line_labels[6] =  paste("x['.6'] == ", b[6], "%.% sin(z[2])")
  line_labels[7] =  paste("x['.7'] == ", b[7], "%.% z[1]%.%z[2]")
  
  
  dat$groups <- factor(groups, levels = 1:7, labels = line_labels)
  
  # add fitted lines to graph
  min_1 <- min(dat$x[groups %in% c(1, 2, 3)])
  max_1 <- max(dat$x[groups %in% c(1, 2, 3)])
  
  X_line_1 <- seq(min_1, max_1, by = 0.01)
  n_seq <- length(X_line_1)
  Y_line <- matrix(0, nrow = n_seq, ncol = G)
  
  Y_line[, 1] <- b[1] * X_line_1
  Y_line[, 2] <- b[2] * X_line_1
  Y_line[, 3] <- b[3] * X_line_1^2
  
  min_2 <- min(dat$x[groups %in% c(4, 5, 6)])
  max_2 <- max(dat$x[groups %in% c( 4, 5, 6)])
  
  X_line_2 <- seq(min_2, max_2, length.out = n_seq)
  Y_line[, 4] <- b[4] * X_line_2
  Y_line[, 5] <- b[5] * X_line_2
  Y_line[, 6] <- b[6] * sin(X_line_2)
  
  min_3 <- min(dat$x[groups == 7])
  max_3 <- max(dat$x[groups == 7])
  
  X_line_3 <- seq(min_3, max_3, length.out = n_seq)
  Y_line[, 7] <- b[7] * X_line_3
  
  dat_line <- data.frame(x = c(rep(X_line_1, 3), rep(X_line_2, 3), X_line_3),
                         y = as.vector(Y_line), 
                         groups = rep(1:G, rep(n_seq, G)))
  dat_line$groups <- factor(dat_line$groups, levels = 1:G, labels = line_labels)
  
  g1 <- ggplot(dat, aes(x = x, y = y)) +
    geom_point() +
    geom_line(data = dat_line, aes(x = x, y = y), color = "red") +
    facet_wrap(~ groups, scales = "free", labeller = label_parsed, nrow = 1) +
    theme_light() +
    labs(x = "Z (estimated)", y = "X")
    theme(strip.background =element_rect(fill= "lightgrey")) +
    theme(strip.text = element_text(colour = 'black')) +
    theme(strip.text = element_text(size= 12))
  
  
  dat$labels <- groups
  
  g2 <- ggplot(dat, aes(x = y_pred, y = y)) +
    geom_point() +
    theme_light() +
    geom_abline(intercept = 0, slope = 1, color = "red") +
    labs(x = "X (estimated)", y = "X")
  #    facet_wrap(~ labels, scales = "free", nrow = 1)
  
  list(map = g1, predict = g2)
  
}

plot_decoder_2 = function(Y, Y_est, X_est) {
  
  N = nrow(Y)
  G = ncol(Y)
  
  Y = as.matrix(Y)
  Y_est = as.matrix(Y_est)
  X_est = as.matrix(X_est)
  
  b_true = 1:7
  
  groups = rep(1:G, rep(N, G))
  
  dat = data.frame(y = as.vector(Y), 
                   y_pred = as.vector(Y_est),
                   x = c(rep(X_est[,1], 3), rep(X_est[, 2], 3), X_est[,1] * X_est[,2]),
                   groups = groups)
  
  b = numeric(7)
  
  d = dat[groups == 1,]
  get_line = lm(y ~ x - 1, data = d)
  b[1] = get_line$coefficients
  
  # rescale for plot so b[1] = b_true[1]
  
  #  dat$x[groups %in% c(1, 2, 3, 7)] <- b[1]/b_true[1] * dat$x[groups %in% c(1, 2, 3, 7)] 
  
  d = dat[groups == 1,]
  get_line = lm(y ~ x - 1, data = d)
  b[1] = round(get_line$coefficients, 1)
  
  
  d = dat[groups == 2,]
  get_line = lm(y ~ x - 1, data = d)
  b[2] = round(get_line$coefficients, 1)
  
  d = dat[groups == 3, ]
  d$x2 <- (d$x)^2
  get_line = lm(y ~ x2 - 1, data = d)
  b[3] = round(get_line$coefficients, 1)
  
  d = dat[groups == 4, ]
  get_line = lm(y ~ x - 1, data = d)
  b[4] = get_line$coefficients
  
  #dat$x[groups %in% c( 4, 5, 6, 7)] <- b[4]/b_true[4] * dat$x[groups %in% c(4, 5, 6, 7)] 
  
  d = dat[groups == 4, ]
  get_line = lm(y ~ x - 1, data = d)
  b[4] = round(get_line$coefficients, 1)
  
  d = dat[groups == 5, ]
  get_line = lm(y ~ x - 1, data = d)
  b[5] = round(get_line$coefficients, 1)
  
  d = dat[groups == 6, ]
  d$x_sin <- sin(d$x)
  get_line = lm(y ~ x_sin - 1, data = d)
  b[6] = round(get_line$coefficients, 1)
  
  d = dat[groups == 7,]
  get_line = lm(y ~ x - 1, data = d)
  b[7] = round(get_line$coefficients, 1)
  
  # equations for graph
  line_labels = character(G)
  line_labels[1] =  paste("x['.1'] == ", b[1], "%.% z[1]")
  line_labels[2] =  paste("x['.2'] == ", b[2], "%.% z[1]")
  line_labels[3] =  paste("x['.3'] == ", b[3], "%.% z[1]^2")
  line_labels[4] =  paste("x['.4'] == ", b[4], "%.% z[1]")
  line_labels[5] =  paste("x['.5'] == ", b[5], "%.% z[1]")
  line_labels[6] =  paste("x['.6'] == ", b[6], "%.% sin(z[1])")
  line_labels[7] =  paste("x['.7'] == ", b[7], "%.% z[1]%.%z[1]")
  
  
  dat$groups <- factor(groups, levels = 1:7, labels = line_labels)
  
  # add fitted lines to graph
  min_1 <- min(dat$x[groups %in% c(1, 2, 3)])
  max_1 <- max(dat$x[groups %in% c(1, 2, 3)])
  
  X_line_1 <- seq(min_1, max_1, by = 0.01)
  n_seq <- length(X_line_1)
  Y_line <- matrix(0, nrow = n_seq, ncol = G)
  
  Y_line[, 1] <- b[1] * X_line_1
  Y_line[, 2] <- b[2] * X_line_1
  Y_line[, 3] <- b[3] * X_line_1^2
  
  min_2 <- min(dat$x[groups %in% c(4, 5, 6)])
  max_2 <- max(dat$x[groups %in% c( 4, 5, 6)])
  
  X_line_2 <- seq(min_2, max_2, length.out = n_seq)
  Y_line[, 4] <- b[4] * X_line_2
  Y_line[, 5] <- b[5] * X_line_2
  Y_line[, 6] <- b[6] * sin(X_line_2)
  
  min_3 <- min(dat$x[groups == 7])
  max_3 <- max(dat$x[groups == 7])
  
  X_line_3 <- seq(min_3, max_3, length.out = n_seq)
  Y_line[, 7] <- b[7] * X_line_3
  
  dat_line <- data.frame(x = c(rep(X_line_1, 3), rep(X_line_2, 3), X_line_3),
                         y = as.vector(Y_line), 
                         groups = rep(1:G, rep(n_seq, G)))
  dat_line$groups <- factor(dat_line$groups, levels = 1:G, labels = line_labels)
  
  g1 <- ggplot(dat, aes(x = x, y = y)) +
    geom_point() +
    geom_line(data = dat_line, aes(x = x, y = y), color = "red") +
    facet_wrap(~ groups, scales = "free", labeller = label_parsed, nrow = 1) +
    theme_light() +
    labs(x = "Z (estimated)", y = "X")
  theme(strip.background =element_rect(fill= "lightgrey")) +
    theme(strip.text = element_text(colour = 'black')) +
    theme(strip.text = element_text(size= 12))
  
  
  dat$labels <- groups
  
  g2 <- ggplot(dat, aes(x = y_pred, y = y)) +
    geom_point() +
    theme_light() +
    geom_abline(intercept = 0, slope = 1, color = "red") +
    labs(x = "X (estimated)", y = "X")
  #    facet_wrap(~ labels, scales = "free", nrow = 1)
  
  list(map = g1, predict = g2)
  
}