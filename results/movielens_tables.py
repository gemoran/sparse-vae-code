import torch
import numpy as np
from model.models import SparseVAESpikeSlab, VAE, VSC
from evaluation.evaluator import Evaluator
from data.dataset import BaseDataset
import pandas as pd
import bottleneck as bn

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    DCG = (np.asarray(heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk]) * tp).sum(axis=1)
    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in np.count_nonzero(heldout_batch,axis=1)])

    out = DCG / IDCG

    return out.mean()

n_folds = 5
model_types = ["spikeslab", "vae", "vsc"] + ["beta=" + str(i) for i in [2.0, 4.0, 6.0, 8.0, 16.0]]

heldout_nll = pd.DataFrame(np.zeros((len(model_types), n_folds)))
heldout_nll.index = model_types

recall = pd.DataFrame(np.zeros((len(model_types), n_folds)))
recall.index = model_types

NDCG = pd.DataFrame(np.zeros((len(model_types), n_folds)))
NDCG.index = model_types

# Load dataset
datafile = "movielens_small"
dataset_zip = np.load('../dat/proc/' + datafile + '_proc.npz')

dataset = BaseDataset(datafile, data_file="../dat/",
                      processed_data_file='../dat/proc/' + datafile + '_proc.npz',
                      is_discrete_data=False,
                      make_data_from_scratch=False)

dataset.assign_splits(num_splits=5)

n_components=50
lambda1 = 1
lambda0 = 10

input_dim = dataset.data.shape[1]
latent_dim = n_components
batch_size = 512
hidden_dim = 300

loss_type='categorical'
z_prior='standard'


for i in range(n_folds):
    dataset.split_data(i)

    for m in model_types:
        if m == 'spikeslab':
            model = SparseVAESpikeSlab(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim, z_prior=z_prior,
                                       loss_type=loss_type, lambda0=lambda0, lambda1=lambda1, row_normalize=False)
            model.load_state_dict(
                torch.load("../out/" + datafile + "/" + m + "_" + str(i), map_location=torch.device('cpu')))

        if m == 'vae':
            model = VAE(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim, z_prior=z_prior, loss_type=loss_type,
                        lambda0=lambda0, lambda1=lambda1)
            model.load_state_dict(
                torch.load("../out/" + datafile + "/" + m + "_" + str(i), map_location=torch.device('cpu')))

        if m == 'vsc':
            model = VSC(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim, z_prior=z_prior,
                                     loss_type=loss_type, lambda0=lambda0, lambda1=lambda1)
            model.load_state_dict(
                torch.load("../out/" + datafile + "/" + m + "_" + str(i), map_location=torch.device('cpu')))


        if "beta" in m:
            beta_param = float(m.replace("beta=", ''))
            model = VAE(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim, z_prior=z_prior, loss_type=loss_type,
                        lambda0=lambda0, lambda1=lambda1, beta_vae=beta_param)
            model.load_state_dict(
                torch.load("../out/" + datafile + "/vae_lr_0.0001_" + str(i) + "_" + m, map_location=torch.device('cpu')))

        evaluator = Evaluator(model, dataset, is_discrete=False, model_name=model_name)

        heldout_nll.loc[m][i] = evaluator.evaluate_heldout_nll()
        recall.loc[m][i] = evaluator.recall_at_R(R=5)

        x_test = dataset.te_data
        model.eval()
        with torch.no_grad():
            x_mean, z, z_mean, z_log_var = model(torch.tensor(x_test, dtype=torch.float))
        x_mean = x_mean.detach().numpy()
        NDCG.loc[m][i] = NDCG_binary_at_k_batch(x_mean, x_test, k=10)



heldout_table = heldout_nll.apply(lambda x: str(round(np.mean(x), 1)) + " (" + str(round(np.std(x), 1)) + ")", axis=1)
heldout_table = pd.DataFrame(heldout_table)
heldout_table.columns=['log loss']

recall_table = recall.apply(lambda x: str(round(np.mean(x), 2)) + " (" + str(round(np.std(x), 3)) + ")", axis=1)
NDCG_table = NDCG.apply(lambda x: str(round(np.mean(x), 2)) + " (" + str(round(np.std(x), 3)) + ")", axis=1)

heldout_table.insert(1, 'recall', recall_table)
heldout_table.insert(2, 'NDCG', NDCG_table)


heldout_table.to_latex('../doc/tables/' + datafile + '_results.tex')


# get topics
#--------------------------
i = 1
m = 'spikeslab'

model = SparseVAESpikeSlab(batch_size, input_dim, latent_dim, hidden_dim=hidden_dim, z_prior=z_prior,
                           loss_type=loss_type, lambda0=lambda0, lambda1=lambda1, row_normalize=False)
model.load_state_dict(
    torch.load("../out/" + datafile + "/" + m + "_" + str(i), map_location=torch.device('cpu')))

evaluator = Evaluator(model, dataset, is_discrete=False)
evaluator.visualize_mask_as_topics(num_words=5)

W = model.W.detach().numpy()
p_star = model.p_star.detach().numpy()
titles = dataset.metadata

for k in range(n_components):
    beta = W[:, k]
    top_n = np.min((5, np.sum(beta > 0)))
    top_movies = (-beta).argsort()[:top_n]
  #  movie_titles = [(titles[t], round(beta[t],2)) for t in top_movies]
    movie_titles = [titles[t] for t in top_movies]
    print('Topic {}: {}'.format(k, ' '.join(movie_titles)))

# anchors
# check histogram for spike/slab cutoff
plt.hist(W)

W_threshold = np.ndarray.copy(W)
W_threshold[np.abs(W_threshold) < 0.01] = 0


for k in range(n_components):
    beta = np.abs(W_threshold[:, k])
    top_n = np.min((5, np.sum(beta > 0)))
    top_movies = (-beta).argsort()[:top_n]
  #  movie_titles = [(titles[t], round(beta[t],2)) for t in top_movies]
    movie_titles = [titles[t] for t in top_movies]
    print('Topic {}: {}'.format(k, ' '.join(movie_titles)))

    anchors = np.where(np.logical_and(np.sum(W_threshold != 0, axis = 1) == 1, W_threshold[:, k] != 0))[0]
    top_n_anchors = np.min((5, len(anchors)))

    anchors = anchors[-beta[anchors].argsort()][:top_n_anchors]

    topic_anchors = [titles[t] for t in anchors]
    print('Topic anchors {}: {}'.format(k, ' '.join(topic_anchors)))