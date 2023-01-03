import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from torch import nn
from torch.nn import functional as F
from pathlib import Path
from tqdm import tqdm
from CSLS_ import eval_alignment_by_sim_mat
from matplotlib import pyplot as plt
import os
import pickle
from Param import *

DATA_PATH = Path('../data/icews_wiki')
os.listdir(DATA_PATH)
device = 'cuda:2'


# %%
def create_triples(ent_file, rel_file, triple_file, times=True):
    df_ent_1 = pd.read_csv(ent_file, delimiter='\t', header=None, names=['id', 'entity'], index_col=0)
    df_rel_1 = pd.read_csv(rel_file, delimiter='\t', header=None, names=['relation', 'id'], index_col=1)

    if times:
        #TODO 修改了这里
        # df_triples_1 = pd.read_csv(triple_file, delimiter='\t', header=None,
        #                            names=['head', 'relation', 'tail', 'time_start', 'time_end'])
        df_triples_1 = pd.read_csv(triple_file, delimiter='\t', header=None,
                                   names=['head', 'relation', 'tail', 'time_start', 'time_end'])
    else:
        df_triples_1 = pd.read_csv(triple_file, delimiter='\t', header=None, names=['head', 'relation', 'tail', 'time'])

    df_trip_ent = df_triples_1.set_index('head').join(df_ent_1, rsuffix='_head').reset_index().rename(
        columns={'index': 'head', 'entity': 'entity_head'})
    df_trip_ent = df_trip_ent.set_index('tail').join(df_ent_1).reset_index().rename(
        columns={'index': 'tail', 'entity': 'entity_tail'})
    df_trip_ent = df_trip_ent.set_index('relation').join(df_rel_1).reset_index().rename(
        columns={'index': 'relation', 'relation': 'rel'})

    entities = set(df_trip_ent['head'].unique().tolist() + df_trip_ent['tail'].unique().tolist())
    relations = set([0, *(df_trip_ent['relation'].unique() + 1)])
    times_start = set([0, *(df_trip_ent['time_start'].unique() + 1)])
    times_end = set([0, *(df_trip_ent['time_end'].unique() + 1)])
    triples = df_trip_ent[['head', 'relation', 'tail', 'time_start', 'time_end']].to_numpy().tolist()

    return entities, relations, times_start, times_end, triples


def load_alignments(align_file):
    return pd.read_csv(align_file, delimiter='\t', header=None, names=['entity_1', 'entity_2']).to_numpy().tolist()

def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    ## creating the degree matrix = D
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    ##
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T  # creating the adj matrix for message passing


# %%

entities_1, relations_1, times_1_start, times_1_end, triples_1 = create_triples(DATA_PATH / 'ent_ids_1', DATA_PATH /
                                                                                'rel_ids_1', DATA_PATH / 'triples_1')
entities_2, relations_2, times_2_start, times_2_end, triples_2 = create_triples(DATA_PATH / 'ent_ids_2', DATA_PATH /
                                                                                'rel_ids_2', DATA_PATH / 'triples_2')

train_alignments = load_alignments(DATA_PATH / 'sup_pairs')
dev_alignments = load_alignments(DATA_PATH / 'ref_pairs')

all_ent, all_rel, all_times_start, all_times_end, all_triples = entities_1.union(entities_2), relations_1.union(relations_2), \
                                                                times_1_start.union(times_2_start), times_1_end.union(times_2_end), triples_1 + triples_2

# %%

print('Train/ Val: {}/ {}\n'.format(len(train_alignments), len(dev_alignments)))


# %%

def get_matrix(all_ent, all_rel, all_times_start, all_times_end, all_triples):
    ent_size = max(all_ent) + 1
    rel_size = max(all_rel) + 1
    time_start_size = max(all_times_start) + 1
    time_end_size = max(all_times_end) + 1

    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []

    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    for i in range(ent_size):
        adj_features[i, i] = 1

    time_link = np.zeros((ent_size, time_start_size))
    # time_end_link = np.zeros((ent_size, time_end_size))

    for h, r, t, tau_s, tau_e in tqdm(all_triples):
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1
        adj_features[t, h] = 1
        radj.append([h, t, r, tau_s])
        #构建逆边
        radj.append([t, h, r + rel_size, tau_s])

        radj.append([h, t, r, tau_e])
        #构建逆边
        radj.append([t, h, r + rel_size, tau_e])

        time_link[h, tau_s] += 1
        time_link[t, tau_s] += 1
        time_link[h, tau_e] += 1
        time_link[t, tau_e] += 1

        rel_out[h, r] += 1
        rel_in[t, r] += 1

    count = -1
    s = set()
    d = {}
    r_index, t_index, r_val = [], [], []

    for h, t, r, tau in tqdm(sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5)):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            t_index.append([count, tau])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            t_index.append([count, tau])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    time_features = time_link

    time_features = normalize_adj(sp.lil_matrix(time_features))  #### using time as an averager for the node features

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    rel_features = normalize_adj(rel_features)  ### use relations as an averager for the node features
    adj_features = normalize_adj(adj_features)

    return adj_matrix, np.array(r_index), np.array(r_val), np.array(t_index), adj_features, rel_features, time_features


# %%

## r_index => [unique head and tail, relation index numbers]
## t_index => [unique head and tail, time index numbers]
## r_val ==> equal weight for the unique head and tail over all relations/timestamps

adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features = get_matrix(all_ent, all_rel, all_times_start, all_times_end,
                                                                                            all_triples)
adj_matrix = np.stack(adj_matrix.nonzero(), axis=1)  ## all indices where there is "some kind of adjancent relation"
rel_matrix, rel_val = np.stack(rel_features.nonzero(), axis=1), rel_features.data  ## Nodes x Relation array
ent_matrix, ent_val = np.stack(adj_features.nonzero(), axis=1), adj_features.data
time_matrix, time_val = np.stack(time_features.nonzero(), axis=1), time_features.data

node_size = adj_features.shape[0]
rel_size = rel_features.shape[1]
time_size = time_features.shape[1]
triple_size = len(adj_matrix)
batch_size = node_size
print('node_size {}, rel_size {}, time_size {}, triple_size {}, batch_size {}\n'.format(node_size, rel_size, time_size, triple_size, batch_size))

# %%

outputs = [np.expand_dims(i, axis=0) for i in [adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features]]

adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features = outputs

# %%

print('adj_matrix shape {}, r_index shape {}, r_val_shape {}, adj_feature shape {}, rel_feature shape {}, time_feature shape {}\n'.format(adj_matrix.shape, r_index.shape, r_val.shape, adj_features.shape, rel_features.shape, time_features.shape))


# ent_dim, rel_dim, time_dim = 64, 64, 64

class ContextualEmbedding(nn.Module):
    def __init__(self, inp_dim, out_dim=50, other_dim=None):
        super(ContextualEmbedding, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.other_dim = other_dim if other_dim is not None else inp_dim
        self.embedding = nn.Embedding(self.other_dim, self.out_dim)

    def forward(self, indices):
        x = self.embedding(torch.arange(self.other_dim).long().to(device))
        size = (self.inp_dim, self.other_dim)
        indices = torch.sparse_coo_tensor(indices, values=torch.ones_like(indices[0, :], dtype=torch.float32),
                                          size=size)
        indices = torch.sparse.softmax(indices, dim=len(indices.size()) - 1)
        return torch.sparse.mm(indices, x)


# %%

# %%
'''
Time2Vec
'''
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], 1)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Time2Vec, self).__init__()
        self.l1 = CosineActivation(1, hiddem_dim)
        self.fc1 = nn.Linear(hiddem_dim, 2)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x




def get_train_set(batch_size=batch_size):
    negative_ratio = batch_size // len(train_alignments) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_alignments, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set);
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set

class Simple_HHEA(nn.Module):
    def __init__(self, rel_size, time_span, ent_name_emb, ent_structure_emb, ent_time_emb, ent_rel_emb, emb_size=64, device='cuda'):
        super(Simple_HHEA, self).__init__()

        self.device = device
        self.emb_size = emb_size
        self.indices = self.adj.coalesce().indices()
        self.indices_size = self.indices.size(1)
        self.activation = nn.ReLU()

        self.rel_translation = nn.Linear(rel_size[0], rel_size[1])
        self.attend_time = nn.Linear(emb_size * 3, 1)

        self.dropout = nn.Dropout(0.1)
        #         self.graph_net = TNR_GraphAttention(adj, node_size, rel_size, time_size, triple_size, emb_size, depth, device)
        # self.ent_emb = self.ent_embedding(ent_sparse).to(self.device)
        # self.rel_emb = self.rel_embedding(rel_sparse).to(self.device)
        # self.time_emb = self.time_embedding(time_sparse).to(self.device)

        self.ent_name_emb = self.ent_embedding(ent_name_emb).to(self.device)
        self.ent_structure_emb = self.ent_embedding(ent_structure_emb).to(self.device)
        self.ent_time_emb = self.ent_embedding(ent_time_emb).to(self.device)
        self.ent_rel_emb = self.ent_embedding(ent_rel_emb).to(self.device)

    def forward(self, inputs, end_name_emb, ent_time_emb, ent_rel_emb):
        r_index_sq, r_val_sq, t_index_sq = inputs
        self.neighs, self.selfs = self.ent_emb[self.indices.T[:, 1]], self.ent_emb[self.indices.T[:, 0]]

        outputs = []
        features = self.activation(self.ent_emb)
        node_size = self.node_size

        outputs.append(features)

        time_encoding = self.activation(torch.sparse.mm(time_indices.float(), times_sum_n))

        return self.dropout(torch.cat([ent_encoding, time_encoding], 1))

alignment_pairs = get_train_set()

ent_name_emb = pickle.load(open(ENT_EMB_PATH, "rb"))
print("read entity embedding shape:", np.array(ent_emb).shape)

ent_structure_emb = pickle.load(open(STRUCTURE_EMB_PATH, "rb"))
print("read entity structure embedding shape:", np.array(structure_ent_emb).shape)

ent_time_emb = pickle.load(open(TIME_EMB_PATH, "rb"))
print("read entity time embedding shape:", np.array(time_ent_emb).shape)

ent_rel_emb = pickle.load(open(REL_EMB_PATH, "rb"))
print("read entity relation embedding shape:", np.array(ent_rel_emb).shape)

rel_size = (len(relations_1), len(relations_2))
time_range = 1 + 27 * 13


model = Simple_HHEA(rel_size, time_range, ent_name_emb, ent_structure_emb, ent_time_emb, ent_rel_emb, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.01)


def l1(ll, rr):
    return torch.sum(torch.abs(ll - rr), axis=-1)


gamma = 1.0
losses = []
t_prec = []
accs = []
t_mrrs = []
for i in tqdm(range(3000)):
    model.train()
    optimizer.zero_grad()
    features = model((r_index_sq, r_val_sq, t_index_sq))
    comp = features[alignment_pairs]
    l, r, fl, fr = comp[:, 0, :], comp[:, 1, :], comp[:, 2, :], comp[:, 3, :]
    loss = torch.sum(nn.ReLU()(gamma + l1(l, r) - l1(l, fr)) + nn.ReLU()(gamma + l1(l, r) - l1(fl, r))) / batch_size
    losses.append(loss.item())
    loss.backward(retain_graph=True)
    optimizer.step()
    if ((i + 1) % 10) == 0:
        model.eval()
        with torch.no_grad():
            feat = model((r_index_sq, r_val_sq, t_index_sq))[dev_alignments]
            Lvec = np.array(feat[:, 0, :].cpu())  # np.array([vec[e1] for e1, e2 in feat])
            Rvec = np.array(feat[:, 1, :].cpu())  # np.array([vec[e2] for e1, e2 in dev_alignments])
            Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
            Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
            t_prec_set, acc, t_mrr = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 1, csls=10, accurate=True)
            accs.append(acc)
            t_mrrs.append(t_mrr)
            t_prec.append(t_prec_set)


