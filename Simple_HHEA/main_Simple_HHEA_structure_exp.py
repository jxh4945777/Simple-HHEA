import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import argparse
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from tqdm import tqdm
from CSLS_ import eval_alignment_by_sim_mat
from matplotlib import pyplot as plt
import os
import pickle
from Param import *
from utils import load_aligned_pair, load_triples

parser = argparse.ArgumentParser(description='Simple_HHEA Experiment')
parser.add_argument('--if_structure',  type=bool, default=False)
parser.add_argument('--lang',  type=str, default='icews_wiki')
parser.add_argument('--add_noise',  type=bool, default=True)
parser.add_argument('--train_ratio',  type=float, default=0.3)
parser.add_argument('--noise_ratio',  type=float, default=0)
parser.add_argument('--cuda_num',  type=int, default=3)
# parser.add_argument('--facts_drop',  type=float, default=0.2)

args = parser.parse_args()
device = 'cuda:'+str(args.cuda_num)
print('start exp: noise_ratio: {}, if_structure: {}, lang: {}\n'.format(args.noise_ratio, args.if_structure, args.lang))

DATA_PATH = r"../data/{}/".format(args.lang)
FILE_PATH = "../data/{}/".format(args.lang) #interaction model save path.
INTERACTION_MODEL_SAVE_PATH = "../Save_model/interaction_model_{}.bin".format(args.lang) #interaction model save path.
BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "{}".format(args.lang)


file_path = DATA_PATH
DATA_PATH = Path(DATA_PATH)
os.listdir(DATA_PATH)
rel_size = (272, 226)
time_range = 1 + 27 * 13


all_triples, node_size, rel_size = load_triples(file_path, True)
# train_pair, test_pair = load_aligned_pair(file_path, ratio=0.3)
# pair_num = len(train_pair)

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
#TODO training ratio改这里
train_alignments = load_alignments(DATA_PATH / 'sup_pairs')
dev_alignments = load_alignments(DATA_PATH / 'ref_pairs')
ent_pair_size = len(train_alignments) + len(dev_alignments)


# %%

print('Train/ Val: {}/ {}\n'.format(len(train_alignments), len(dev_alignments)))



## r_index => [unique head and tail, relation index numbers]
## t_index => [unique head and tail, time index numbers]
## r_val ==> equal weight for the unique head and tail over all relations/timestamps

#TODO batch size

batch_size = node_size
# batch_size = BATCH_SIZE

print('node_size {}, rel_size {}, batch_size {}\n'.format(node_size, rel_size, batch_size))

# %%

# outputs = [np.expand_dims(i, axis=0) for i in [adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features]]
#
# adj_matrix, r_index, r_val, t_index, adj_features, rel_features, time_features = outputs

# %%


# ent_dim, rel_dim, time_dim = 64, 64, 64
def rel_time_cal(time_year, time_month):
    return (time_year - 1995) * 13 + time_month + 1

def load_ent_time_matrix(file_path):
    ent_1_num = 0
    ent_2_num = 0
    ent_1_list = []
    ent_2_list = []
    ent_pairs = []
    for line in open(file_path + 'ref_ent_ids', 'r+', encoding='utf-8'):
        line = line.replace('\n','').split('\t')
        ent_pairs.append((line[0], line[1]))

    for line in open(file_path + 'ent_ids_1', 'r+', encoding='utf-8'):
        line = line.replace('\n','').split('\t')
        ent_1_list.append(line[0])
        if line != '':
            ent_1_num += 1

    for line in open(file_path + 'ent_ids_2', 'r+', encoding='utf-8'):
        line = line.replace('\n','').split('\t')
        ent_2_list.append(line[0])
        if line != '':
            ent_2_num += 1

    time_dict = dict()
    old_time_id_list = []
    time_y = 0
    for line in open(file_path + 'time_id', 'r+', encoding='utf-8'):
        if line != '':
            line = line.replace('\n','').split('\t')
            time_dict[line[0]] = line[1]
            if line[1] == '' or line[1] == '-400000':
                line[1] = '~'
            if line[1] == '~':
                continue
            try:
                time_y = int(float(line[1].split('-')[0]))
            except Exception as e:
                print(e)

            if time_y < 1995:
                old_time_id_list.append(line[0])

    #这里代表年份
    #2 (old) + [(2021-1995) + 1] * 13 (这里包括00月)
    ent_1_emb = np.zeros([ent_1_num, 1 + 27 * 13])
    ent_2_emb = np.zeros([ent_2_num, 1 + 27 * 13])

    for line in tqdm(open(file_path + 'triples_1', 'r+', encoding='utf-8')):
        line = line.replace('\n','').split('\t')
        if time_dict[line[3]] != '~':
            time_y = int(float(time_dict[line[3]].split('-')[0]))
            time_m = int(float(time_dict[line[3]].split('-')[1]))
            if time_y < 1995:
                ent_1_emb[int(float(line[0])), 0] += 1
            else:
                ent_1_emb[int(float(line[0])), rel_time_cal(time_y, time_m)] += 1

        if time_dict[line[4]] != '~':
            time_y = int(float(time_dict[line[4]].split('-')[0]))
            time_m = int(float(time_dict[line[4]].split('-')[1]))
            if time_y < 1995:
                ent_1_emb[int(float(line[0])), 0] += 1
            else:
                ent_1_emb[int(float(line[0])), rel_time_cal(time_y, time_m)] += 1

    for line in tqdm(open(file_path + 'triples_2', 'r+', encoding='utf-8')):
        line = line.replace('\n', '').split('\t')
        time_y_s = 0
        time_m_s = 0
        time_y_e = 0
        time_m_e = 0
        if time_dict[line[3]] != '~' and time_dict[line[3]] != '-400000':
            time_y_s = int(float(time_dict[line[3]].split('-')[0]))
            time_m_s = int(float(time_dict[line[3]].split('-')[1]))
            if time_y_s < 1995:
                ent_2_emb[int(float(line[0])) - ent_1_num, 0] += 1
                time_y_s = 1995
                time_m_s = 0

        if time_dict[line[4]] != '~' and time_dict[line[3]] != '-400000':
            time_y_e = int(float(time_dict[line[4]].split('-')[0]))
            time_m_e = int(float(time_dict[line[4]].split('-')[1]))
            # ent_2_emb[int(float(line[0])) - ent_1_num, 0] += 1
            if time_y_e >= 1995:
                ent_2_emb[int(float(line[0])) - ent_1_num, rel_time_cal(time_y_s, time_m_s): rel_time_cal(time_y_e, time_m_e)] += 1
    return np.array(ent_1_emb.tolist() + ent_2_emb.tolist())


# class ContextualEmbedding(nn.Module):
#     def __init__(self, inp_dim, out_dim=50, other_dim=None):
#         super(ContextualEmbedding, self).__init__()
#         self.inp_dim = inp_dim
#         self.out_dim = out_dim
#         self.other_dim = other_dim if other_dim is not None else inp_dim
#         self.embedding = nn.Embedding(self.other_dim, self.out_dim)
#
#     def forward(self, indices):
#         x = self.embedding(torch.arange(self.other_dim).long().to(device))
#         size = (self.inp_dim, self.other_dim)
#         indices = torch.sparse_coo_tensor(indices, values=torch.ones_like(indices[0, :], dtype=torch.float32),
#                                           size=size)
#         indices = torch.sparse.softmax(indices, dim=len(indices.size()) - 1)
#         return torch.sparse.mm(indices, x)


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
    def __init__(self, hidden_dim):
        super(Time2Vec, self).__init__()
        self.l1 = CosineActivation(1, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x



def get_train_set(batch_size=batch_size):
    #这里改neg_num
    negative_ratio = batch_size // len(train_alignments) + 1
    train_set = np.reshape(np.repeat(np.expand_dims(train_alignments, axis=0), axis=0, repeats=negative_ratio),
                           newshape=(-1, 2))
    np.random.shuffle(train_set);
    train_set = train_set[:batch_size]
    train_set = np.concatenate([train_set, np.random.randint(0, node_size, train_set.shape)], axis=-1)
    return train_set

class Simple_HHEA(nn.Module):
    def __init__(self, ent_pair_size, time_span, ent_dw_emb, ent_name_emb, ent_time_emb, ent_rel_emb, emb_size=64, device='cuda'):
        super(Simple_HHEA, self).__init__()

        self.device = device
        self.emb_size = emb_size
        # self.indices = self.adj.coalesce().indices()
        # self.indices_size = self.indices.size(1)
        self.activation = nn.ReLU()
        self.time_span = time_span
        # self.rel_translation = nn.Linear(rel_size[0], rel_size[1])
        self.fc_final = nn.Linear(emb_size * 2, emb_size)
        self.fc_name = nn.Linear(emb_size, emb_size)
        self.fc_time = nn.Linear(32, 16)
        # self.fc_structure = nn.Linear(ent_pair_size, 128)
        self.time2vec = Time2Vec(32)
        self.fc_degree = nn.Linear(1, 1)
        self.fc_dw = nn.Linear(emb_size,16)
        # self.time_translation = nn.Linear(emb_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p = 0.1)
        #         self.graph_net = TNR_GraphAttention(adj, node_size, rel_size, time_size, triple_size, emb_size, depth, device)
        # self.ent_emb = self.ent_embedding(ent_sparse).to(self.device)
        # self.rel_emb = self.rel_embedding(rel_sparse).to(self.device)
        # self.time_emb = self.time_embedding(time_sparse).to(self.device)

        self.ent_name_emb = torch.tensor(ent_name_emb).to(self.device).float()
        self.node_num = self.ent_name_emb.size(0)

        # self.ent_degree_emb = torch.sum(torch.tensor(ent_structure_emb).to(self.device).float(), dim = 1).reshape(self.node_num, 1) / ent_pair_size
        # self.ent_structure_emb = F.softmax(torch.tensor(ent_structure_emb).to(self.device).float())

        ent_time_emb = torch.tensor(ent_time_emb).to(self.device).float()
        # self.ent_time_dense = F.softmax(ent_time_emb, dim = 1)
        self.ent_time_dense = ent_time_emb
        self.ent_rel_emb = torch.tensor(ent_rel_emb).to(self.device).float()
        time_span_index = [i for i in range(time_span)]
        self.time_span_index = torch.tensor(time_span_index).to(self.device).unsqueeze(1).float()
        self.ent_dw_emb = torch.tensor(ent_dw_emb).to(self.device).float()

    #后续如果需要batch_size的话改这里
    def forward(self):
        time_span_feature = self.time2vec(self.time_span_index)#352 * 64
        # r_time_span_feature = time_span_feature.reshape(time_span_feature.shape[0], time_span_feature.shape[1], 1)
        # ent_time_dense = F.softmax(self.sigmoid(self.ent_time_emb), dim = 1)
        ent_time_feature = torch.mm(self.ent_time_dense , time_span_feature) / self.time_span
        ent_time_feature = self.fc_time(ent_time_feature)
        ent_name_feature = self.fc_name(self.ent_name_emb)
        ent_dw_feature = self.fc_dw(self.ent_dw_emb)
        # ent_structure_feature = self.fc_structure(self.ent_structure_emb)
        # ent_degree_feature =  self.fc_degree(self.ent_degree_emb)

        #TODO
        # ent_rel_emb

        # ent_time_emb_new = ent_time_emb

        outputs = []
        # features = torch.cat((self.name_emb, ent_time_feature),1)

        # time_encoding = self.activation(torch.sparse.mm(time_indices.float(), times_sum_n))

        # return self.fc_final(torch.cat([ent_name_feature, ent_time_feature], 1))
        #TODO这里决定是否有structure
        if args.if_structure:
            output_feature = self.dropout(torch.cat([ent_name_feature, ent_time_feature, ent_dw_feature], 1))
        else:
            output_feature = self.dropout(torch.cat([ent_name_feature, ent_time_feature], 1))
        # output_feature = self.dropout(torch.cat([ent_structure_feature, ent_time_feature, ent_degree_feature], 1))
        # output_feature = self.dropout(ent_name_feature)
        return output_feature
        # return self.fc_name(self.ent_name_emb)
        # return self.ent_name_emb

alignment_pairs = get_train_set()

kg1_name_emb = np.loadtxt(file_path + 'ent_1_emb_64.txt')
kg2_name_emb = np.loadtxt(file_path + 'ent_2_emb_64.txt')
ent_name_emb = np.array(kg1_name_emb.tolist() + kg2_name_emb.tolist())
print("read entity time embedding shape:", np.array(ent_name_emb).shape)
import random
#来点噪音hhhh

# ent_structure_emb = pickle.load(open(STRUCTURE_EMB_PATH, "rb"))
# print("read entity structure embedding shape:", np.array(ent_structure_emb).shape)

kg1_time_emb = np.loadtxt(file_path + 'ent_1_time_emb.txt')
kg2_time_emb = np.loadtxt(file_path + 'ent_2_time_emb.txt')
ent_time_emb = np.array(kg1_time_emb.tolist() + kg2_time_emb.tolist())
print("read entity time embedding shape:", np.array(ent_time_emb).shape)

noise_ratio = args.noise_ratio
sample_list = [i for i in range(352)]
mask_id = random.sample(sample_list, int(352 * noise_ratio))
ent_time_emb[:, mask_id] = 0
# ent_time_emb = load_ent_time_matrix(file_path)

ent_context_emb = np.loadtxt(file_path + 'context_emb.txt')
# ent_time_emb = np.array(kg1_time_emb.tolist() + kg2_time_emb.tolist())
print("read entity context_emb shape:", np.array(ent_context_emb).shape)

ent_dw_emb = np.loadtxt(file_path + 'deep_emb.txt')
# ent_time_emb = np.array(kg1_time_emb.tolist() + kg2_time_emb.tolist())
print("read entity transe emb shape:", np.array(ent_dw_emb).shape)
noise_ratio = args.noise_ratio
sample_list = [i for i in range(64)]
mask_id = random.sample(sample_list, int(64 * noise_ratio))
ent_dw_emb[:, mask_id] = 0

ent_transe_emb = np.loadtxt(file_path + 'transe_emb.txt')
# ent_time_emb = np.array(kg1_time_emb.tolist() + kg2_time_emb.tolist())
print("read entity deepwalk emb shape:", np.array(ent_transe_emb).shape)


# ent_rel_emb = pickle.load(open(REL_EMB_PATH, "rb"))
# print("read entity relation embedding shape:", np.array(ent_rel_emb).shape)
#TODO 加rel emb
ent_rel_emb = torch.zeros((26943, 498))


model = Simple_HHEA(ent_pair_size, time_range, ent_dw_emb, ent_name_emb, ent_time_emb, ent_rel_emb, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)


def l1(ll, rr):
    return torch.sum(torch.abs(ll - rr), axis=-1)


gamma = 1.0
losses = []
t_prec = []
accs = []
t_mrrs = []
best_acc = [0, 0, 0]
best_mrr = 0
for i in tqdm(range(1500)):
    model.train()
    optimizer.zero_grad()
    #TODO 后续需要改batch实验的话，改这里
    #(r_index_sq, r_val_sq, t_index_sq) 指的是attention的几个要素
    features = model()
    comp = features[alignment_pairs]
    l, r, fl, fr = comp[:, 0, :], comp[:, 1, :], comp[:, 2, :], comp[:, 3, :]
    loss = torch.sum(nn.ReLU()(gamma + l1(l, r) - l1(l, fr)) + nn.ReLU()(gamma + l1(l, r) - l1(fl, r))) / batch_size
    losses.append(loss.item())
    loss.backward(retain_graph=True)
    optimizer.step()
    if ((i + 1) % 10) == 0:
        model.eval()
        with torch.no_grad():
            feat = model()[dev_alignments]
            Lvec = np.array(feat[:, 0, :].cpu())  # np.array([vec[e1] for e1, e2 in feat])
            Rvec = np.array(feat[:, 1, :].cpu())  # np.array([vec[e2] for e1, e2 in dev_alignments])
            Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
            Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
            t_prec_set, acc, t_mrr = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 1, csls=10, accurate=True)
            if best_acc[0] < acc[0]:
                best_acc[0] = acc[0]
            if best_acc[1] < acc[1]:
                best_acc[1] = acc[1]
            if best_acc[2] < acc[2]:
                best_acc[2] = acc[2]
            if best_acc[2] < acc[2]:
                best_acc[2] = acc[2]
            if best_mrr < t_mrr:
                best_mrr = t_mrr

            print("//best results: hits@[1, 5, 10] = {}, mrr = {:.3f}// ".format(best_acc, best_mrr))
            accs.append(acc)
            t_mrrs.append(t_mrr)
            t_prec.append(t_prec_set)
result_file = open(file_path + 'result_file_structure_ratio.txt', 'a', encoding='utf-8')
result_file.write("settings: noise_ratio: {}, if_structure {}\n best results: hits@[1, 5, 10] = {}, mrr = {:.3f}\n".format(str(args.noise_ratio), str(args.if_structure), best_acc, best_mrr))


