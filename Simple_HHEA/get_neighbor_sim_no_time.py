from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
from tqdm import tqdm
from Param import *
from utils import fixed,cos_sim_mat_generate,batch_topk
# from ..Basic_Bert_Unit_model import Basic_Bert_Unit_model
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
def cos_sim(vec1, vec2):
    if (np.linalg.norm(vec1) * np.linalg.norm(vec2)) == 0:
        return 0
    return vec1.dot(vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def softmax(x):
    x -= np.max(x)
    x = np.exp(x) / np.sum(np.exp(x))
    return x

def sigmoid(x):
    return 1.0/(1+np.exp(-x))
    # return x

def norm(x):
    Min = np.min(x)
    Max = np.max(x)
    if Max-Min == 0:
        return x
    else:
        x = (x - Min) / (Max - Min)
        return x

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

    total_neighbor_sim = 0
    for i in range(ent_1_num):
        ent_1_emb[i,:] = ent_1_emb[i,:]
        pass
    for i in range(ent_2_num):
        ent_2_emb[i,:] = ent_2_emb[i,:]

    for i in ent_pairs:
        total_neighbor_sim += cos_sim(ent_1_emb[int(i[0]),:], ent_2_emb[int(i[1]) - ent_1_num, :])

    np.savetxt(file_path + "ent_1_time_emb.txt".format(1 + 27 * 13), ent_1_emb)
    np.savetxt(file_path + "ent_2_time_emb.txt".format(1 + 27 * 13), ent_2_emb)
    print('The avg neighbor sim: ' + str(total_neighbor_sim / len(ent_pairs)) + '\n')

    return
    #dbp_en_fr: -
    #icews_wiki: 0.2956733226083592
    #icews_yago: 0.4516185381983681

def load_ent_time_matrix_new(file_path):
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
                ent_1_emb[int(float(line[0])), 0] = 1
            else:
                ent_1_emb[int(float(line[0])), rel_time_cal(time_y, time_m)] = 1

        if time_dict[line[4]] != '~':
            time_y = int(float(time_dict[line[4]].split('-')[0]))
            time_m = int(float(time_dict[line[4]].split('-')[1]))
            if time_y < 1995:
                ent_1_emb[int(float(line[0])), 0] = 1
            else:
                ent_1_emb[int(float(line[0])), rel_time_cal(time_y, time_m)] = 1

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
                ent_2_emb[int(float(line[0])) - ent_1_num, 0] = 1
                time_y_s = 1995
                time_m_s = 0

        if time_dict[line[4]] != '~' and time_dict[line[3]] != '-400000':
            time_y_e = int(float(time_dict[line[4]].split('-')[0]))
            time_m_e = int(float(time_dict[line[4]].split('-')[1]))
            # ent_2_emb[int(float(line[0])) - ent_1_num, 0] += 1
            if time_y_e >= 1995:
                ent_2_emb[int(float(line[0])) - ent_1_num, rel_time_cal(time_y_s, time_m_s): rel_time_cal(time_y_e, time_m_e)] = 1

    total_neighbor_sim = 0
    for i in ent_pairs:
        total_neighbor_sim += cos_sim(ent_1_emb[int(i[0]),:], ent_2_emb[int(i[1]) - ent_1_num, :])
    for i in range(ent_1_num):
        ent_now = ent_1_emb[i,:]
        ent_1_emb[i,:] = ent_1_emb[i,:]
        pass
    for i in range(ent_2_num):
        ent_2_emb[i,:] = ent_2_emb[i,:]

    np.savetxt(file_path + "ent_1_time_emb.txt".format(1 + 27 * 13), ent_1_emb)
    np.savetxt(file_path + "ent_2_time_emb.txt".format(1 + 27 * 13), ent_2_emb)
    print('The avg neighbor sim: ' + str(total_neighbor_sim / len(ent_pairs)) + '\n')

    return
    #dbp_en_fr: -
    #icews_wiki: 0.2956733226083592
    #icews_yago: 0.4516185381983681


def load_ent_relation_matrix(file_path):
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

    rel_dict_1 = dict()
    for line in open(file_path + 'rel_ids_1', 'r+', encoding='utf-8'):
        if line != '':
            line = line.replace('\n','').split('\t')
            rel_dict_1[line[0]] = line[1]

    rel_dict_2 = dict()
    for line in open(file_path + 'rel_ids_2', 'r+', encoding='utf-8'):
        if line != '':
            line = line.replace('\n','').split('\t')
            rel_dict_2[line[0]] = line[1]

    #这里代表年份
    #2 (old) + [(2021-1995) + 1] * 13 (这里包括00月)
    ent_1_emb = np.zeros([ent_1_num, len(rel_dict_1.keys())])
    ent_2_emb = np.zeros([ent_2_num, len(rel_dict_2.keys())])

    for line in tqdm(open(file_path + 'triples_1', 'r+', encoding='utf-8')):
        line = line.replace('\n','').split('\t')
        ent_1_emb[int(line[0]), int(line[1])] += 1

    for line in tqdm(open(file_path + 'triples_2', 'r+', encoding='utf-8')):
        line = line.replace('\n','').split('\t')
        ent_2_emb[int(line[0]) - ent_1_num, int(line[1]) - len(rel_dict_1.keys())] += 1


    total_neighbor_sim = 0
    # for i in ent_pairs:
    #     total_neighbor_sim += cos_sim(softmax(ent_1_emb[int(i[0]),:]), softmax(ent_2_emb[int(i[1]) - ent_1_num, :]))
    #
    for i in range(ent_1_num):
        ent_1_emb[i,:] = softmax(ent_1_emb[i,:])
    for i in range(ent_2_num):
        ent_2_emb[i,:] = softmax(ent_2_emb[i,:])

    np.savetxt(file_path + "ent_1_rel_emb.txt".format(len(rel_dict_1.keys())), ent_1_emb)
    np.savetxt(file_path + "ent_2_rel_emb.txt".format(len(rel_dict_2.keys())), ent_2_emb)
    # print('The avg neighbor sim: ' + str(total_neighbor_sim / len(ent_pairs)) + '\n')

    return

def load_ent_pair_matrix(file_path):
    ent_1_num = 0
    ent_2_num = 0
    ent_1_id2pos = dict()
    ent_2_id2pos = dict()
    for line in open(file_path + 'ent_ids_1', 'r+', encoding='utf-8'):
        if line != '':
            line = line.split('\t')
            ent_1_id2pos[line[0]] = ent_1_num
            ent_1_num += 1
    for line in open(file_path + 'ent_ids_2', 'r+', encoding='utf-8'):
        if line != '':
            line = line.split('\t')
            ent_2_id2pos[line[0]] = ent_2_num
            ent_2_num += 1
    ent_1_id = dict()
    ent_2_id = dict()
    ent_pairs = []
    ent_1_list = []
    ent_2_list = []
    entity_count = 0
    for line in open(file_path + 'ref_ent_ids', 'r+', encoding='utf-8'):
        line = line.replace('\n','').split('\t')
        ent_pairs.append((line[0], line[1]))
        ent_1_list.append(line[0])
        ent_2_list.append(line[1])
        ent_1_id[line[0]] = entity_count
        ent_2_id[line[1]] = entity_count
        entity_count += 1
    matrix_dim = len(ent_pairs)
    # ent_sub_emb = np.zeros([ent_1_num + ent_2_num, matrix_dim])
    ent_1_emb = np.zeros([ent_1_num, matrix_dim])
    ent_2_emb = np.zeros([ent_2_num, matrix_dim])
    ent_pair_matrix_1 = np.zeros([matrix_dim, matrix_dim])
    ent_pair_matrix_2 = np.zeros([matrix_dim, matrix_dim])

    for line in tqdm(open(file_path + 'triples_1', 'r+', encoding='utf-8')):
        line = line.replace('\n','').split('\t')
        if (line[0] in ent_1_list) and (line[2] in ent_1_list):
            head_ent_id = ent_1_id[line[0]]
            tail_ent_id = ent_1_id[line[2]]
            ent_pair_matrix_1[head_ent_id, tail_ent_id] = 1
            ent_pair_matrix_1[tail_ent_id, head_ent_id] = 1

    for line in tqdm(open(file_path + 'triples_2', 'r+', encoding='utf-8')):
        line = line.replace('\n','').split('\t')
        if (line[0] in ent_2_list) and (line[2] in ent_2_list):
            head_ent_id = ent_2_id[line[0]]
            tail_ent_id = ent_2_id[line[2]]
            ent_pair_matrix_2[head_ent_id, tail_ent_id] = 1
            ent_pair_matrix_2[tail_ent_id, head_ent_id] = 1

    total_neighbor_sim = 0
    for i in tqdm(range(matrix_dim)):
        total_neighbor_sim += cos_sim(ent_pair_matrix_1[i,:], ent_pair_matrix_2[i, :])
        ent_1_emb[ent_1_id2pos[ent_1_list[i]],:] = ent_pair_matrix_1[i,:]
        ent_2_emb[ent_2_id2pos[ent_2_list[i]],:] = ent_pair_matrix_2[i,:]

    print('The avg neighbor sim: ' + str(total_neighbor_sim / matrix_dim) + '\n')
    np.savetxt(file_path + "ent_1_neighbor_emb.txt".format(matrix_dim), ent_1_emb)
    np.savetxt(file_path + "ent_2_neighbor_emb.txt".format(matrix_dim), ent_2_emb)

    return ent_pair_matrix_1, ent_pair_matrix_2,ent_1_id, ent_2_id
    #dbp_en_fr: 0.6336324836185819
    #icews_wiki: 0.11135192569120823
    #icews_yago: 0.13999421825263766

def main():
    print("----------------get neighbor sim--------------------")
    # file_path = "KGs/dbp15k/fr_en"
    cuda_num = CUDA_NUM
    batch_size = 256
    print("GPU NUM:",cuda_num)
    #读取entity embedding
    kg_sim_matrix_1, kg_sim_matrix_2,  ent2id_kg1, ent2id_kg2 = load_ent_pair_matrix(FILE_PATH)
    # load_ent_time_matrix(FILE_PATH)
    # load_ent_relation_matrix(FILE_PATH)
    # pass

    # #读取KGs以及train/test entity pairs
    # all_triples, node_size, rel_size = load_triples(file_path, True)
    #
    # train_ill, test_ill = load_aligned_pair(file_path, ratio=TRAIN_RATIO)
    # train_ill = train_ill.tolist()
    # test_ill = test_ill.tolist()
    # train_ill = [tuple(ent_pairs_now) for ent_pairs_now in train_ill]
    # test_ill = [tuple(ent_pairs_now) for ent_pairs_now in test_ill]
    #
    # print("train_ill num: {} /test_ill num: {} / train_ill & test_ill num: {}".format(len(train_ill),len(test_ill),
    #                                                                              len(set(train_ill) & set(test_ill) )))
    #
    # #Generate candidates(likely to be aligned) for entities in train_set/test_set
    # #we apply interaction model to infer a matching score on candidates.
    # test_ids_1 = [e1 for e1, e2 in test_ill]
    # test_ids_2 = [e2 for e1, e2 in test_ill]
    # train_ids_1 = [e1 for e1, e2 in train_ill]
    # train_ids_2 = [e2 for e1, e2 in train_ill]

if __name__ == '__main__':
    fixed(SEED_NUM)
    main()