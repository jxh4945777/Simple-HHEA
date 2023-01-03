from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.basicConfig(level=logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pickle
import time
from Param import *
from utils import fixed,cos_sim_mat_generate,batch_topk,load_triples,load_aligned_pair
# from ..Basic_Bert_Unit_model import Basic_Bert_Unit_model


def candidate_generate(ents1,ents2,ent_emb,candidate_topk = 50,bs = 32, cuda_num = 0):
    """
    return a dict, key = entity, value = candidates (likely to be aligned entities)
    """
    emb1 = np.array(ent_emb)[ ents1 ].tolist()
    emb2 = np.array(ent_emb)[ ents2 ].tolist()
    print("Test(get candidate) embedding shape:",np.array(emb1).shape,np.array(emb2).shape)
    print("get candidate by cosine similartity.")
    res_mat = cos_sim_mat_generate(emb1,emb2,bs,cuda_num=cuda_num)
    score,index = batch_topk(res_mat,bs,candidate_topk,largest=True,cuda_num=cuda_num)
    ent2candidates = dict()
    for i in range( len(index) ):
        e1 = ents1[i]
        e2_list = np.array(ents2)[index[i]].tolist()
        ent2candidates[e1] = e2_list
    return ent2candidates


def all_entity_pairs_gene(candidate_dict_list, ill_pair_list):
    #generate list of all candidate entity pairs.
    entity_pairs_list = []
    for candidate_dict in candidate_dict_list:
        for e1 in candidate_dict.keys():
            for e2 in candidate_dict[e1]:
                entity_pairs_list.append((e1, e2))
    for ill_pair in ill_pair_list:
        for e1, e2 in ill_pair:
            entity_pairs_list.append((e1, e2))
    entity_pairs_list = list(set(entity_pairs_list))
    print("entity_pair (e1,e2) num is: {}".format(len(entity_pairs_list)))
    return entity_pairs_list




def main():
    print("----------------get entity embedding--------------------")
    cuda_num = CUDA_NUM
    batch_size = 256
    print("GPU NUM:",cuda_num)
    #读取entity embedding
    file_path = "../data/fr_en/"
    kg1_emb = np.loadtxt(file_path + 'ent_1_emb_64.txt')
    kg2_emb = np.loadtxt(file_path + 'ent_2_emb_64.txt')
    ent_emb = np.array(kg1_emb.tolist() + kg2_emb.tolist())

    #读取KGs以及train/test entity pairs
    all_triples, node_size, rel_size = load_triples(file_path, True)
    train_ill, test_ill = load_aligned_pair(file_path, ratio=TRAIN_RATIO)
    train_ill = train_ill.tolist()
    test_ill = test_ill.tolist()
    train_ill = [tuple(ent_pairs_now) for ent_pairs_now in train_ill]
    test_ill = [tuple(ent_pairs_now) for ent_pairs_now in test_ill]

    print("train_ill num: {} /test_ill num: {} / train_ill & test_ill num: {}".format(len(train_ill),len(test_ill),
                                                                                 len(set(train_ill) & set(test_ill) )))


    #generate entity embedding by basic bert unit
    print("entity embedding shape: ", np.array(ent_emb).shape)

    #save entity embedding.
    pickle.dump(ent_emb, open(ENT_EMB_PATH, "wb"))
    print("save entity embedding....")

    kg1_structure_emb = np.loadtxt(file_path + 'ent_1_neighbor_emb.txt')
    kg2_structure_emb = np.loadtxt(file_path + 'ent_2_neighbor_emb.txt')
    ent_structure_emb = np.array(kg1_structure_emb.tolist() + kg2_structure_emb.tolist())

    #generate entity embedding by basic bert unit
    print("structure entity embedding shape: ", np.array(ent_structure_emb).shape)

    #save entity embedding.
    pickle.dump(ent_structure_emb, open(STRUCTURE_EMB_PATH, "wb"))
    print("save structure entity embedding....")


    # kg1_time_emb = np.loadtxt(file_path + 'ent_1_time_emb.txt')
    # kg2_time_emb = np.loadtxt(file_path + 'ent_2_time_emb.txt')
    # ent_time_emb = np.array(kg1_time_emb.tolist() + kg2_time_emb.tolist())
    #
    # #generate entity embedding by basic bert unit
    # print("time entity embedding shape: ", np.array(ent_time_emb).shape)

    # #save entity embedding.
    # pickle.dump(ent_time_emb, open(TIME_EMB_PATH, "wb"))
    # print("save time entity embedding....")

    # kg1_rel_emb = np.loadtxt(file_path + 'ent_1_rel_emb.txt')
    # kg2_rel_emb = np.loadtxt(file_path + 'ent_2_rel_emb.txt')
    # ent_rel_emb = np.array(kg1_rel_emb.tolist() + kg2_rel_emb.tolist())
    #
    # #generate entity embedding by basic bert unit
    # print("rel entity embedding shape: ", np.array(ent_rel_emb).shape)

    #save entity embedding.
    pickle.dump(ent_rel_emb, open(REL_EMB_PATH, "wb"))
    print("save rel entity embedding....")

    #Generate candidates(likely to be aligned) for entities in train_set/test_set
    #we apply interaction model to infer a matching score on candidates.
    test_ids_1 = [e1 for e1, e2 in test_ill]
    test_ids_2 = [e2 for e1, e2 in test_ill]
    train_ids_1 = [e1 for e1, e2 in train_ill]
    train_ids_2 = [e2 for e1, e2 in train_ill]
    train_candidates = candidate_generate(train_ids_1, train_ids_2, ent_emb, CANDIDATE_NUM, bs=2048, cuda_num=CUDA_NUM)
    test_candidates = candidate_generate(test_ids_1, test_ids_2, ent_emb, CANDIDATE_NUM, bs=2048, cuda_num=CUDA_NUM)
    pickle.dump(train_candidates, open(TRAIN_CANDIDATES_PATH, "wb"))
    print("save candidates for training ILL data....")
    pickle.dump(test_candidates, open(TEST_CANDIDATES_PATH, "wb"))
    print("save candidates for testing ILL data....")

    #entity_pairs (entity_pairs is list of (likely to be aligned) entity pairs : [(e1,ea),(e1,eb),(e1,ec) ....])
    entity_pairs = all_entity_pairs_gene([ train_candidates,test_candidates ],[ train_ill ])
    pickle.dump(entity_pairs, open(ENT_PAIRS_PATH, "wb"))
    print("save entity_pairs save....")
    print("entity_pairs num: {}".format(len(entity_pairs)))


if __name__ == '__main__':
    fixed(SEED_NUM)
    main()







