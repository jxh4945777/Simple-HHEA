import os
import pickle
import logging
logging.basicConfig(level=logging.ERROR)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from read_data_func import read_structure_datas
from Param import *
from utils import *
from dual_aggregation_func import *



def _test_read_emb(ent_emb,train_ill,test_ill, bs = 128,candidate_topk = 50):
    test_ids_1 = [e1 for e1, e2 in test_ill]
    test_ids_2 = [e2 for e1, e2 in test_ill]
    test_emb1 = np.array(ent_emb)[test_ids_1].tolist()
    test_emb2 = np.array(ent_emb)[test_ids_2].tolist()
    train_ids_1 = [e1 for e1, e2 in train_ill]
    train_ids_2 = [e2 for e1, e2 in train_ill]
    train_emb1 = np.array(ent_emb)[train_ids_1].tolist()
    train_emb2 = np.array(ent_emb)[train_ids_2].tolist()

    print("Eval entity emb sim in train set.")
    emb1 = train_emb1
    emb2 = train_emb2

    res_mat = cos_sim_mat_generate(emb1, emb2, bs)
    score, index = batch_topk(res_mat, bs, candidate_topk, largest=True)
    test_topk_res(index)
    
    print("Eval entity emb sim in test set.")
    emb1 = test_emb1
    emb2 = test_emb2

    res_mat = cos_sim_mat_generate(emb1, emb2, bs)
    score, index = batch_topk(res_mat, bs, candidate_topk, largest=True)
    test_topk_res(index)




def neighborView_interaction_F_gene(ent_pairs, ent_emb_list, neigh_dict, ent_pad_id,
                                    kernel_num = 21,cuda_num = 0,batch_size = 512):
    """
    Neighbor-View Interaction.
    use Dual Aggregation and Neighbor-View Interaction to generate Similarity Feature between entity pairs.
    return entity pairs and features(between entity pairs)
    """
    start_time = time.time()
    e_emb = torch.FloatTensor(ent_emb_list).cuda(cuda_num)
    e_emb = F.normalize(e_emb,p=2,dim=-1)
    sigmas = kernel_sigmas(kernel_num).cuda(cuda_num)
    mus = kernel_mus(kernel_num).cuda(cuda_num)
    # print("sigmas:",sigmas)
    # print("mus:",mus)
    sigmas = sigmas.view(1,1,-1)     
    mus = mus.view(1,1,-1)
    
    all_ent_pairs = []
    all_features = []
    # print("entity_embedding shape:",e_emb.shape)
    for start_pos in range(0,len(ent_pairs),batch_size):
        batch_ent_pairs = ent_pairs[start_pos : start_pos + batch_size]
        e1s = [e1 for e1,e2 in batch_ent_pairs]
        e2s = [e2 for e1,e2 in batch_ent_pairs]
        #构造矩阵中每行，取50个neighbor
        e1_tails = [neigh_dict[e1] for e1 in e1s]#size: [B(Batchsize),ne1(e1_neighbor_max_num)]
        e2_tails = [neigh_dict[e2] for e2 in e2s]#[B,ne2]
        e1_masks = np.ones(np.array(e1_tails).shape)
        e2_masks = np.ones(np.array(e2_tails).shape)
        #这里是neighbor不够的情况下，补0
        e1_masks[np.array(e1_tails) == ent_pad_id] = 0
        e2_masks[np.array(e2_tails) == ent_pad_id] = 0
        e1_masks = torch.FloatTensor(e1_masks.tolist()).cuda(cuda_num).unsqueeze(-1)#[B,ne1,1]
        e2_masks = torch.FloatTensor(e2_masks.tolist()).cuda(cuda_num).unsqueeze(-1)#[B,ne2,1]
        e1_tails = torch.LongTensor(e1_tails).cuda(cuda_num)#[B,ne1]
        e2_tails = torch.LongTensor(e2_tails).cuda(cuda_num)#[B,ne2]
        e1_tail_emb = e_emb[e1_tails]#[B,ne1,embedding_dim]
        e2_tail_emb = e_emb[e2_tails]#[B,ne2,embedding_dim]
        #这里通过emb计算矩阵中的元素相似度
        #sim_matrix: 2048*50*50, 我觉得这里完全可以构建relation semantic和temporal的sim_matrix
        sim_matrix = torch.bmm(e1_tail_emb,torch.transpose(e2_tail_emb,1,2))#[B,ne1,ne2]
        features = batch_dual_aggregation_feature_gene(sim_matrix,mus,sigmas,e1_masks,e2_masks)
        features = features.detach().cpu().tolist()
        all_ent_pairs.extend(batch_ent_pairs)
        all_features.extend(features)


    print("all ent pair neighbor-view interaction features shape:",np.array(all_features).shape)
    print("get ent pair neighbor-view interaction features using time {:.3f}".format(time.time()-start_time))
    return all_ent_pairs,all_features



def desornameView_interaction_F_gene(ent_pairs, e_emb_list,
                      cuda_num = 0,batch_size = 512):
    start_time = time.time()
    e_emb = torch.FloatTensor(e_emb_list).cuda(cuda_num)
    all_ent_pairs = []
    all_features = []
    # print("entity embedding shape:", e_emb.shape)
    for start_pos in range(0,len(ent_pairs),batch_size):
        batch_ent_pairs = ent_pairs[start_pos : start_pos + batch_size]
        e1s = [e1 for e1,e2 in batch_ent_pairs]#[B]
        e2s = [e2 for e1,e2 in batch_ent_pairs]
        e1s = torch.LongTensor(e1s).cuda(cuda_num)
        e2s = torch.LongTensor(e2s).cuda(cuda_num)
        e1_embs = e_emb[e1s]#[B,embedding_dim]
        e2_embs = e_emb[e2s]
        cos_sim = F.cosine_similarity(e1_embs,e2_embs)
        cos_sim = cos_sim.detach().cpu().unsqueeze(-1).tolist()
        all_ent_pairs.extend(batch_ent_pairs)
        all_features.extend(cos_sim)

    print("all ent description/name-view interaction feature shape:",np.array(all_features).shape)
    print("get ent description/name-view interaction feature using time {:.3f}".format(time.time() - start_time))
    return all_ent_pairs,all_features

def structureView_interaction_F_gene(ent_pairs, e_emb_list,
                      cuda_num = 0,batch_size = 512):
    start_time = time.time()
    e_emb = torch.FloatTensor(e_emb_list).cuda(cuda_num)
    all_ent_pairs = []
    all_features = []
    # print("entity embedding shape:", e_emb.shape)
    for start_pos in range(0,len(ent_pairs),batch_size):
        batch_ent_pairs = ent_pairs[start_pos : start_pos + batch_size]
        e1s = [e1 for e1,e2 in batch_ent_pairs]#[B]
        e2s = [e2 for e1,e2 in batch_ent_pairs]
        e1s = torch.LongTensor(e1s).cuda(cuda_num)
        e2s = torch.LongTensor(e2s).cuda(cuda_num)
        e1_embs = e_emb[e1s]#[B,embedding_dim]
        e2_embs = e_emb[e2s]
        cos_sim = F.cosine_similarity(e1_embs,e2_embs)
        cos_sim = cos_sim.detach().cpu().unsqueeze(-1).tolist()
        all_ent_pairs.extend(batch_ent_pairs)
        all_features.extend(cos_sim)

    print("all ent structure-view interaction feature shape:",np.array(all_features).shape)
    print("get ent structure-view interaction feature using time {:.3f}".format(time.time() - start_time))
    return all_ent_pairs,all_features

def timeView_interaction_F_gene(ent_pairs, e_emb_list,
                      cuda_num = 0,batch_size = 512):
    start_time = time.time()
    e_emb = torch.FloatTensor(e_emb_list).cuda(cuda_num)
    all_ent_pairs = []
    all_features = []
    # print("entity embedding shape:", e_emb.shape)
    for start_pos in range(0,len(ent_pairs),batch_size):
        batch_ent_pairs = ent_pairs[start_pos : start_pos + batch_size]
        e1s = [e1 for e1,e2 in batch_ent_pairs]#[B]
        e2s = [e2 for e1,e2 in batch_ent_pairs]
        e1s = torch.LongTensor(e1s).cuda(cuda_num)
        e2s = torch.LongTensor(e2s).cuda(cuda_num)
        e1_embs = e_emb[e1s]#[B,embedding_dim]
        e2_embs = e_emb[e2s]
        cos_sim = F.cosine_similarity(e1_embs,e2_embs)
        cos_sim = cos_sim.detach().cpu().unsqueeze(-1).tolist()
        all_ent_pairs.extend(batch_ent_pairs)
        all_features.extend(cos_sim)

    print("all ent time-view interaction feature shape:",np.array(all_features).shape)
    print("get ent time-view interaction feature using time {:.3f}".format(time.time() - start_time))
    return all_ent_pairs,all_features

def main():
    print("----------------get multi view interaction feature--------------------")
    cuda_num = CUDA_NUM
    print("GPU num: {}".format(cuda_num))

    # read structure data
    ent_ill, index2rel, index2entity, \
    rel2index, entity2index, \
    rel_triples_1, rel_triples_2 = read_structure_datas(DATA_PATH)
    rel_triples = []#all relation triples
    rel_triples.extend(rel_triples_1)
    rel_triples.extend(rel_triples_2)

    # load entity embedding
    #TODO 这里修改ent_emb
    ent_emb = pickle.load(open(ENT_EMB_PATH,"rb"))
    print("read entity embedding shape:", np.array(ent_emb).shape)

    structure_ent_emb = pickle.load(open(STRUCTURE_EMB_PATH,"rb"))
    print("read entity structure embedding shape:", np.array(structure_ent_emb).shape)

    time_ent_emb = pickle.load(open(TIME_EMB_PATH,"rb"))
    print("read entity time embedding shape:", np.array(time_ent_emb).shape)

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)
    #TODO 这里读取跨网络实体对
    bert_model_other_data_path = BASIC_BERT_UNIT_MODEL_SAVE_PATH + BASIC_BERT_UNIT_MODEL_SAVE_PREFIX + 'other_data.pkl'
    # train_ill, test_ill, eid2data = pickle.load(open(bert_model_other_data_path, "rb"))
    file_path = FILE_PATH
    train_ill, test_ill = load_aligned_pair(file_path, ratio=TRAIN_RATIO)
    train_ill = train_ill.tolist()
    test_ill = test_ill.tolist()
    train_ill = [tuple(ent_pairs_now) for ent_pairs_now in train_ill]
    test_ill = [tuple(ent_pairs_now) for ent_pairs_now in test_ill]
    
    #test entity embedding
    _test_read_emb(ent_emb, train_ill,test_ill,bs=256,candidate_topk=CANDIDATE_NUM)

    #test entity structure embedding
    _test_read_emb(structure_ent_emb, train_ill,test_ill,bs=256,candidate_topk=CANDIDATE_NUM)

    #test entity structure embedding
    _test_read_emb(time_ent_emb, train_ill,test_ill,bs=256,candidate_topk=CANDIDATE_NUM)

    #read pair_index list
    #这里entity pairs指的是KG1和KG2的候选pair
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))
    
    #set <PAD> (entity pad symbol) id and <PAD> embedding.
    ent_pad_id = len(ent_emb)
    dim = len(ent_emb[0])
    ent_emb = ent_emb.tolist()
    ent_emb.append([0.0 for _ in range(dim)]) #<PAD> embedding
    ent_emb = np.array(ent_emb)
    index2entity[ent_pad_id] = '<PAD>'
    
    #neigh_dict(key = entity ,value = (padding) neighbors of entity)
    neigh_dict = neigh_ent_dict_gene(rel_triples,max_length=ENTITY_NEIGH_MAX_NUM,
                                     pad_id=ent_pad_id)
    #generate neighbor-view interaction feature
    #entity_pairs是所有facts的头实体&尾实体；ent_emb是所有entity的embedding，neigh_dict是每个entity的邻居信息，ent_pad_id是token的索引值
    #这里对于neighView的信息，考虑到我们的数据集facts过多，因此可能需要再进行一次处理，例如只处理对齐的实体相关facts
    entity_pairs_1,neighViewInterF = neighborView_interaction_F_gene(entity_pairs,ent_emb,neigh_dict,ent_pad_id,
                                  kernel_num=KERNEL_NUM,cuda_num=cuda_num,batch_size=2048)

    #generate description/name-view interaction feature
    entity_pairs_2,desViewInterF = desornameView_interaction_F_gene(entity_pairs,ent_emb,
                                                    cuda_num=cuda_num,batch_size=512)

    #generate description/name-view interaction feature
    entity_pairs_3,structureViewInterF = structureView_interaction_F_gene(entity_pairs,structure_ent_emb,
                                                    cuda_num=cuda_num,batch_size=256)

    #generate description/name-view interaction feature
    entity_pairs_3,timeViewInterF = timeView_interaction_F_gene(entity_pairs,time_ent_emb,
                                                    cuda_num=cuda_num,batch_size=256)

    for i in range(len(entity_pairs)):
        assert entity_pairs[i] == entity_pairs_1[i]
        assert entity_pairs[i] == entity_pairs_2[i]

    #save
    pickle.dump(neighViewInterF,open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH,"wb"))
    pickle.dump(desViewInterF,open(DESVIEW_SIMILARITY_FEATURE_PATH,"wb"))
    pickle.dump(structureViewInterF,open(STRUCTUREVIEW_SIMILARITY_FEATURE_PATH,"wb"))
    pickle.dump(timeViewInterF,open(TIMEVIEW_SIMILARITY_FEATURE_PATH,"wb"))
    print("save neighbor-view similarty feature in:",NEIGHBORVIEW_SIMILARITY_FEATURE_PATH)
    print("save name-view similarity Feature in:",DESVIEW_SIMILARITY_FEATURE_PATH)
    print("save structure-view similarity Feature in:",STRUCTUREVIEW_SIMILARITY_FEATURE_PATH)
    print("save time-view similarity Feature in:",TIMEVIEW_SIMILARITY_FEATURE_PATH)


if __name__ == '__main__':
    fixed(SEED_NUM)
    main()