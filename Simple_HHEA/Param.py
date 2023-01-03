"""
hyper-parameters:
"""
CUDA_NUM = 3 #GPU num
LANG = 'icews_yago' #'icews_wiki' #'dbp15k/fr_en' #language 'zh'/'ja'/'fr' dbp15k/fr_en
# ENTITY_NEIGH_MAX_NUM = 100 # max sampling neighbor num of entity
ENTITY_NEIGH_MAX_NUM = 128 # max sampling neighbor num of entity
ENTITY_ATTVALUE_MAX_NUM = 50 #max sampling attributeValue num of entity
KERNEL_NUM = 21
SEED_NUM = 11037
NOISE_RATIO = 0
# SEED_NUM = 66666
# CANDIDATE_NUM = 50 # candidate number
CANDIDATE_NUM = 50 # candidate number

BATCH_SIZE = 128 # train batch size
NEG_NUM = 5 # negative sampling num
# NEG_NUM = 5 # negative sampling num
LEARNING_RATE = 5e-4 # learning rate
MARGIN = 1 # margin
EPOCH_NUM = 400 # train epoch num
FILE_PATH = "../data/{}/".format(LANG) #interaction model save path.
TRAIN_RATIO = 0.3

INTERACTION_MODEL_SAVE_PATH = "../Save_model/interaction_model_{}.bin".format(LANG) #interaction model save path.

#load model(base_bert_unit_model) path
BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/"
BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "{}".format(LANG)
LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM = 4
BASIC_BERT_UNIT_MODEL_OUTPUT_DIM = 300

#load data path
DATA_PATH = r"../data/{}/".format(LANG)


#candidata_save_path
TRAIN_CANDIDATES_PATH = DATA_PATH + 'train_candidates.pkl'
TEST_CANDIDATES_PATH = DATA_PATH + 'test_candidates.pkl'

#entity embedding and attributeValue embedding save path.
ENT_EMB_PATH = DATA_PATH + '{}_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
STRUCTURE_EMB_PATH = DATA_PATH + '{}_structure_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
TIME_EMB_PATH = DATA_PATH + '{}_time_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
REL_EMB_PATH = DATA_PATH + '{}_rel_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)


ATTRIBUTEVALUE_EMB_PATH = DATA_PATH + 'attributeValue_embedding.pkl'
ATTRIBUTEVALUE_LIST_PATH = DATA_PATH + 'attributeValue_list.pkl' #1-1 match to attributeValue embedding.

#(candidate) entity_pairs save path.
ENT_PAIRS_PATH = DATA_PATH + 'ent_pairs.pkl' #[(e1,ea),(e1,eb)...]

#interaction feature save filepath name
NEIGHBORVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'neighbor_view_similarity_feature.pkl' #1-1 match to entity_pairs
ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'attribute_similarity_feature.pkl' #1-1 match to entity_pairs
DESVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'des_view_similarity_feature.pkl' #1-1 match to entity_pairs
STRUCTUREVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'structure_view_similarity_feature.pkl' #1-1 match to entity_pairs
TIMEVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + 'time_view_similarity_feature.pkl' #1-1 match to entity_pairs
REL_SIMILARITY_FEATURE_PATH = DATA_PATH + 'rel_view_similarity_feature.pkl' #1-1 match to entity_pairs
