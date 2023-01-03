from model_train_test_func import *
from Param import *
import numpy as np
from utils import *

def main():
    print("----------------interaction model--------------------")
    cuda_num = CUDA_NUM
    print("GPU num {}".format(cuda_num))
    #print("ko~ko~da~yo~")

    # read other data from bert unit model(train ill/test ill/eid2data)
    # (These files were saved during the training of basic bert unit)

    file_path = FILE_PATH
    train_ill, test_ill = load_aligned_pair(file_path, ratio=TRAIN_RATIO)
    train_ill = train_ill.tolist()
    test_ill = test_ill.tolist()
    train_ill = [tuple(ent_pairs_now) for ent_pairs_now in train_ill]
    test_ill = [tuple(ent_pairs_now) for ent_pairs_now in test_ill]

    print("train_ill num: {} /test_ill num:{} / train_ill & test_ill num: {}".format(len(train_ill),len(test_ill), len(set(train_ill) & set(test_ill) )))


    #(candidate) entity pairs
    entity_pairs = pickle.load(open(ENT_PAIRS_PATH, "rb"))

    #interaction features
    nei_features = pickle.load(open(NEIGHBORVIEW_SIMILARITY_FEATURE_PATH, "rb")) #neighbor-view interaction similarity feature
    # att_features = pickle.load(open(ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH,'rb')) #attribute-view interaction similarity feature
    des_features = pickle.load(open(DESVIEW_SIMILARITY_FEATURE_PATH, "rb")) #description/name-view interaction similarity feature
    structure_features = pickle.load(open(STRUCTUREVIEW_SIMILARITY_FEATURE_PATH, "rb"))  # description/name-view interaction similarity feature
    time_features = pickle.load(open(TIMEVIEW_SIMILARITY_FEATURE_PATH, "rb"))

    train_candidate = pickle.load(open(TRAIN_CANDIDATES_PATH, "rb"))
    test_candidate = pickle.load(open(TEST_CANDIDATES_PATH, "rb"))
    all_features = [] #[nei-view cat att-view cat des/name-view]
    for i in range(len(entity_pairs)):
        all_features.append(nei_features[i] + des_features[i] + structure_features[i] + time_features[i])# 42 concat 1.
        # all_features.append(nei_features[i])# 42 concat 1.
    print("All features embedding shape: ", np.array(all_features).shape)


    entpair2f_idx = {entpair: feature_idx for feature_idx, entpair in enumerate(entity_pairs)}
    Train_gene = Train_index_generator(train_ill, train_candidate, entpair2f_idx,neg_num=NEG_NUM, batch_size=BATCH_SIZE)
    # Model = MlP(42 ,11).cuda(cuda_num)
    Model = MlP(42 + 1 + 1 + 1, 11).cuda(cuda_num)
    Optimizer = optim.Adam(Model.parameters(), lr=LEARNING_RATE)
    Criterion = nn.MarginRankingLoss(margin=MARGIN, size_average=True)

    #train
    train(Model, Optimizer, Criterion, Train_gene, all_features, test_candidate, test_ill,
          entpair2f_idx, epoch_num=EPOCH_NUM, eval_num=10, cuda_num=cuda_num, test_topk=CANDIDATE_NUM)

    #save
    torch.save(Model, open(INTERACTION_MODEL_SAVE_PATH, "wb"))


if __name__ == '__main__':
    fixed(SEED_NUM)
    main()