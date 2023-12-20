import numpy as np
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp

def load_acm(prefix="./acm"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_p.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_a.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = [features_0, features_1, features_2]

    #标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    adj = torch.LongTensor(sp.load_npz("./acm/adjM.npz").toarray())
    PA = (adj[:4019, 4019:4019 + 7167] > 0) * 1
    PS = (adj[:4019, 4019 + 7167:] > 0) * 1
    NS = [PA, PS]

    PAP = scipy.sparse.load_npz(prefix + '/PAP.npz').toarray()
    norm_PAP = F.normalize(torch.from_numpy(PAP).type(torch.FloatTensor), dim=1, p=2)
    PSP = scipy.sparse.load_npz(prefix + '/PSP.npz').toarray()
    norm_PSP = F.normalize(torch.from_numpy(PSP).type(torch.FloatTensor), dim=1, p=2)
    ADJ=[norm_PSP,norm_PAP]
    return ADJ,NS,features,labels,num_classes,train_idx,val_idx,test_idx

def load_dblp(prefix="./dblp"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_A.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_P.npz').toarray()
    features_2 = sp.load_npz(prefix + '/features_2_T.npz').toarray()
    features_3 = np.eye(20)
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    features = [features_0, features_1, features_1,features_1]  #因为只有p是属性，所以只传P的特征即可

    # 标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 4

    adj = torch.LongTensor(sp.load_npz(prefix+"/adjM.npz").toarray())
    AP = (adj[:4057, 4057:4057 + 14328] > 0) * 1
    NS = [AP,AP]

    APA = scipy.sparse.load_npz(prefix + '/apa.npz').toarray()
    APTPA = scipy.sparse.load_npz(prefix + '/aptpa.npz').toarray()
    APCPA = scipy.sparse.load_npz(prefix + '/apcpa.npz').toarray()
    APAPA = np.load(prefix + '/apapa.npy')

    norm_APA = F.normalize(torch.from_numpy(APA).type(torch.FloatTensor), dim=1, p=2)
    norm_APTPA = F.normalize(torch.from_numpy(APTPA).type(torch.FloatTensor), dim=1, p=2)
    norm_APCPA = F.normalize(torch.from_numpy(APCPA).type(torch.FloatTensor), dim=1, p=2)
    norm_APAPA = F.normalize(torch.from_numpy(APAPA).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_APA,norm_APCPA]


    return ADJ, NS, features, labels, num_classes, train_idx, val_idx, test_idx


def load_imdb(prefix="./imdb"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_M.npz').toarray()
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_D.npz').toarray()
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_A.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features = [features_0, features_1, features_2]

    #标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    adj = torch.LongTensor(sp.load_npz("./acm/adjM.npz").toarray())
    MD = (adj[:4278, 4278:4278 + 2081] > 0) * 1
    MA = (adj[:4278, 4278 + 2081:] > 0) * 1
    NS = [MD, MA]

    MAM = sp.load_npz(prefix + '/MAM.npz').toarray()
    norm_MAM = F.normalize(torch.from_numpy(MAM).type(torch.FloatTensor), dim=1, p=2)
    MDM = scipy.sparse.load_npz(prefix + '/MDM.npz').toarray()
    norm_MDM = F.normalize(torch.from_numpy(MDM).type(torch.FloatTensor), dim=1, p=2)
    ADJ=[norm_MAM,norm_MDM]

    return ADJ, NS, features, labels, num_classes, train_idx, val_idx, test_idx

def load_Yelp(prefix=r"G:\OtherCode\__dataset\ACM_DBLP_IMDB\4_Yelp"):
    features_0 = scipy.sparse.load_npz(prefix + '/features_0_b.npz').toarray()  #B:2614
    features_1 = scipy.sparse.load_npz(prefix + '/features_1_u.npz').toarray()  #U:1286
    features_2 = scipy.sparse.load_npz(prefix + '/features_2_s.npz').toarray()  #S:4
    features_3 = scipy.sparse.load_npz(prefix + '/features_3_l.npz').toarray()  #L:9
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features_2 = torch.FloatTensor(features_2)
    features_3 = torch.FloatTensor(features_3)
    features = [features_0, features_1, features_1,features_1]

    #标签 训练集，验证集，测试集 分类数量
    labels = np.load(prefix + '/labels.npy')
    labels = torch.LongTensor(labels)
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npy', allow_pickle=True)
    train_idx = train_val_test_idx.item()['train_idx'].astype(int)
    val_idx = train_val_test_idx.item()['val_idx']
    test_idx = train_val_test_idx.item()['test_idx']

    train_idx = torch.from_numpy(train_idx).to(torch.int64)
    val_idx = torch.from_numpy(val_idx).to(torch.int64)
    test_idx = torch.from_numpy(test_idx).to(torch.int64)
    num_classes = 3

    BUB = sp.load_npz(prefix + '/adj_bub_one.npz').toarray()
    norm_BUB = F.normalize(torch.from_numpy(BUB).type(torch.FloatTensor), dim=1, p=2)
    BSB = scipy.sparse.load_npz(prefix + '/adj_bsb_one.npz').toarray()
    norm_BSB = F.normalize(torch.from_numpy(BSB).type(torch.FloatTensor), dim=1, p=2)
    BLB = scipy.sparse.load_npz(prefix + '/adj_blb_one.npz').toarray()
    norm_BLB = F.normalize(torch.from_numpy(BLB).type(torch.FloatTensor), dim=1, p=2)
    ADJ = [norm_BSB, norm_BUB]

    adj = torch.LongTensor(sp.load_npz(prefix+"/adjM.npz").toarray())
    BU = (adj[:2614, 2614:2614 + 1286] > 0) * 1
    BS = (adj[:2614, 2614 + 1286:2614 + 1286+4] > 0) * 1
    BL = (adj[:2614, 2614+1286+4:2614 + 1286+4+9] > 0) * 1
    NS = [BU, BS]



    return ADJ, NS, features, labels, num_classes, train_idx, val_idx, test_idx

if __name__=="__main__":
    load_Yelp()