import argparse
import torch
from tools import evaluate_results_nc, EarlyStopping
from load_data import load_acm,load_Yelp,load_imdb
import numpy as np
import random
from new_model import models
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
def main(args):
    ADJ,NS,features,labels,num_classes,train_idx,val_idx,test_idx = load_Yelp()
    in_dims = [feature.shape[1] for feature in features]
    features = [feature.to(args['device']) for feature in features]
    labels = labels.to(args['device'])
    adj_len = len(ADJ)
    ns_len = len(NS)
    model = models(in_dims,args["hidden_units"],num_classes,adj_len,args["feat_drop"],ns_len,args["sample_rate"])
    early_stopping = EarlyStopping(patience=args['patience'], verbose=True,
                                   save_path='checkpoint/checkpoint_{}.pt'.format('ACM'))
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    # if torch.cuda.is_available():
    #     print('Using CUDA')
    #     model.cuda()
    #     features = [feat.cuda() for feat in features]
    #     ADJ = [adj.cuda() for adj in ADJ]
    #     NS = [mp.cuda() for mp in NS]
    #     labels = labels.cuda()
    #     train_idx = [i.cuda() for i in train_idx]
    #     val_idx = [i.cuda() for i in val_idx]
    #     test_idx = [i.cuda() for i in test_idx]
    #     model.to("cuda")
    for epoch in range(args['num_epochs']):
        model.train()
        logits, h = model(features, ADJ, NS)
        loss = loss_fcn(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        logits, h = model(features, ADJ, NS)
        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        test_loss = loss_fcn(logits[test_idx], labels[test_idx])
        print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, loss.item(), val_loss.item(), test_loss.item()))
        early_stopping(val_loss.data.item(), model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break
    model.load_state_dict(torch.load('checkpoint/checkpoint_{}.pt'.format('ACM')))
    model.eval()
    logits, h = model(features, ADJ, NS)
    evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(), int(labels.max()) + 1)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--weight_decay', default=0.0006, help='权重衰减')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数')
    parser.add_argument('--feat_drop', default=0.53, help='特征丢弃率')
    parser.add_argument('--nei_num', default=2, help='邻居数量')
    parser.add_argument('--alpha', default=0.5, help='alpha')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--patience', type=int, default=10, help='耐心值')
    parser.add_argument('--seed', type=int, default=2, help='随机种子')
    parser.add_argument('--sample_rate', default=[5,5], help='属性节点数量')
    parser.add_argument('--device', type=str, default='cpu', help='使用cuda:0或者cpu')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)
