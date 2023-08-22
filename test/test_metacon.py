import sys
sys.path.append('./')

import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from src.graph.defense import GCN
from src.graph.global_attack import Metacon, MetaconPlus
from src.graph.utils import *
from src.graph.data import Dataset
import argparse
from torch_geometric.utils import dense_to_sparse, subgraph
import networkx as nx
from data_utils import *

def analysis_attack(modified_adj, clean_adj, idx_train):

    atk_adj = modified_adj.detach().cpu() - clean_adj
    atk_adj_sp = dense_to_sparse(atk_adj)[0]
    atk_src = atk_adj_sp[0, :]
    atk_dst = atk_adj_sp[1, :]

    n_ll = (torch.isin(atk_src, torch.tensor(idx_train)) & torch.isin(atk_dst, torch.tensor(idx_train))).sum().item()
    n_lu = (torch.isin(atk_src, torch.tensor(idx_train)) & ~torch.isin(atk_dst, torch.tensor(idx_train))).sum().item() * 2
    n_uu = (~torch.isin(atk_src, torch.tensor(idx_train)) & ~torch.isin(atk_dst, torch.tensor(idx_train))).sum().item()

    # assert len(atk_src)/2 == n_ll + n_lu + n_uu

    print("# of attack for LABELED & LAELED NODES:{:d},\n# of attack for LABELED & UNLAELED NODES:{:d},\n# of attack for UNLABELED & UNLAELED NODES:{:d}".format(n_ll, n_lu, n_uu))

    clean_adj_sp = dense_to_sparse(clean_adj)[0]
    clean_src = clean_adj_sp[0, :]
    clean_dst = clean_adj_sp[1, :]

    prop_ll = (torch.isin(clean_src, torch.tensor(idx_train)) & torch.isin(clean_dst, torch.tensor(idx_train))).sum().item() 
    prop_lu = (torch.isin(clean_src, torch.tensor(idx_train)) & ~torch.isin(clean_dst, torch.tensor(idx_train))).sum().item() * 2
    prop_uu = (~torch.isin(clean_src, torch.tensor(idx_train)) & ~torch.isin(clean_dst, torch.tensor(idx_train))).sum().item() 

    prop_ll = n_ll / prop_ll * 100
    prop_lu = n_lu / prop_lu * 100
    prop_uu = n_uu / prop_uu * 100

    print("Proportion of attack for LABELED & LAELED NODES:{:3f},\nLABELED & UNLAELED NODES:{:3f},\nUNLABELED & UNLAELED NODES: {:3f}".format(prop_ll, prop_lu, prop_uu))

    return (n_ll, n_lu, n_uu), (prop_ll, prop_lu, prop_uu)



parser = argparse.ArgumentParser()

## Basic Setting
parser.add_argument('--model', type=str, default='metacon+')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

parser.add_argument('--dataset', type=str, default='cora', \
    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.001,  help='pertubation rate')
parser.add_argument('--hids', nargs='+', type=int, default=[16])
parser.add_argument('--split', nargs='+', type=float, default=[0.1, 0.1, 0.8])


## Metattack Parameters
parser.add_argument('--inner_train_iters', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')

## ContraMettack Parameters                    
parser.add_argument('--droprate1', type=float, default=0.,
                    help='Initial learning rate.')
parser.add_argument('--droprate2', type=float, default=0.3,
                    help='Initial learning rate.')
parser.add_argument('--coef1', type=float, default=1.0,
                    help='Initial learning rate.')
parser.add_argument('--coef2', type=float, default=0.2,
                    help='Initial learning rate.')

parser.add_argument('--savepath', type=str, default="result")
parser.add_argument('--savegraph', type=str, default=False)


## No use
parser.add_argument('--analysis_mode', type=bool, default=False)


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)


data = Dataset('./datasets/', name=args.dataset, setting='nettack', seed=args.seed, split=args.split)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
perturbations = int(args.ptb_rate * (adj.sum()//2))
adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False)
nclass = max(labels).item() + 1

nhids = [features.shape[1]] + args.hids + [nclass]
nlayer = len(args.hids) + 1




# Setup Surrogate Model
surrogate = GCN(nhids=nhids, dropout=0.5, with_relu=False, with_bias=True, weight_decay=5e-4, device=device)
surrogate = surrogate.to(device)
surrogate.fit(features, adj, labels, idx_train, train_iters=1000)


# Setup Attack Loss 
lambda_ = 0

# Setup Attack Model
if (args.model == 'metacon'):
    model = Metacon(model=surrogate, 
    nnodes=adj.shape[0], feature_shape=features.shape, 
    attack_structure=True, attack_features=False, device=device, lambda_=lambda_, analysis_mode=args.analysis_mode, 
    train_iters=args.inner_train_iters, lr=args.lr, momentum=args.momentum,
    droprate1=args.droprate1, droprate2=args.droprate2, coef1=args.coef1, coef2=args.coef2)

if (args.model == 'metacon+'):
    model = MetaconPlus(model=surrogate, 
    nnodes=adj.shape[0], feature_shape=features.shape, 
    attack_structure=True, attack_features=False, device=device, lambda_=lambda_, analysis_mode=args.analysis_mode, 
    train_iters=args.inner_train_iters, lr=args.lr, momentum=args.momentum,
    droprate1=args.droprate1, droprate2=args.droprate2, coef1=args.coef1, coef2=args.coef2)



model = model.to(device)



def test(new_adj, gcn=None, swap=False):
    ''' test on GCN '''
    if gcn is None:
        # adj = normalize_adj_tensor(adj)
        gcn = GCN(nhids=nhids,
                  dropout=0.5, device=device)
        gcn = gcn.to(device)
        # gcn.fit(features, new_adj, labels, idx_train) # train without model picking
        
        gcn.fit(features, new_adj, labels, idx_train, idx_val, train_iter=1000, patience=200) # train with validation model picking
        gcn.eval()
        output = gcn.predict().cpu()
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()
        
    else:
        gcn.eval()
        output = gcn.predict(features.to(device), new_adj.to(device)).cpu()

        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()



def main():

    _adj = adj
    _perturbations = perturbations

    model.attack(features, _adj, labels, idx_train, idx_unlabeled, _perturbations, ll_constraint=False)

    target_gcn = GCN(nhids=nhids, dropout=0.5, device=device, lr=0.01)

    target_gcn = target_gcn.to(device)
    target_gcn.fit(features, adj, labels, idx_train, idx_val, train_iters=1000, patience=30)

    print("#########################  GCN Result  ##############################")
    print('=== testing GCN on clean graph ===')
    acc = test(adj, target_gcn)
    print('=== testing GCN on Evasion attack ===')
    modified_adj = model.modified_adj
    # modified_features = model.modified_features
    acc_eva = test(modified_adj, target_gcn)
    # modified_features = model.modified_features
    print('=== testing GCN on Poisoning attack ===')
    acc_poison = test(modified_adj)
    
    n_attack, prop_attack = analysis_attack(modified_adj, adj, idx_train)

    result = f'Clean ACC:{acc:.4f} Evasive ACC:{acc_eva:.4f} Poison ACC:{acc_poison:.4f} \n' \
    + f'Num Attack LL:{n_attack[0]} LU:{n_attack[1]} UU:{n_attack[2]} Prop. Attack LL:{prop_attack[0]:.3f} LU:{prop_attack[1]:.3f}, UU:{prop_attack[2]:.3f}\n'

    savepath = f'{args.savepath}/summary_result/{args.dataset}_ptb({args.ptb_rate:.2f})_split({args.split[0]})'

    mk_dir(savepath)
    with open(savepath + f'/{args.model}_victim(gcn).txt', 'a') as f:
        f.write(f'{args}\n'+result)

    if args.savegraph=='True':
        _ptb = round(args.ptb_rate, 2)
        mk_dir(f'./{args.savepath}/generated/{args.dataset}_{_ptb:.2f}_split({args.split[0]})')
        torch.save(modified_adj.cpu(), 
        f'./{args.savepath}/generated/{args.dataset}_{_ptb:.2f}_split({args.split[0]})/{args.model}_seed({args.seed}).pt')
        


    
if __name__ == '__main__':
    main()

