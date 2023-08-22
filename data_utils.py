import networkx as nx
import numpy as np
from scipy import sparse
from torch_geometric.utils import to_undirected, to_dense_adj
import os

def mk_dir(path):

    path_list = path.split('/')

    for i in range(len(path_list)):
        _path = os.path.join(*path_list[:i+1])

        if os.path.isdir(_path):
            pass
        else:
            os.mkdir(_path)

    return None

def subgraph_preprocess(adj, features, labels, idx_train, idx_val, idx_test):
    sample_ratio = 0.2

    g = nx.Graph(adj)
    sub_g_nodes = np.random.choice(g.nodes, round(sample_ratio*len(g.nodes)), replace=False)
    sub_g = g.subgraph(sub_g_nodes.tolist())
    adj = nx.to_numpy_array(sub_g)
    adj = sparse.csr_matrix(adj)
    features = features[sub_g, :]
    labels = labels[sub_g]

    idx_train = list(set(idx_train) & set(sub_g_nodes))
    idx_val = list(set(idx_val) & set(sub_g_nodes))
    idx_test = list(set(idx_test) & set(sub_g_nodes))

    idx = idx_train + idx_val + idx_test
    idx.sort()
    idx2sidx = {id:i for i, id in enumerate(idx)}

    idx_train = np.array([idx2sidx[id] for id in idx_train])
    idx_val = np.array([idx2sidx[id] for id in idx_val])
    idx_test = np.array([idx2sidx[id] for id in idx_test])

    idx_unlabeled = np.union1d(idx_val, idx_test)
    
    return adj, features, labels, idx_train, idx_val, idx_test, idx_unlabeled