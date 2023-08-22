"""
    My Own Code
"""

import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from src.graph import utils
from src.graph.global_attack import BaseAttack
from torch_geometric.utils import dense_to_sparse, to_dense_adj, degree, to_undirected
from copy import deepcopy

def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
        """
        :param input: (batch, *)
        :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
        """
        # logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1), dim=1)
        logprobs = input
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        if reduction == 'none':
            return batchloss
        elif reduction == 'mean':
            return torch.mean(batchloss)
        elif reduction == 'sum':
            return torch.sum(batchloss)
        else:
            raise NotImplementedError('Unsupported reduction mode.')



class BaseMeta(BaseAttack):
    """Abstract base class for meta attack. Adversarial Attacks on Graph Neural
    Networks via Meta Learning, ICLR 2019,
    https://openreview.net/pdf?id=Bylnx209YX

    Parameters
    ----------
    model :
        model to attack. Default `None`.
    nnodes : int
        number of nodes in the input graph
    lambda_ : float
        lambda_ is used to weight the two objectives in Eq. (10) in the paper.
    feature_shape : tuple
        shape of the input node features
    attack_structure : bool
        whether to attack graph structure
    attack_features : bool
        whether to attack node features
    undirected : bool
        whether the graph is undirected
    device: str
        'cpu' or 'cuda'

    """

    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, undirected=True, device='cpu'):

        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):

        ## Meta Code
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        modified_adj = self.adj_changes + ori_adj
        
        ## PGD Code
        # if self.complementary is None:
        #     self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
        # m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        # tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        # m[tril_indices[0], tril_indices[1]] = self.adj_changes
        # m = m + m.t()
        # modified_adj = self.complementary * m + ori_adj

        ## My
        # if self.complementary is None:
        #     self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
        
        # tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        # m = torch.tril(self.adj_changes)
        # m = m + m.t()
        # modified_adj = self.complementary * m + ori_adj

        return modified_adj
        

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training

    def self_training_softlabel(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        softlabels_self_training = F.softmax(self.surrogate.output)
        onehot_labels = F.one_hot(labels).float()
        softlabels_self_training[idx_train] = onehot_labels[idx_train]
        return softlabels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.

        Note that different data type (float, double) can effect the final results.
        """
        t_d_min = torch.tensor(2.0).to(self.device)
        if self.undirected:
            t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        else:
            t_possible_edges = np.array((np.ones((self.nnodes, self.nnodes)) - np.eye(self.nnodes)).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff, undirected=self.undirected)
        return allowed_mask, current_ratio

    def get_adj_score(self, adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff):
        adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
        # Make sure that the minimum entry is 0.
        adj_meta_grad -= adj_meta_grad.min()
        # Filter self-loops
        adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
        # # Set entries to 0 that could lead to singleton nodes.
        singleton_mask = self.filter_potential_singletons(modified_adj)
        adj_meta_grad = adj_meta_grad *  singleton_mask

        if ll_constraint:
            allowed_mask, self.ll_ratio = self.log_likelihood_constraint(modified_adj, ori_adj, ll_cutoff)
            allowed_mask = allowed_mask.to(self.device)
            adj_meta_grad = adj_meta_grad * allowed_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad



class Metacon(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import Metattack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, 
    droprate1=0.5, droprate2=0.5, coef1=1.0, coef2=0.01,
    train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False):

        super(Metacon, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.momentum = momentum
        self.lr = lr

        self.droprate1 = droprate1
        self.droprate2 = droprate2
        self.coef1 = coef1
        self.coef2 = coef2

        self.train_iters = train_iters
        self.with_bias = with_bias
        self.analysis_mode= analysis_mode

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)


    def drop_edge_weighted(self, edge_index, edge_weights, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
        return edge_index[:, sel_mask]

    def drop_edge(self, edge_index, p):
        
        return self.drop_edge_weighted(edge_index, self.drop_weights, p=p, threshold=0.7)
    
    def degree_drop_weights(self, edge_index):
        edge_index_ = to_undirected(edge_index)
        deg = degree(edge_index_[1])
        deg_col = deg[edge_index[1]].to(torch.float32)
        s_col = torch.log(deg_col)
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
        return weights
        
    def inner_train(self, features, adj_norm, adj, idx_train, idx_unlabeled, labels):
        
        self._initialize()

        edge_index = dense_to_sparse(adj)[0]

        self.drop_weights = self.degree_drop_weights(edge_index).to(self.device)

        # edge_index1 = self.drop_edge(edge_index, self.droprate1)
        edge_index2 = self.drop_edge(edge_index, self.droprate2)

        # adj_norm1 = utils.normalize_adj_tensor(to_dense_adj(edge_index1, max_num_nodes=self.nnodes)[0])
        # adj_norm1 = adj_norm
        adj_norm2 = utils.normalize_adj_tensor(to_dense_adj(edge_index2, max_num_nodes=self.nnodes)[0])

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            hidden1 = features
            hidden2 = features
            for ix, w in enumerate(self.weights):
                
                b = self.biases[ix] if self.with_bias else 0

                if self.with_relu and ix != len(self.weights) - 1:
                    continue
                
                if self.sparse_features:
                    hidden1 = adj_norm @ torch.spmm(hidden1, w) + b
                    hidden2 = adj_norm2 @ torch.spmm(hidden2, w) + b
                else:
                    hidden1 = adj_norm @ hidden1 @ w + b
                    hidden2 = adj_norm2 @ hidden2 @ w + b

                

            output = F.log_softmax(hidden, dim=1)

            output1 = F.elu(hidden1)
            output2 = F.elu(hidden2)

            # loss_labeled = (F.nll_loss(output1[idx_train], labels[idx_train]) + F.nll_loss(output2[idx_train], labels[idx_train]))/2
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train]) 
            
            self.tau = 0.5
            f = lambda x: torch.exp(x / self.tau)

            # output1 = F.normalize(output1)
            # output2 = F.normalize(output2)
            
            refl = f(torch.mm(output1, output1.t()))
            between = f(torch.mm(output1, output2.t()))
            loss_unlabeled = -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()))

            # loss = self.coef1 * loss_labeled + self.coef2 * loss_unlabeled[idx_unlabeled].mean()
            loss = self.coef1 * loss_labeled + self.coef2 * loss_unlabeled[idx_unlabeled].mean()

            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            # weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    
    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

        return adj_grad, feature_grad


    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):

        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        attacked_idx = []
        self.analysis_idx = []
        self.analysis_inner_grad = []
        self.analysis_output = []


        self.set_grad_stats()

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)

            self.inner_train(modified_features, adj_norm, modified_adj, idx_train, idx_unlabeled, labels)

            

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


            with torch.no_grad():
                topk = 30
                adj_grad_flat = adj_grad.flatten().detach().cpu().numpy()
                topk_index = np.argsort(adj_grad_flat)[::-1][:topk]
                topk_grad = adj_grad_flat[topk_index]
                topk_index = np.array(np.unravel_index(topk_index, ori_adj.shape)).T

                topk_ll, topk_ul, topk_uu = [], [], []

                for j, (row, col) in enumerate(topk_index):
                    if (row in idx_train) & (col in idx_train):
                        topk_ll.append(topk_grad[j].item())
                    elif (row in idx_train) & (col not in idx_train):
                        topk_ul.append(topk_grad[j].item())
                    elif (row not in idx_train) & (col not in idx_train):
                        topk_uu.append(topk_grad[j].item())

                self.grad_stats['grad_topk_avg_ll'].append(np.mean(topk_ll))
                self.grad_stats['grad_topk_avg_ul'].append(np.mean(topk_ul))
                self.grad_stats['grad_topk_avg_uu'].append(np.mean(topk_uu))
                self.grad_stats['grad_topk_cnt_ll'].append(len(topk_ll))
                self.grad_stats['grad_topk_cnt_ul'].append(len(topk_ul))
                self.grad_stats['grad_topk_cnt_uu'].append(len(topk_uu))

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

        
            adj_meta_argmax = torch.argmax(adj_meta_score)
            row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            if self.undirected:
                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)


            if self.analysis_mode:
                attacked_idx.append([row_idx.item(), col_idx.item()])
                self.analysis(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                if i == 0:
                    attacked_idx = torch.tensor(attacked_idx)
                    # torch.save(attacked_idx, f'result/analysis/polblogs_Meta-Self_attack.pt')
                    exit()
                
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}



class MetaconPlus(BaseMeta):
    """Meta attack. Adversarial Attacks on Graph Neural Networks
    via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import Metattack
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, 
    droprate1=0.5, droprate2=0.5, coef1=1.0, coef2=0.01,
    train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False):

        super(MetaconPlus, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.momentum = momentum
        self.lr = lr

        self.droprate1 = droprate1
        self.droprate2 = droprate2
        self.coef1 = coef1
        self.coef2 = coef2

        self.train_iters = train_iters
        self.with_bias = with_bias
        self.analysis_mode= analysis_mode

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()

    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)


    def drop_edge_weighted(self, edge_index, edge_weights, p: float, threshold: float = 1.):
        edge_weights = edge_weights / edge_weights.mean() * p
        edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
        sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
        return edge_index[:, sel_mask]

    def drop_edge(self, edge_index, p):
        
        return self.drop_edge_weighted(edge_index, self.drop_weights, p=p, threshold=0.7)
    
    def degree_drop_weights(self, edge_index):
        edge_index_ = to_undirected(edge_index)
        deg = degree(edge_index_[1])
        deg_col = deg[edge_index[1]].to(torch.float32)
        s_col = torch.log(deg_col)
        weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
        return weights
        
    def inner_train(self, features, adj_norm, adj, idx_train, idx_unlabeled, labels):
        
        self._initialize()

        edge_index = dense_to_sparse(adj)[0]

        self.drop_weights = self.degree_drop_weights(edge_index).to(self.device)

        # edge_index1 = self.drop_edge(edge_index, self.droprate1)
        edge_index2 = self.drop_edge(edge_index, self.droprate2)

        # adj_norm1 = utils.normalize_adj_tensor(to_dense_adj(edge_index1, max_num_nodes=self.nnodes)[0])
        # adj_norm1 = adj_norm
        adj_norm2 = utils.normalize_adj_tensor(to_dense_adj(edge_index2, max_num_nodes=self.nnodes)[0])

        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            hidden1 = features
            hidden2 = features
            for ix, w in enumerate(self.weights):
                
                b = self.biases[ix] if self.with_bias else 0

                if self.with_relu and ix != len(self.weights) - 1:
                    continue
                
                if self.sparse_features:
                    hidden1 = adj_norm @ torch.spmm(hidden1, w) + b
                    hidden2 = adj_norm2 @ torch.spmm(hidden2, w) + b
                else:
                    hidden1 = adj_norm @ hidden1 @ w + b
                    hidden2 = adj_norm2 @ hidden2 @ w + b

                

            output = F.log_softmax(hidden, dim=1)

            output1 = F.elu(hidden1)
            output2 = F.elu(hidden2)

            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train]) 
            
            self.tau = 0.5
            f = lambda x: torch.exp(x / self.tau)

            refl = f(torch.mm(output1, output1.t()))
            between = f(torch.mm(output1, output2.t()))
            loss_unlabeled = -torch.log(between.diag() / (refl.sum(1) + between.sum(1) - refl.diag()))

            loss = self.coef1 * loss_labeled + self.coef2 * loss_unlabeled[idx_unlabeled].mean()

            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    
    def get_meta_grad(self, features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)

        output = F.log_softmax(hidden, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])

        p = torch.exp(output)
        loss_unlabeled = -p[idx_unlabeled, labels_self_training[idx_unlabeled]].mean()
        

        if self.lambda_ == 1:
            attack_loss = loss_labeled
        elif self.lambda_ == 0:
            attack_loss = loss_unlabeled
        else:
            attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

        return adj_grad, feature_grad


    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations, ll_constraint=True, ll_cutoff=0.004):

        """Generate n_perturbations on the input graph.

        Parameters
        ----------
        ori_features :
            Original (unperturbed) node feature matrix
        ori_adj :
            Original (unperturbed) adjacency matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_unlabeled:
            unlabeled nodes indices
        n_perturbations : int
            Number of perturbations on the input graph. Perturbations could
            be edge removals/additions or feature removals/additions.
        ll_constraint: bool
            whether to exert the likelihood ratio test constraint
        ll_cutoff : float
            The critical value for the likelihood ratio test of the power law distributions.
            See the Chi square distribution with one degree of freedom. Default value 0.004
            corresponds to a p-value of roughly 0.95. It would be ignored if `ll_constraint`
            is False.

        """

        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        # labels_self_training = self.self_training_softlabel(labels, idx_train)
        modified_adj = ori_adj
        modified_features = ori_features

        attacked_idx = []
        self.analysis_idx = []
        self.analysis_inner_grad = []
        self.analysis_output = []


        self.set_grad_stats()

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)

            self.inner_train(modified_features, adj_norm, modified_adj, idx_train, idx_unlabeled, labels)

            

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


            with torch.no_grad():
                topk = 30
                adj_grad_flat = adj_grad.flatten().detach().cpu().numpy()
                topk_index = np.argsort(adj_grad_flat)[::-1][:topk]
                topk_grad = adj_grad_flat[topk_index]
                topk_index = np.array(np.unravel_index(topk_index, ori_adj.shape)).T

                topk_ll, topk_ul, topk_uu = [], [], []

                for j, (row, col) in enumerate(topk_index):
                    if (row in idx_train) & (col in idx_train):
                        topk_ll.append(topk_grad[j].item())
                    elif (row in idx_train) & (col not in idx_train):
                        topk_ul.append(topk_grad[j].item())
                    elif (row not in idx_train) & (col not in idx_train):
                        topk_uu.append(topk_grad[j].item())

                self.grad_stats['grad_topk_avg_ll'].append(np.mean(topk_ll))
                self.grad_stats['grad_topk_avg_ul'].append(np.mean(topk_ul))
                self.grad_stats['grad_topk_avg_uu'].append(np.mean(topk_uu))
                self.grad_stats['grad_topk_cnt_ll'].append(len(topk_ll))
                self.grad_stats['grad_topk_cnt_ul'].append(len(topk_ul))
                self.grad_stats['grad_topk_cnt_uu'].append(len(topk_uu))

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

        
            adj_meta_argmax = torch.argmax(adj_meta_score)
            row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            if self.undirected:
                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)


            if self.analysis_mode:
                attacked_idx.append([row_idx.item(), col_idx.item()])
                self.analysis(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                if i == 0:
                    attacked_idx = torch.tensor(attacked_idx)
                    # torch.save(attacked_idx, f'result/analysis/polblogs_Meta-Self_attack.pt')
                    exit()
                
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}



