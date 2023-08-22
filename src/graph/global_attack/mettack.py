"""
    Adversarial Attacks on Graph Neural Networks via Meta Learning. ICLR 2019
        https://openreview.net/pdf?id=Bylnx209YX
    Author Tensorflow implementation:
        https://github.com/danielzuegner/gnn-meta-attack
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
from copy import deepcopy


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


class Metattack(BaseMeta):
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

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False):

        super(Metattack, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        
        self._initialize()

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

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            # weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    


    def analysis_dfdA(self, features, adj, idx_train, idx_unlabeled, labels, iterate):

        analysis_idx = []
        analysis_output = dict()

        adj_norm = utils.normalize_adj_tensor(adj)

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b

            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)
        
        ## original output
        output = F.log_softmax(hidden, dim=1)
        
        modified_features = features

        for i in tqdm(self.sampled_idx):
            for j in self.sampled_idx:

                if i==j:
                    continue

                _adj = adj.detach().clone()

                ## Flip edge (i,j)
                if _adj[i][j] == 1:
                    _adj[i][j] = 0
                else:
                    _adj[i][j] = 1

                _adj_norm = utils.normalize_adj_tensor(_adj)

                ## forward loop of GCN
                hidden = features
                for ix, w in enumerate(self.weights):
                    b = self.biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = _adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = _adj_norm @ hidden @ w + b

                    if self.with_relu and ix != len(self.weights) - 1:
                        hidden = F.relu(hidden)
                
                # Changed output
                _output = F.log_softmax(hidden, dim=1)
                
                
                analysis_output[str(i) + '_' + str(j)] = (torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())

                del _output

        
        torch.save(analysis_output, f'figures/analysis_citeseer_Meta-Self_output8.pt')
            
            
    def analysis_inner(self, features, adj, idx_train, idx_unlabeled, labels, iterate):

        grad_analysis = {}
        for i in self.sampled_idx:
            for j in self.sampled_idx:
                if i != j:
                    grad_analysis[str(i)+'_'+str(j)] = []

        # self._initialize()

        adj_norm = utils.normalize_adj_tensor(adj)

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

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            # weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

            ## Changed input

            

            for ii in tqdm(self.sampled_idx):
                for jj in self.sampled_idx:

                    if ii==jj:
                        continue

                    _adj = adj.detach().clone()

                    ## Flip edge (i,j)
                    if _adj[ii][jj] == 1:
                        _adj[ii][jj] = 0
                    else:
                        _adj[ii][jj] = 1

                    _adj_norm = utils.normalize_adj_tensor(_adj)
                    _hidden = features.detach().clone()
        
                    for ix, w in enumerate(self.weights):
                        b = self.biases[ix] if self.with_bias else 0
                        _hidden = _adj_norm @ _hidden @ w + b
                        _output = F.log_softmax(_hidden, dim=1)
                        _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

                    _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)

                    grads_norm = (torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu()
                    grad_analysis[str(ii)+'_'+str(jj)].append(grads_norm.item())

            torch.save(grad_analysis, f'figures/analysis_citeseer_Meta-Self_{self.train_iters}_inner8.pt')
            if j == 9:
                print("Inner Train Analysis", j)
                exit()
            
    


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

            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            if self.analysis_mode:
                ratio1 = 0.005
                ratio2 = 0.005
                nnodes = len(idx_train) + len(idx_unlabeled)
                sampled_idx_train = idx_train
                sampled_idx_train = idx_train[torch.randperm(len(idx_train))][:int(ratio1*nnodes)]
                sampled_idx_unlabeled = idx_unlabeled[torch.randperm(len(idx_unlabeled))][:int(ratio2*nnodes)]
                self.sampled_idx = np.concatenate([sampled_idx_train, sampled_idx_unlabeled], -1)
                torch.save({'analysis_idx': torch.tensor(self.sampled_idx)}, f'figures/analysis_citeseer_Meta-Self_sample_idx8.pt')

                ## df/dA: output change
                self.analysis_dfdA(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                self.analysis_inner(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                exit()

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


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

                
            
                

            # else:
            #     feature_meta_argmax = torch.argmax(feature_meta_score)
            #     row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
            #     self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}



class MetaApprox(BaseMeta):
    """Approximated version of Meta Attack. Adversarial Attacks on
    Graph Neural Networks via Meta Learning, ICLR 2019.

    Examples
    --------

    >>> import numpy as np
    >>> from deeprobust.graph.data import Dataset
    >>> from deeprobust.graph.defense import GCN
    >>> from deeprobust.graph.global_attack import MetaApprox
    >>> from deeprobust.graph.utils import preprocess
    >>> data = Dataset(root='/tmp/', name='cora')
    >>> adj, features, labels = data.adj, data.features, data.labels
    >>> adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False) # conver to tensor
    >>> idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
    >>> idx_unlabeled = np.union1d(idx_val, idx_test)
    >>> # Setup Surrogate model
    >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
    >>> # Setup Attack Model
    >>> model = MetaApprox(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
            attack_structure=True, attack_features=False, device='cpu', lambda_=0).to('cpu')
    >>> # Attack
    >>> model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=True)
    >>> modified_adj = model.modified_adj

    """

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.01):

        super(MetaApprox, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.lr = lr
        self.train_iters = train_iters
        self.adj_meta_grad = None
        self.features_meta_grad = None
        if self.attack_structure:
            self.adj_grad_sum = torch.zeros(nnodes, nnodes).to(device)
        if self.attack_features:
            self.feature_grad_sum = torch.zeros(feature_shape).to(device)

        self.with_bias = with_bias

        self.weights = []
        self.biases = []

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
        self.weights.append(output_weight)
        self.biases.append(output_bias)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=lr) # , weight_decay=5e-4)
        self._initialize()

    def _initialize(self):
        for w, b in zip(self.weights, self.biases):
            # w.data.fill_(1)
            # b.data.fill_(1)
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

        self.optimizer = optim.Adam(self.weights + self.biases, lr=self.lr)

    def inner_train(self, features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training):
        adj_norm = utils.normalize_adj_tensor(modified_adj)
        for j in range(self.train_iters):
            # hidden = features
            # for w, b in zip(self.weights, self.biases):
            #     if self.sparse_features:
            #         hidden = adj_norm @ torch.spmm(hidden, w) + b
            #     else:
            #         hidden = adj_norm @ hidden @ w + b
            #     if self.with_relu:
            #         hidden = F.relu(hidden)

            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
            loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])

            if self.lambda_ == 1:
                attack_loss = loss_labeled
            elif self.lambda_ == 0:
                attack_loss = loss_unlabeled
            else:
                attack_loss = self.lambda_ * loss_labeled + (1 - self.lambda_) * loss_unlabeled

            self.optimizer.zero_grad()
            loss_labeled.backward(retain_graph=True)

            if self.attack_structure:
                self.adj_changes.grad.zero_()
                self.adj_grad_sum += torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
            if self.attack_features:
                self.feature_changes.grad.zero_()
                self.feature_grad_sum += torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]

            self.optimizer.step()


        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])
        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))


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
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        self.sparse_features = sp.issparse(ori_features)
        modified_adj = ori_adj
        modified_features = ori_features

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            self._initialize()

            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
                self.adj_grad_sum.data.fill_(0)
            if self.attack_features:
                modified_features = ori_features + self.feature_changes
                self.feature_grad_sum.data.fill_(0)

            self.inner_train(modified_features, modified_adj, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)

            if self.attack_structure:
                adj_meta_score = self.get_adj_score(self.adj_grad_sum, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(self.feature_grad_sum, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()





class MetattackNoinner(BaseMeta):
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

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', 
                with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False):

        super(MetattackNoinner, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias
        self.analysis_mode=analysis_mode

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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()

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

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            # weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=False)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                # bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=False)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]


    def analysis(self, features, adj, idx_train, idx_unlabeled, labels, iterate):

        adj_norm = utils.normalize_adj_tensor(adj)

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

        weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=False)
        self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
        if self.with_bias:
            bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=False)
            self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]


        # analysis_output = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_inner_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))

        analysis_idx = []
        analysis_output = []
        analysis_grad = []
        analysis_inner_grad = []

        modified_features = features
        adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, self.self_training_label(labels, idx_train))

        ratio = 0.05
        # sampled_idx_train = idx_train[torch.randperm(idx_train.shape[0])][:int(ratio*idx_train.shape[0])]
        sampled_idx_train = idx_train
        sampled_idx_unlabeled = idx_unlabeled[torch.randperm(idx_unlabeled.shape[0])][:int(ratio*idx_unlabeled.shape[0])]

        sampled_idx = np.concatenate([sampled_idx_train, sampled_idx_unlabeled], -1)


        for i in tqdm(sampled_idx):
            for j in sampled_idx:

                if i==j:
                    continue

                _adj = adj.detach().clone()

                if _adj[i][j] == 1:
                    _adj[i][j] = 0
                else:
                    _adj[i][j] = 1

                _adj_norm = utils.normalize_adj_tensor(_adj)

                ## forward loop of GCN
                hidden = features
                for ix, w in enumerate(self.weights):
                    b = self.biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = _adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = _adj_norm @ hidden @ w + b

                    if self.with_relu and ix != len(self.weights) - 1:
                        hidden = F.relu(hidden)

                _output = F.log_softmax(hidden, dim=1)
                _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

                _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
                if self.with_bias:
                    _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

                # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
                # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()


                analysis_idx.append([i, j])
                analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
                analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())

                del _output
                del _loss_labeled
                del _weight_grads
                

        
        attacked_idx = torch.load('result/analysis/cora_Meta-Self-Noinner_attack.pt')

        row = list(attacked_idx[:, 0])
        col = list(attacked_idx[:, 1])

        for i, j in zip(row, col):

            if (i in sampled_idx) & (j in sampled_idx):
                continue

            _adj = adj.detach().clone()

            if _adj[i][j] == 1:
                _adj[i][j] = 0
            else:
                _adj[i][j] = 1

            _adj_norm = utils.normalize_adj_tensor(_adj)

            ## forward loop of GCN
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = _adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = _adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            _output = F.log_softmax(hidden, dim=1)
            _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

            _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
            if self.with_bias:
                _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

            # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
            # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()

            analysis_idx.append([i, j])
            analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
            analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())


        # for i in tqdm(sampled_idx):
        #     for j in sampled_idx:
        #         analysis_grad[iterate][i][j] = adj_grad[iterate][i][j]

        # self.analysis_idx.append(analysis_idx)
        # self.analysis_output.append(analysis_output)
        # self.analysis_inner_grad.append(analysis_inner_grad)
        
        if iterate == 0:
            analysis_dict = {'analysis_idx': torch.tensor(analysis_idx),
                                'analysis_output': torch.tensor(analysis_output),
                                'analysis_inner_grad': torch.tensor(analysis_inner_grad)}

            torch.save(analysis_dict, f'result/analysis/cora_Meta-Self-Noinner_ana.pt')
            
            # torch.save({"analysis_output": analysis_output.detach().cpu(), 
            # "analysis_inner_grad": analysis_inner_grad.detach(), 
            # "analysis_grad": analysis_grad.detach().cpu()}, 
            # f'result/analysis/cora_tensors.pt')
            exit()

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

        self.set_grad_stats()
        
        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            
            if i==0:
                self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


            # with torch.no_grad():
            #     topk = 30
            #     adj_grad_flat = adj_grad.flatten().detach().cpu().numpy()
            #     topk_index = np.argsort(adj_grad_flat)[::-1][:topk]
            #     topk_grad = adj_grad_flat[topk_index]
            #     topk_index = np.array(np.unravel_index(topk_index, ori_adj.shape)).T

            #     topk_ll, topk_ul, topk_uu = [], [], []

            #     for i, (row, col) in enumerate(topk_index):
            #         if (row in idx_train) & (col in idx_train):
            #             topk_ll.append(topk_grad[i].item())
            #         elif (row in idx_train) & (col not in idx_train):
            #             topk_ul.append(topk_grad[i].item())
            #         elif (row not in idx_train) & (col not in idx_train):
            #             topk_uu.append(topk_grad[i].item())

            #     self.grad_stats['grad_topk_avg_ll'].append(np.mean(topk_ll))
            #     self.grad_stats['grad_topk_avg_ul'].append(np.mean(topk_ul))
            #     self.grad_stats['grad_topk_avg_uu'].append(np.mean(topk_uu))
            #     self.grad_stats['grad_topk_cnt_ll'].append(len(topk_ll))
            #     self.grad_stats['grad_topk_cnt_ul'].append(len(topk_ul))
            #     self.grad_stats['grad_topk_cnt_uu'].append(len(topk_uu))

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)


                if self.analysis_mode:
                    attacked_idx.append([row_idx.item(), col_idx.item()])
                    self.analysis(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                    if i == 50:
                        # attacked_idx = torch.tensor(attacked_idx)
                        # torch.save(attacked_idx, f'result/analysis/cora_Meta-Self-Noinner_attack.pt')
                        exit()
                
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}



class MetattackSelfTrain(BaseMeta):
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

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9):

        super(MetattackSelfTrain, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        self.stage = 20
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):

        # self.surrogate.initialize()

        self._initialize()
        self.idx_train_pseudo = idx_train
        self.labels_pseudo = labels.clone()

        for stage in range(self.stage):

            # self.surrogate.train()
            # self.surrogate.fit(features, adj_norm, labels, idx_train, initialize=False)
            # self.surrogate.eval()

            # Set requires_grad=True
            for ix in range(len(self.hidden_sizes) + 1):
                self.weights[ix] = self.weights[ix].detach()
                self.weights[ix].requires_grad = True
                self.w_velocities[ix] = self.w_velocities[ix].detach()
                self.w_velocities[ix].requires_grad = True

                if self.with_bias:
                    self.biases[ix] = self.biases[ix].detach()
                    # self.biases[ix].requires_grad = True
                    self.b_velocities[ix] = self.b_velocities[ix].detach()
                    # self.b_velocities[ix].requires_grad = True

            # pretrain-epoch step
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
                output = F.log_softmax(hidden, dim=1)
                loss_labeled = F.nll_loss(output[self.idx_train_pseudo], self.labels_pseudo[self.idx_train_pseudo])

                ## backward the surrogate model's parameter
                weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
                self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
                if self.with_bias:
                    bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                    self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]
                self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
                if self.with_bias:
                    self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]


            class_prob = F.softmax(output, dim=1) # prediction -> class_prob
                
            pred = torch.argmax(class_prob, dim=1)
            confidence = torch.max(class_prob, dim=1)[0]
            sorted_index = torch.argsort(-confidence) # sort by confidence

            n_class = len(labels.unique())
            self_train_rate = 0.05
            threshold = int(len(labels) / n_class * self_train_rate)

            ## Obtain Pseudo Train index/mask
            idx_train_pseudo = []
            # pseudo_train_mask = torch.zeros_like(data.train_mask).bool().to(self.device)
            count_pseudo_class = [0 for i in range(n_class)]
            
            for idx in sorted_index:
                _pred = pred[idx]
                if (count_pseudo_class[_pred] < threshold) & (idx.item() not in self.idx_train_pseudo):
                    idx_train_pseudo.append(idx.item())
                    count_pseudo_class[_pred] += 1

            # pseudo_train_mask[pseudo_train_index] = True
            self.idx_train_pseudo = np.concatenate([self.idx_train_pseudo, np.array(idx_train_pseudo)])

            ## Obtain Pseudo Labels
            # pseudo_labels = F.one_hot(labels).float()

            labels_pseudo = self.labels_pseudo.clone()
            labels_pseudo[idx_train_pseudo] = pred[idx_train_pseudo]
            self.labels_pseudo = deepcopy(labels_pseudo)
        
        ## last training
        # Initial parameters of meta model
        self._initialize()

        # Set requires_grad=True
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

        # pretrain-epoch step
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
            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[self.idx_train_pseudo], self.labels_pseudo[self.idx_train_pseudo])

            ## backward the surrogate model's parameter
            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
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

        self.set_grad_stats()

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)

            if self.attack_features:
                modified_features = ori_features + self.feature_changes

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)


            with torch.no_grad():
                topk = 30
                adj_grad_flat = adj_grad.flatten().detach().cpu().numpy()
                topk_index = np.argsort(adj_grad_flat)[::-1][:topk]
                topk_grad = adj_grad_flat[topk_index]
                topk_index = np.array(np.unravel_index(topk_index, ori_adj.shape)).T

                topk_ll, topk_ul, topk_uu = [], [], []

                for i, (row, col) in enumerate(topk_index):
                    if (row in idx_train) & (col in idx_train):
                        topk_ll.append(topk_grad[i].item())
                    elif (row in idx_train) & (col not in idx_train):
                        topk_ul.append(topk_grad[i].item())
                    elif (row not in idx_train) & (col not in idx_train):
                        topk_uu.append(topk_grad[i].item())

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

            if adj_meta_score.max() >= feature_meta_score.max():
                adj_meta_argmax = torch.argmax(adj_meta_score)
                row_idx, col_idx = utils.unravel_index(adj_meta_argmax, ori_adj.shape)
                self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                if self.undirected:
                    self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}





class Metattack_Reg(BaseMeta):
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

    def __init__(self, model, nnodes, y_adj, coef=0.01, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False):

        super(Metattack_Reg, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias
        self.analysis_mode= analysis_mode

        self.y_adj = torch.tensor(y_adj.todense()).to(device)
        self.coef = coef

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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        
        self._initialize()

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

            output = F.log_softmax(hidden, dim=1)

            estim_adj = F.sigmoid(torch.mm(hidden, hidden.t()))

            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight= torch.tensor(float(self.y_adj.shape[0] * self.y_adj.shape[0] - self.y_adj.sum()) / self.y_adj.sum())
            loss_adj = F.binary_cross_entropy_with_logits(estim_adj, self.y_adj, weight=weight)
            loss = loss_labeled + self.coef * loss_adj

            # weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    


    def analysis(self, features, adj, idx_train, idx_unlabeled, labels, iterate):

        adj_norm = utils.normalize_adj_tensor(adj)

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

        weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=False)
        self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
        if self.with_bias:
            bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=False)
            self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]


        # analysis_output = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_inner_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))

        analysis_idx = []
        analysis_output = []
        analysis_grad = []
        analysis_inner_grad = []

        modified_features = features
        adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, self.self_training_label(labels, idx_train))

        ratio1 = 0.05
        ratio2 = 0.05
        nnodes = len(idx_train) + len(idx_unlabeled)
        sampled_idx_train = idx_train
        sampled_idx_train = idx_train[torch.randperm(len(idx_train))][:int(ratio1*nnodes)]
        sampled_idx_unlabeled = idx_unlabeled[torch.randperm(len(idx_unlabeled))][:int(ratio2*nnodes)]

        sampled_idx = np.concatenate([sampled_idx_train, sampled_idx_unlabeled], -1)


        for i in tqdm(sampled_idx):
            for j in sampled_idx:

                if i==j:
                    continue

                _adj = adj.detach().clone()

                if _adj[i][j] == 1:
                    _adj[i][j] = 0
                else:
                    _adj[i][j] = 1

                _adj_norm = utils.normalize_adj_tensor(_adj)

                ## forward loop of GCN
                hidden = features
                for ix, w in enumerate(self.weights):
                    b = self.biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = _adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = _adj_norm @ hidden @ w + b

                    if self.with_relu and ix != len(self.weights) - 1:
                        hidden = F.relu(hidden)

                _output = F.log_softmax(hidden, dim=1)
                _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

                _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
                if self.with_bias:
                    _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

                # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
                # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()


                analysis_idx.append([i, j])
                analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
                analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())

                del _output
                del _loss_labeled
                del _weight_grads
                

        
        # attacked_idx = torch.load('result/analysis/polblogs_Meta-Self_attack.pt')

        # row = list(attacked_idx[:, 0])
        # col = list(attacked_idx[:, 1])

        # for i, j in zip(row, col):

        #     if (i in sampled_idx) & (j in sampled_idx):
        #         continue

        #     _adj = adj.detach().clone()

        #     if _adj[i][j] == 1:
        #         _adj[i][j] = 0
        #     else:
        #         _adj[i][j] = 1

        #     _adj_norm = utils.normalize_adj_tensor(_adj)

        #     ## forward loop of GCN
        #     hidden = features
        #     for ix, w in enumerate(self.weights):
        #         b = self.biases[ix] if self.with_bias else 0
        #         if self.sparse_features:
        #             hidden = _adj_norm @ torch.spmm(hidden, w) + b
        #         else:
        #             hidden = _adj_norm @ hidden @ w + b

        #         if self.with_relu and ix != len(self.weights) - 1:
        #             hidden = F.relu(hidden)

        #     _output = F.log_softmax(hidden, dim=1)
        #     _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

        #     _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
        #     if self.with_bias:
        #         _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

            # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
            # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()

            # analysis_idx.append([i, j])
            # analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
            # analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())


        # for i in tqdm(sampled_idx):
        #     for j in sampled_idx:
        #         analysis_grad[iterate][i][j] = adj_grad[iterate][i][j]

        # self.analysis_idx.append(analysis_idx)
        # self.analysis_output.append(analysis_output)
        # self.analysis_inner_grad.append(analysis_inner_grad)
        
        if iterate == 0:
            analysis_dict = {'analysis_idx': torch.tensor(analysis_idx),
                                'analysis_output': torch.tensor(analysis_output),
                                'analysis_inner_grad': torch.tensor(analysis_inner_grad)}

            torch.save(analysis_dict, f'result/analysis/cora_Meta-Self_{self.train_iters}_ana.pt')
            
            # torch.save({"analysis_output": analysis_output.detach().cpu(), 
            # "analysis_inner_grad": analysis_inner_grad.detach(), 
            # "analysis_grad": analysis_grad.detach().cpu()}, 
            # f'result/analysis/cora_tensors.pt')
            # exit()
            



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

            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

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

            # if True:
            #     if i == 0:
            #         self.debug=[]

            #     if (row_idx.item(), col_idx.item()) in self.debug:
            #         print("?")

            #     self.debug.append((row_idx.item(), col_idx.item()))

                



            if self.analysis_mode:
                attacked_idx.append([row_idx.item(), col_idx.item()])
                self.analysis(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                if i == 0:
                    attacked_idx = torch.tensor(attacked_idx)
                    # torch.save(attacked_idx, f'result/analysis/polblogs_Meta-Self_attack.pt')
                    exit()
                

            # else:
            #     feature_meta_argmax = torch.argmax(feature_meta_score)
            #     row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
            #     self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}







class MetaPGD(BaseMeta):
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

    def __init__(self, model, nnodes, y_adj, coef=0.01, initial_step=20.0, loss_type='CE', initial_graph = None,
                feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, 
                lambda_=0.5, train_iters=100, outer_iters=200, lr=0.1, momentum=0.9, analysis_mode=False):

        super(MetaPGD, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.initial_step = initial_step
        self.train_iters = train_iters
        self.outer_iters = outer_iters
        self.with_bias = with_bias
        self.analysis_mode= analysis_mode

        self.y_adj = torch.tensor(y_adj.todense()).to(device)
        self.coef = coef
        self.outer_iters = outer_iters
        self.loss_type='CE'
        self.complementary = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            # self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))

            if initial_graph:
                self.adj_changes = Parameter(torch.tril(torch.load(initial_graph) - torch.tensor(y_adj.todense())))
            else:
                self.adj_changes = Parameter(torch.zeros((nnodes, nnodes)).float())
                self.adj_changes.data.fill_(0)

            
                 

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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        
        self._initialize()

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

            output = F.log_softmax(hidden, dim=1)

            estim_adj = F.sigmoid(torch.mm(hidden, hidden.t()))

            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight= torch.tensor(float(self.y_adj.shape[0] * self.y_adj.shape[0] - self.y_adj.sum()) / self.y_adj.sum())
            loss_adj = F.binary_cross_entropy_with_logits(estim_adj, self.y_adj, weight=weight)
            loss = loss_labeled + self.coef * loss_adj

            # weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    


    def analysis(self, features, adj, idx_train, idx_unlabeled, labels, iterate):

        adj_norm = utils.normalize_adj_tensor(adj)

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

        weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=False)
        self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
        if self.with_bias:
            bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=False)
            self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]


        # analysis_output = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_inner_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))

        analysis_idx = []
        analysis_output = []
        analysis_grad = []
        analysis_inner_grad = []

        modified_features = features
        adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, self.self_training_label(labels, idx_train))

        ratio1 = 0.05
        ratio2 = 0.05
        nnodes = len(idx_train) + len(idx_unlabeled)
        sampled_idx_train = idx_train
        sampled_idx_train = idx_train[torch.randperm(len(idx_train))][:int(ratio1*nnodes)]
        sampled_idx_unlabeled = idx_unlabeled[torch.randperm(len(idx_unlabeled))][:int(ratio2*nnodes)]

        sampled_idx = np.concatenate([sampled_idx_train, sampled_idx_unlabeled], -1)


        for i in tqdm(sampled_idx):
            for j in sampled_idx:

                if i==j:
                    continue

                _adj = adj.detach().clone()

                if _adj[i][j] == 1:
                    _adj[i][j] = 0
                else:
                    _adj[i][j] = 1

                _adj_norm = utils.normalize_adj_tensor(_adj)

                ## forward loop of GCN
                hidden = features
                for ix, w in enumerate(self.weights):
                    b = self.biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = _adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = _adj_norm @ hidden @ w + b

                    if self.with_relu and ix != len(self.weights) - 1:
                        hidden = F.relu(hidden)

                _output = F.log_softmax(hidden, dim=1)
                _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

                _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
                if self.with_bias:
                    _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

                # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
                # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()


                analysis_idx.append([i, j])
                analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
                analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())

                del _output
                del _loss_labeled
                del _weight_grads
                

        
        # attacked_idx = torch.load('result/analysis/polblogs_Meta-Self_attack.pt')

        # row = list(attacked_idx[:, 0])
        # col = list(attacked_idx[:, 1])

        # for i, j in zip(row, col):

        #     if (i in sampled_idx) & (j in sampled_idx):
        #         continue

        #     _adj = adj.detach().clone()

        #     if _adj[i][j] == 1:
        #         _adj[i][j] = 0
        #     else:
        #         _adj[i][j] = 1

        #     _adj_norm = utils.normalize_adj_tensor(_adj)

        #     ## forward loop of GCN
        #     hidden = features
        #     for ix, w in enumerate(self.weights):
        #         b = self.biases[ix] if self.with_bias else 0
        #         if self.sparse_features:
        #             hidden = _adj_norm @ torch.spmm(hidden, w) + b
        #         else:
        #             hidden = _adj_norm @ hidden @ w + b

        #         if self.with_relu and ix != len(self.weights) - 1:
        #             hidden = F.relu(hidden)

        #     _output = F.log_softmax(hidden, dim=1)
        #     _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

        #     _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
        #     if self.with_bias:
        #         _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

            # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
            # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()

            # analysis_idx.append([i, j])
            # analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
            # analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())


        # for i in tqdm(sampled_idx):
        #     for j in sampled_idx:
        #         analysis_grad[iterate][i][j] = adj_grad[iterate][i][j]

        # self.analysis_idx.append(analysis_idx)
        # self.analysis_output.append(analysis_output)
        # self.analysis_inner_grad.append(analysis_inner_grad)
        
        if iterate == 0:
            analysis_dict = {'analysis_idx': torch.tensor(analysis_idx),
                                'analysis_output': torch.tensor(analysis_output),
                                'analysis_inner_grad': torch.tensor(analysis_inner_grad)}

            torch.save(analysis_dict, f'result/analysis/cora_Meta-Self_{self.train_iters}_ana.pt')
            
            # torch.save({"analysis_output": analysis_output.detach().cpu(), 
            # "analysis_inner_grad": analysis_inner_grad.detach(), 
            # "analysis_grad": analysis_grad.detach().cpu()}, 
            # f'result/analysis/cora_tensors.pt')
            # exit()
            
    def get_modified_adj(self, ori_adj):

        ## Meta Code
        # adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        # if self.undirected:
        #     adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        # adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        # modified_adj = self.adj_changes + ori_adj
        
        ## PGD Code
        # if self.complementary is None:
        #     self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
        # m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        # tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        # m[tril_indices[0], tril_indices[1]] = self.adj_changes
        # m = m + m.t()
        # modified_adj = self.complementary * m + ori_adj

        ## My
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
        
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m = torch.tril(self.adj_changes)
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj


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

        for i in tqdm(range(self.outer_iters), desc="Perturbing graph"):
            
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
            

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            # adj_meta_score = torch.tensor(0.0).to(self.device)
            # feature_meta_score = torch.tensor(0.0).to(self.device)
            # if self.attack_structure:
            #     adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            # if self.attack_features:
            #     feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            ## PGD
            lr = self.initial_step / (i+1)**(1/2)
            self.adj_changes.data.add_(lr * adj_grad)
            self.projection(n_perturbations)
            
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()

        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        # self.random_sample(ori_adj, ori_features, labels_self_training, np.concatenate([idx_train, idx_unlabeled], -1), n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)

    def random_sample(self, ori_adj, ori_features, labels, idx_train, n_perturbations):
        K = 100
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                # loss = F.nll_loss(output[idx_train], labels[idx_train])
                # print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss
        
    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=5e-6)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations
        cnt = 0
        miu = a
        while ((b-a) >= epsilon):
            cnt += 1

            if cnt % 10 == 0:
                epsilon = 2*epsilon
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu


    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}




class MetaIPGD(BaseMeta):
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

    def __init__(self, model, nnodes, y_adj, coef=0.01, initial_step=20.0, loss_type='CE', initial_graph = None,
                feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, 
                lambda_=0.5, train_iters=100, outer_iters=200, lr=0.1, momentum=0.9, analysis_mode=False, topk=None):

        super(MetaIPGD, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
        self.initial_step = initial_step
        self.train_iters = train_iters
        self.outer_iters = outer_iters
        self.with_bias = with_bias
        self.analysis_mode= analysis_mode
        self.topk = topk
        self.loss_type = loss_type

        self.y_adj = torch.tensor(y_adj.todense()).to(device)
        self.coef = coef
        self.outer_iters = outer_iters
        self.loss_type='CE'
        self.complementary = None

        if attack_structure:
            assert nnodes is not None, 'Please give nnodes='
            # self.adj_changes = Parameter(torch.FloatTensor(int(nnodes*(nnodes-1)/2)))

            if initial_graph:
                self.adj_changes = Parameter(torch.tril(torch.load(initial_graph) - torch.tensor(y_adj.todense())))
            else:
                self.adj_changes = Parameter(torch.zeros((nnodes, nnodes)).float())
                self.adj_changes.data.fill_(0)

            
                 

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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        
        self._initialize()

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

            output = F.log_softmax(hidden, dim=1)

            estim_adj = F.sigmoid(torch.mm(hidden, hidden.t()))

            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight= torch.tensor(float(self.y_adj.shape[0] * self.y_adj.shape[0] - self.y_adj.sum()) / self.y_adj.sum())
            loss_adj = F.binary_cross_entropy_with_logits(estim_adj, self.y_adj, weight=weight)
            loss = loss_labeled + self.coef * loss_adj

            # weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            weight_grads = torch.autograd.grad(loss, self.weights, create_graph=True)
            weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    


    def analysis(self, features, adj, idx_train, idx_unlabeled, labels, iterate):

        adj_norm = utils.normalize_adj_tensor(adj)

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

        weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=False)
        self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
        if self.with_bias:
            bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=False)
            self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]


        # analysis_output = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_inner_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))

        analysis_idx = []
        analysis_output = []
        analysis_grad = []
        analysis_inner_grad = []

        modified_features = features
        adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, self.self_training_label(labels, idx_train))

        ratio1 = 0.05
        ratio2 = 0.05
        nnodes = len(idx_train) + len(idx_unlabeled)
        sampled_idx_train = idx_train
        sampled_idx_train = idx_train[torch.randperm(len(idx_train))][:int(ratio1*nnodes)]
        sampled_idx_unlabeled = idx_unlabeled[torch.randperm(len(idx_unlabeled))][:int(ratio2*nnodes)]

        sampled_idx = np.concatenate([sampled_idx_train, sampled_idx_unlabeled], -1)


        for i in tqdm(sampled_idx):
            for j in sampled_idx:

                if i==j:
                    continue

                _adj = adj.detach().clone()

                if _adj[i][j] == 1:
                    _adj[i][j] = 0
                else:
                    _adj[i][j] = 1

                _adj_norm = utils.normalize_adj_tensor(_adj)

                ## forward loop of GCN
                hidden = features
                for ix, w in enumerate(self.weights):
                    b = self.biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = _adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = _adj_norm @ hidden @ w + b

                    if self.with_relu and ix != len(self.weights) - 1:
                        hidden = F.relu(hidden)

                _output = F.log_softmax(hidden, dim=1)
                _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

                _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
                if self.with_bias:
                    _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

                # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
                # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()


                analysis_idx.append([i, j])
                analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
                analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())

                del _output
                del _loss_labeled
                del _weight_grads
                

        
        # attacked_idx = torch.load('result/analysis/polblogs_Meta-Self_attack.pt')

        # row = list(attacked_idx[:, 0])
        # col = list(attacked_idx[:, 1])

        # for i, j in zip(row, col):

        #     if (i in sampled_idx) & (j in sampled_idx):
        #         continue

        #     _adj = adj.detach().clone()

        #     if _adj[i][j] == 1:
        #         _adj[i][j] = 0
        #     else:
        #         _adj[i][j] = 1

        #     _adj_norm = utils.normalize_adj_tensor(_adj)

        #     ## forward loop of GCN
        #     hidden = features
        #     for ix, w in enumerate(self.weights):
        #         b = self.biases[ix] if self.with_bias else 0
        #         if self.sparse_features:
        #             hidden = _adj_norm @ torch.spmm(hidden, w) + b
        #         else:
        #             hidden = _adj_norm @ hidden @ w + b

        #         if self.with_relu and ix != len(self.weights) - 1:
        #             hidden = F.relu(hidden)

        #     _output = F.log_softmax(hidden, dim=1)
        #     _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

        #     _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
        #     if self.with_bias:
        #         _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

            # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
            # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()

            # analysis_idx.append([i, j])
            # analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
            # analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())


        # for i in tqdm(sampled_idx):
        #     for j in sampled_idx:
        #         analysis_grad[iterate][i][j] = adj_grad[iterate][i][j]

        # self.analysis_idx.append(analysis_idx)
        # self.analysis_output.append(analysis_output)
        # self.analysis_inner_grad.append(analysis_inner_grad)
        
        if iterate == 0:
            analysis_dict = {'analysis_idx': torch.tensor(analysis_idx),
                                'analysis_output': torch.tensor(analysis_output),
                                'analysis_inner_grad': torch.tensor(analysis_inner_grad)}

            torch.save(analysis_dict, f'result/analysis/cora_Meta-Self_{self.train_iters}_ana.pt')
            
            # torch.save({"analysis_output": analysis_output.detach().cpu(), 
            # "analysis_inner_grad": analysis_inner_grad.detach(), 
            # "analysis_grad": analysis_grad.detach().cpu()}, 
            # f'result/analysis/cora_tensors.pt')
            # exit()
            
    def get_modified_adj(self, ori_adj):

        ## Meta Code
        # adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        # if self.undirected:
        #     adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        # adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        # modified_adj = self.adj_changes + ori_adj
        
        ## PGD Code
        # if self.complementary is None:
        #     self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
        # m = torch.zeros((self.nnodes, self.nnodes)).to(self.device)
        # tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        # m[tril_indices[0], tril_indices[1]] = self.adj_changes
        # m = m + m.t()
        # modified_adj = self.complementary * m + ori_adj

        ## My
        if self.complementary is None:
            self.complementary = (torch.ones_like(ori_adj) - torch.eye(self.nnodes).to(self.device) - ori_adj) - ori_adj
        
        tril_indices = torch.tril_indices(row=self.nnodes, col=self.nnodes, offset=-1)
        m = torch.tril(self.adj_changes)
        m = m + m.t()
        modified_adj = self.complementary * m + ori_adj

        return modified_adj


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

        for i in tqdm(range(self.outer_iters), desc="Perturbing graph"):
            
            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)
            

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            # adj_meta_score = torch.tensor(0.0).to(self.device)
            # feature_meta_score = torch.tensor(0.0).to(self.device)
            # if self.attack_structure:
            #     adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj, ll_constraint, ll_cutoff)
            # if self.attack_features:
            #     feature_meta_score = self.get_feature_score(feature_grad, modified_features)

            # Improved Gradient
            grad_mask = torch.zeros_like(adj_grad)
            if self.topk is None:
                grad_mask[torch.topk(adj_grad, n_perturbations)[1]]=1
            else:
                grad_mask[torch.topk(adj_grad, self.topk)[1]]=1

            ## PGD
            lr = self.initial_step / (i+1)**(1/2)
            self.adj_changes.data.add_(lr * adj_grad * grad_mask)
            self.projection(n_perturbations)
            
        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()

        self.random_sample(ori_adj, ori_features, labels, idx_train, n_perturbations)
        # self.random_sample(ori_adj, ori_features, labels_self_training, np.concatenate([idx_train, idx_unlabeled], -1), n_perturbations)
        self.modified_adj = self.get_modified_adj(ori_adj).detach()
        self.check_adj_tensor(self.modified_adj)

    def random_sample(self, ori_adj, ori_features, labels, idx_train, n_perturbations):
        K = 100
        best_loss = -1000
        victim_model = self.surrogate
        victim_model.eval()
        with torch.no_grad():
            s = self.adj_changes.cpu().detach().numpy()
            for i in range(K):
                sampled = np.random.binomial(1, s)

                # print(sampled.sum())
                if sampled.sum() > n_perturbations:
                    continue
                self.adj_changes.data.copy_(torch.tensor(sampled))
                modified_adj = self.get_modified_adj(ori_adj)
                adj_norm = utils.normalize_adj_tensor(modified_adj)
                output = victim_model(ori_features, adj_norm)
                loss = self._loss(output[idx_train], labels[idx_train])
                # loss = F.nll_loss(output[idx_train], labels[idx_train])
                # print(loss)
                if best_loss < loss:
                    best_loss = loss
                    best_s = sampled
            self.adj_changes.data.copy_(torch.tensor(best_s))

    def _loss(self, output, labels):
        if self.loss_type == "CE":
            loss = F.nll_loss(output, labels)
        if self.loss_type == "CW":
            onehot = utils.tensor2onehot(labels)
            best_second_class = (output - 1000*onehot).argmax(1)
            margin = output[np.arange(len(output)), labels] - \
                   output[np.arange(len(output)), best_second_class]
            k = 0
            loss = -torch.clamp(margin, min=k).mean()
            # loss = torch.clamp(margin.sum()+50, min=k)
        return loss
        
    def projection(self, n_perturbations):
        # projected = torch.clamp(self.adj_changes, 0, 1)
        if torch.clamp(self.adj_changes, 0, 1).sum() > n_perturbations:
            left = (self.adj_changes - 1).min()
            right = self.adj_changes.max()
            miu = self.bisection(left, right, n_perturbations, epsilon=5e-6)
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data - miu, min=0, max=1))
        else:
            self.adj_changes.data.copy_(torch.clamp(self.adj_changes.data, min=0, max=1))

    def bisection(self, a, b, n_perturbations, epsilon):
        def func(x):
            return torch.clamp(self.adj_changes-x, 0, 1).sum() - n_perturbations
        cnt = 0
        miu = a
        while ((b-a) >= epsilon):
            cnt += 1

            if cnt % 10 == 0:
                epsilon = 2*epsilon
            miu = (a+b)/2
            # Check if middle point is root
            if (func(miu) == 0.0):
                break
            # Decide the side to repeat the steps
            if (func(miu)*func(a) < 0):
                b = miu
            else:
                a = miu
        # print("The value of root is : ","%.4f" % miu)
        return miu


    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}




class MetattackCW(BaseMeta):
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

    def __init__(self, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9, analysis_mode=False):

        super(MetattackCW, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)
        self.momentum = momentum
        self.lr = lr
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

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        
        self._initialize()

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

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            # weight_grads = tuple(torch.clamp(w, min=-1, max=1) for w in weight_grads)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def _loss(self, output, labels):
        onehot = utils.tensor2onehot(labels)
        best_second_class = (output - 1000*onehot).argmax(1)
        margin = output[np.arange(len(output)), labels] - \
                output[np.arange(len(output)), best_second_class]
        k = 0
        loss = -torch.clamp(margin, min=k).mean()
        # loss = torch.clamp(margin.sum()+50, min=k)
        return loss


    def analysis(self, features, adj, idx_train, idx_unlabeled, labels, iterate):

        adj_norm = utils.normalize_adj_tensor(adj)

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

        weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=False)
        self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
        if self.with_bias:
            bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=False)
            self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]


        # analysis_output = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))
        # analysis_inner_grad = torch.zeros((50, adj_norm.shape[0], adj_norm.shape[1]))

        analysis_idx = []
        analysis_output = []
        analysis_grad = []
        analysis_inner_grad = []

        modified_features = features
        adj_grad, feature_grad = self.get_meta_grad(modified_features, adj_norm, idx_train, idx_unlabeled, labels, self.self_training_label(labels, idx_train))

        ratio1 = 0.05
        ratio2 = 0.05
        nnodes = len(idx_train) + len(idx_unlabeled)
        sampled_idx_train = idx_train
        sampled_idx_train = idx_train[torch.randperm(len(idx_train))][:int(ratio1*nnodes)]
        sampled_idx_unlabeled = idx_unlabeled[torch.randperm(len(idx_unlabeled))][:int(ratio2*nnodes)]

        sampled_idx = np.concatenate([sampled_idx_train, sampled_idx_unlabeled], -1)


        for i in tqdm(sampled_idx):
            for j in sampled_idx:

                if i==j:
                    continue

                _adj = adj.detach().clone()

                if _adj[i][j] == 1:
                    _adj[i][j] = 0
                else:
                    _adj[i][j] = 1

                _adj_norm = utils.normalize_adj_tensor(_adj)

                ## forward loop of GCN
                hidden = features
                for ix, w in enumerate(self.weights):
                    b = self.biases[ix] if self.with_bias else 0
                    if self.sparse_features:
                        hidden = _adj_norm @ torch.spmm(hidden, w) + b
                    else:
                        hidden = _adj_norm @ hidden @ w + b

                    if self.with_relu and ix != len(self.weights) - 1:
                        hidden = F.relu(hidden)

                _output = F.log_softmax(hidden, dim=1)
                _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

                _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
                if self.with_bias:
                    _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

                # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
                # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()


                analysis_idx.append([i, j])
                analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
                analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())

                del _output
                del _loss_labeled
                del _weight_grads
                

        
        # attacked_idx = torch.load('result/analysis/polblogs_Meta-Self_attack.pt')

        # row = list(attacked_idx[:, 0])
        # col = list(attacked_idx[:, 1])

        # for i, j in zip(row, col):

        #     if (i in sampled_idx) & (j in sampled_idx):
        #         continue

        #     _adj = adj.detach().clone()

        #     if _adj[i][j] == 1:
        #         _adj[i][j] = 0
        #     else:
        #         _adj[i][j] = 1

        #     _adj_norm = utils.normalize_adj_tensor(_adj)

        #     ## forward loop of GCN
        #     hidden = features
        #     for ix, w in enumerate(self.weights):
        #         b = self.biases[ix] if self.with_bias else 0
        #         if self.sparse_features:
        #             hidden = _adj_norm @ torch.spmm(hidden, w) + b
        #         else:
        #             hidden = _adj_norm @ hidden @ w + b

        #         if self.with_relu and ix != len(self.weights) - 1:
        #             hidden = F.relu(hidden)

        #     _output = F.log_softmax(hidden, dim=1)
        #     _loss_labeled = F.nll_loss(_output[idx_train], labels[idx_train])

        #     _weight_grads = torch.autograd.grad(_loss_labeled, self.weights, create_graph=False)
        #     if self.with_bias:
        #         _bias_grads = torch.autograd.grad(_loss_labeled, self.biases, create_graph=False)

            # analysis_output[iterate][i][j] = torch.abs(F.softmax(output) - F.softmax(_output)).sum()
            # analysis_inner_grad[iterate][i][j] = torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()

            # analysis_idx.append([i, j])
            # analysis_output.append(torch.abs(F.softmax(output) - F.softmax(_output)).sum().detach().cpu())
            # analysis_inner_grad.append((torch.abs(weight_grads[0] - _weight_grads[0]).sum() + torch.abs(weight_grads[1] - _weight_grads[1]).sum()).detach().cpu())


        # for i in tqdm(sampled_idx):
        #     for j in sampled_idx:
        #         analysis_grad[iterate][i][j] = adj_grad[iterate][i][j]

        # self.analysis_idx.append(analysis_idx)
        # self.analysis_output.append(analysis_output)
        # self.analysis_inner_grad.append(analysis_inner_grad)
        
        if iterate == 0:
            analysis_dict = {'analysis_idx': torch.tensor(analysis_idx),
                                'analysis_output': torch.tensor(analysis_output),
                                'analysis_inner_grad': torch.tensor(analysis_inner_grad)}

            torch.save(analysis_dict, f'result/analysis/cora_Meta-Self_{self.train_iters}_ana.pt')
            
            # torch.save({"analysis_output": analysis_output.detach().cpu(), 
            # "analysis_inner_grad": analysis_inner_grad.detach(), 
            # "analysis_grad": analysis_grad.detach().cpu()}, 
            # f'result/analysis/cora_tensors.pt')
            # exit()
            



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

        loss_labeled = self._loss(output[idx_train], labels[idx_train])
        loss_unlabeled = self._loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = self._loss(output[idx_unlabeled], labels[idx_unlabeled])

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

            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

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

            # if True:
            #     if i == 0:
            #         self.debug=[]

            #     if (row_idx.item(), col_idx.item()) in self.debug:
            #         print("?")

            #     self.debug.append((row_idx.item(), col_idx.item()))

                



            if self.analysis_mode:
                attacked_idx.append([row_idx.item(), col_idx.item()])
                self.analysis(modified_features, modified_adj, idx_train, idx_unlabeled, labels, iterate=i)
                if i == 0:
                    attacked_idx = torch.tensor(attacked_idx)
                    # torch.save(attacked_idx, f'result/analysis/polblogs_Meta-Self_attack.pt')
                    exit()
                

            # else:
            #     feature_meta_argmax = torch.argmax(feature_meta_score)
            #     row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
            #     self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

    def set_grad_stats(self):

        keys = ['grad_avg_ll', 'grad_avg_ul', 'grad_avg_uu', 'grad_topk_avg_ll', 'grad_topk_avg_ul', 
        'grad_topk_avg_uu', 'grad_topk_cnt_ll', 'grad_topk_cnt_ul', 'grad_topk_cnt_uu']

        self.grad_stats = {k: [] for k in keys}