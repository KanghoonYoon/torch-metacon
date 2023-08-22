from deeprobusts.graph.defense import GCN

import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
from tqdm import tqdm
import utils
import math
import scipy.sparse as sp

class BaseAttack(Module):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_bias=False, lr=0.01, with_relu=False):
        super(BaseAttack, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.nfeat = nfeat
        self.nclass = nclass
        self.with_bias = with_bias
        self.with_relu = with_relu

        self.gcn = GCN(nfeat=nfeat,
                       nhid=hidden_sizes[0],
                       nclass=nclass,
                       dropout=dropout,
                       with_relu=False)


        self.train_iters = train_iters
        self.surrogate_optimizer = optim.Adam(self.gcn.parameters(), lr=lr, weight_decay=5e-4)

        self.attack_features = attack_features
        self.lambda_ = lambda_
        self.device = device
        self.nnodes = nnodes

        self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes))
        self.adj_changes.data.fill_(0)

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.

        Returns
        -------
        torch.Tensor shape [N, N], float with ones everywhere except the entries of potential singleton nodes,
        where the returned tensor has value 0.

        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()

        l_and = resh * modified_adj
        logical_and_symmetric = l_and + l_and.t()
        flat_mask = 1 - logical_and_symmetric
        return flat_mask

    def train_surrogate(self, features, adj, labels, idx_train, train_iters=200):
        print('=== training surrogate model to predict unlabled data for self-training')
        surrogate = self.gcn
        surrogate.initialize()

        adj_norm = utils.normalize_adj_tensor(adj)
        surrogate.train()
        for i in range(train_iters):
            self.surrogate_optimizer.zero_grad()
            output = surrogate(features, adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            self.surrogate_optimizer.step()

        # Predict the labels of the unlabeled nodes to use them for self-training.
        surrogate.eval()
        output = surrogate(features, adj_norm)
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        # reset parameters for later updating
        surrogate.initialize()
        return labels_self_training


    def log_likelihood_constraint(self, modified_adj, ori_adj, ll_cutoff):
        """
        Computes a mask for entries that, if the edge corresponding to the entry is added/removed, would lead to the
        log likelihood constraint to be violated.
        """

        t_d_min = torch.tensor(2.0).to(self.device)
        t_possible_edges = np.array(np.triu(np.ones((self.nnodes, self.nnodes)), k=1).nonzero()).T
        allowed_mask, current_ratio = utils.likelihood_ratio_filter(t_possible_edges,
                                                                    modified_adj,
                                                                    ori_adj, t_d_min,
                                                                    ll_cutoff)

        return allowed_mask, current_ratio


class GraD(BaseAttack):

    def __init__(self, nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters,
                 attack_features, device, args, lambda_=0.5, with_relu=False, with_bias=False, lr=0.1):

        super(GraD, self).__init__(nfeat, hidden_sizes, nclass, nnodes, dropout, train_iters, attack_features, lambda_, device, with_bias=with_bias, with_relu=with_relu)

        self.lr = lr
        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        previous_size = nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            bias = Parameter(torch.FloatTensor(nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            b_velocity = torch.zeros(bias.shape).to(device)
            previous_size = nhid

            self.weights.append(weight)
            self.biases.append(bias)
            self.w_velocities.append(w_velocity)
            self.b_velocities.append(b_velocity)

        output_weight = Parameter(torch.FloatTensor(previous_size, nclass).to(device))
        output_bias = Parameter(torch.FloatTensor(nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        output_b_velocity = torch.zeros(output_bias.shape).to(device)

        self.weights.append(output_weight)
        self.biases.append(output_bias)
        self.w_velocities.append(output_w_velocity)
        self.b_velocities.append(output_b_velocity)

        # Set the momentum to train the surrogate model properly
        if args.dataset == 'cora':
            self.momentum = 0.9
        elif args.dataset == 'cora_ml':
            self.momentum = 0.97
        elif args.dataset == 'citeseer':
            self.momentum = 0.9
        elif args.dataset == 'polblogs':
            self.momentum = 0.99
        elif args.dataset == 'pubmed':
            self.momentum = 0.98

        self._initialize()

    def _initialize(self):

        for w, b in zip(self.weights, self.biases):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)


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
                if self.with_relu:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]

    def get_structural_grad(self, features, adj_norm, idx_unlabeled, labels_self_training):

        hidden = features
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu:
                hidden = F.relu(hidden)

        output = F.softmax(hidden,dim=1)
        attack_objective_GraD = -output[idx_unlabeled,labels_self_training[idx_unlabeled]].mean()
        adj_grad = torch.autograd.grad(attack_objective_GraD, self.adj_changes, retain_graph=True)[0]

        return adj_grad


    def forward(self, features, ori_adj, labels, idx_train, idx_unlabeled, perturbations):
        self.sparse_features = sp.issparse(features)

        adj_norm = utils.normalize_adj_tensor(ori_adj)
        self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
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
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]

        iters = 0
        for i in tqdm(range(perturbations), desc="Perturbing graph"):
            iters = iters+1
            adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
            ind = np.diag_indices(self.adj_changes.shape[0])
            adj_changes_symm = torch.clamp(adj_changes_square + torch.transpose(adj_changes_square, 1, 0), -1, 1)
            modified_adj = adj_changes_symm + ori_adj

            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(features, adj_norm, idx_train, idx_unlabeled, labels)
            adj_grad = self.get_structural_grad(features, adj_norm, idx_unlabeled, labels_self_training)

            directed_adj_grad = adj_grad * (-2 * modified_adj + 1)
            directed_adj_grad -= directed_adj_grad.min()
            directed_adj_grad -= torch.diag(torch.diag(directed_adj_grad, 0))
            singleton_mask = self.filter_potential_singletons(modified_adj)
            directed_adj_grad = directed_adj_grad *  singleton_mask

            # Get argmax of the gradients.
            adj_grad_argmax = torch.argmax(directed_adj_grad)
            row_idx, col_idx = utils.unravel_index(adj_grad_argmax, ori_adj.shape)
            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
            self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)

            if self.attack_features:
                pass

        return self.adj_changes + ori_adj