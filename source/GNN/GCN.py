from sklearn.preprocessing import LabelEncoder
from utils.helper_functions import *
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from copy import deepcopy
import torch
from scipy.sparse import identity
import numpy as np
import logging

from source.utils.helper_functions import load_raw_input


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        # self.weight.data.fill_(1)
        # if self.bias is not None:
        #     self.bias.data.fill_(1)

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        if input.data.is_sparse:
            support = torch.spmm(input, self.weight)
        else:
            support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):


    def __init__(self, nfeat, nhid, nclass, dropout, lr, weight_decay,  n_epochs=1000, early_stopping=200, with_relu=True, with_bias=True,
                 device=None, log_path= None, verbose=1):
        super(GCN, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.verbose = verbose
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, nclass, with_bias=with_bias)

        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.adj_norm = None
        self.features = None
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping


        logging.basicConfig(filename=log_path, level=logging.DEBUG)


    def forward(self, x, adj):
        '''
            adj: normalized adjacency matrix
        '''
        if self.with_relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def normalize_adj_baseline(self, adj):
        """
        A_hat = D_rev*(D+A)*D_rev
        A_hat = D_rev*(D-A)*D_rev
        todo: many ways to try!
        """
        A_hat = adj
        degree = np.array(np.sum(A_hat, axis=0))[0]
        D_hat = sp.diags(degree)
        degree_reverse = degree ** (-1 / 2)
        degree_reverse[np.isinf(degree_reverse)] = 0.
        D_hat_reverse = sp.diags(degree_reverse)
        adj = D_hat_reverse * (D_hat + A_hat) * D_hat_reverse
        return adj


    def normalize_adj_gcn(self, adj):
        """
        A_hat = D'_rev*(A')*D'_rev
        """
        adj_tilde = adj + identity(adj.shape[0], dtype='int8')
        degree = np.array(np.sum(adj_tilde, axis=0))[0]
        degree_inverse = degree ** (-1 / 2)
        D_tilde_inverse = sp.diags(degree_inverse)
        adj_hat = D_tilde_inverse * adj_tilde * D_tilde_inverse
        return adj_hat

    def norm_and_to_tensor(self, normalize, adj, features, labels = None):
        if normalize != "no_norm":
            print(f"Normalizing by {normalize}()...")
            logging.info(f"Normalizing by {normalize}()...")

            if normalize == "normalize_adj_gcn":
                adj_norm = self.normalize_adj_gcn(adj)
            elif normalize == "normalize_adj_baseline":
                adj_norm = self.normalize_adj_baseline(adj)
            else:
                adj_norm = adj
        else:
            adj_norm = adj

        adj_norm = self.sparse_mx_to_torch_sparse_tensor(adj_norm)
        features = sp.csr_matrix(features, dtype=np.float32)
        features = self.sparse_mx_to_torch_sparse_tensor(features)

        if labels is not None:
            labels = torch.LongTensor(labels)
            labels = labels.to(self.device)

        features = features.to(self.device)
        adj_norm = adj_norm.to(self.device)

        return adj_norm, features, labels

    def fit(self, features, adj, labels, idx_train, idx_val = None,  initialize = True,
             normalize = "normalize_adj_gcn"):

        verbose = self.verbose
        patience = self.early_stopping
        train_iters = self.n_epochs

        self.device = self.gc1.weight.device
        if initialize:
            self.initialize()

        self.adj_norm, self.features, self.labels = self.norm_and_to_tensor(normalize, adj, features, labels)
        self._train_with_early_stopping(self.labels, idx_train, idx_val, train_iters, patience, verbose)

    def _train_with_early_stopping(self, labels, idx_train, idx_val, train_iters, patience, verbose):
        if verbose:
            print('training gcn model...')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        early_stopping = patience
        best_loss_val = np.inf

        for i in range(1,train_iters+1):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.adj_norm)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])  # , weight=_weight)
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward(self.features, self.adj_norm)

            loss_val = F.nll_loss(output[idx_val], labels[idx_val])

            if verbose and i % verbose == 0:
                msg = 'Epoch {}, training loss: {}, valid loss: {}'.format(i, loss_train.item(), loss_val.item())
                print(msg)
                logging.info(msg)


            if best_loss_val > loss_val:
                best_loss_val = loss_val
                self.output = output
                weights = deepcopy(self.state_dict())
                patience = early_stopping
            else:
                patience -= 1
            if i > early_stopping and patience <= 0:
                break

        if verbose:
            msg = '=== stopped at iter {0}, loss_val = {1} ==='.format(i, best_loss_val)
            print(msg)
            logging.info(msg)

        self.load_state_dict(weights)

    def test(self, idx_test):
        self.eval()
        self.output = self._predict()


        loss_test = F.nll_loss(self.output[idx_test], self.labels[idx_test])
        acc_test = accuracy(self.output[idx_test], self.labels[idx_test])

        msg = "Test set results: loss= {:.4f}, accuracy= {:.4f}".format(loss_test.item(), acc_test.item())
        print(msg)
        logging.info(msg)

        return acc_test, loss_test


    def test_attack(self, normalize, idx_test, data_path):
        self.eval()
        adj, features, labels_origin = load_raw_input(data_path)

        le = LabelEncoder()
        labels = le.fit_transform(labels_origin)

        self.adj_norm, self.features, self.labels = self.norm_and_to_tensor(normalize, adj, features, labels)

        self.features = self.features.to(self.device)
        self.adj_norm = self.adj_norm.to(self.device)

        self.output =  self.forward(self.features, self.adj_norm)


        loss_test = F.nll_loss(self.output[idx_test], self.labels[idx_test])
        acc_test_origin = accuracy_origin(self.output[idx_test], labels_origin[idx_test], le)
        acc_test = accuracy(self.output[idx_test], labels[idx_test])


        msg = "Test set results for attack_log: loss= {:.4f}, accuracy_origin= {:.4f}, accuracy= {:.4f}".format(loss_test.item(), acc_test_origin, acc_test.item())
        print(msg)
        return acc_test, loss_test


    def _predict(self, features = None, adj_norm = None):
        '''By default, inputs are unnormalized data'''

        self.eval()

        if features is None and adj_norm is None:
            features = self.features.to(self.device)
            adj_norm = self.adj_norm.to(self.device)
        else:
            features = features.to(self.device)
            adj_norm = adj_norm.to(self.device)
        return self.forward(features, adj_norm)





