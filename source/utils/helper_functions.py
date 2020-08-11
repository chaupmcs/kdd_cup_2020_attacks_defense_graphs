import yaml
import os
import pickle
import scipy.sparse as sp
import numpy as np
import torch
import ntpath
import random
from random import shuffle
from scipy.sparse import identity

def read_config():
    with open("config.yaml", 'r') as ymlfile:
        parameters = yaml.load(ymlfile)
    return parameters

def read_train_valid_test(path):
    with open(os.path.join(path, 'idx_train.pkl'), 'rb') as handle:
        idx_train = pickle.load(handle)

    with open(os.path.join(path, 'idx_valid.pkl'), 'rb') as handle:
        idx_valid = pickle.load(handle)

    with open(os.path.join(path, 'idx_test.pkl'), 'rb') as handle:
        idx_test = pickle.load(handle)

    try:
        # read pickle
        with open("/Users/chaupham/minhchau/jupyter_projects/KDD_Attack_Defense_Graph/notebooks/test_half.pkl", 'rb') as handle:
            idx_test_half = pickle.load(handle)
    except:
        idx_test_half = None


    return idx_train, idx_valid, idx_test, idx_test_half

def load_raw_input(data_path):
    adj = np.load(os.path.join(data_path, 'adj.pkl'), allow_pickle=True)
    try:
        features = np.load(os.path.join(data_path, 'features.pkl'), allow_pickle=True)
    except:
        features = np.load(os.path.join(data_path, 'features.npy'), allow_pickle=True)


    try:
        labels = np.load(os.path.join(data_path, 'train.pkl'), allow_pickle=True)
    except:
        labels = None
    return adj, features, labels


def accuracy_origin(output, labels, le, return_preds_only=False):
    preds = output.max(1)[1]
    preds = np.array(le.inverse_transform(preds))


    if return_preds_only:
        return  preds
    else:
        correct = 0
        for p, l in zip(preds, labels):
            if p == l:
                correct += 1
        return correct / len(labels)

def accuracy(output, labels):
    if type(labels) is not torch.Tensor:
        labels = torch.LongTensor(labels)
    preds = output.max(1)[1].type_as(labels)

    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def reshape_mx(mx, shape):
    indices = mx.nonzero()
    return sp.csr_matrix((mx.data, (indices[0], indices[1])), shape=shape)

def get_origin_metric(file_path):
    with open(file_path, 'r') as f:
        last_line = f.readlines()[-1]
    # print("last_line:", last_line)

    acc = float(last_line.split(",")[-1].split("=")[-1].strip())
    logloss = float(last_line.split(",")[-2].split("=")[-1].strip())
    return acc, logloss


def path_filename(path):
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

def chunks(lst, n,  seed = 2020, is_shuffle=True):
    """Yield successive n-sized chunks from lst."""
    random.seed(seed)
    if is_shuffle:
        shuffle(lst)

    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def normalize_adj_gcn(adj):
    """
    A_hat = D'_rev*(A')*D'_rev
    """
    adj_tilde = adj + identity(adj.shape[0], dtype='int8')

    degree = np.array(np.sum(adj_tilde, axis=0))[0]
    degree_inverse = degree ** (-1 / 2)

    D_tilde_inverse = sp.diags(degree_inverse)
    adj_hat = D_tilde_inverse * adj_tilde * D_tilde_inverse

    return adj_hat

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)