import os
import pickle
import torch.nn.functional as F
import logging
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from utils.helper_functions import chunks, reshape_mx
import datetime
import random
from utils.helper_functions import *
from scipy import sparse
from scipy.special import softmax

from sklearn.metrics import log_loss
def cross_entropy(predictions, targets):
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions)) / N
    return ce


### Add Nodes/Features
def add_edges_to_adj(adj, list_nodes):
    old_dimension = adj.shape[0]
    new_shape = old_dimension + len(list_nodes)
    new_adj = reshape_mx(adj, (new_shape, new_shape))

    new_adj = new_adj.tolil()
    for i in range(500):
        nodes = list_nodes[i]
        for node in nodes:
            new_adj[old_dimension+i, node] = 1
            new_adj[node, old_dimension + i] = 1

    return sparse.csr_matrix(new_adj)

def add_features(features, added_features):
    new_features = np.vstack([features, added_features])
    return new_features


### Nodes
def pick_nodes_to_add_edges_randomly(idx_test, adj, seed = 2020, n_edges=100, is_shuffle=True):
    list_nodes = list(chunks(idx_test, n_edges,  seed, is_shuffle))
    return list_nodes[:500]

def pick_nodes_to_add_edges_randomly_HALF(idx_test_half, adj, seed = 2020, n_edges=100, is_shuffle=True):
    list_nodes_1 = list(chunks(idx_test_half, n_edges,  seed, is_shuffle))
    list_nodes_2 = list(chunks(idx_test_half, n_edges,  seed+1, is_shuffle))
    list_nodes = list_nodes_1 + list_nodes_2
    return list_nodes

def pick_nodes_to_add_edges_randomly_HALF_Cluster(idx_test_half_cluster, adj, seed = 2020, n_edges=100, is_shuffle=False):
    res = list(chunks(list(idx_test_half_cluster) + list(idx_test_half_cluster), n_edges,  seed, is_shuffle))
    return res

def pick_nodes_by_cluster():
    pass


### Features
def initialize_features_randomly(n_nodes, random_range = (-1,1), seed=2020):
    np.random.seed(seed)
    added_features = np.random.uniform(*random_range, n_nodes * 100).reshape(-1, 100)
    return added_features

def initialize_features_by_randomly_copy_nodes(features, n_nodes = 500, seed = 2020):
    random.seed(seed)
    idx_to_copy_list = random.sample(range(1, len(features) + 1), n_nodes)
    return features[idx_to_copy_list]

def initialize_features_by_pick_some_nodes(features, n_nodes = 500, seed = 2020):
    pass




##### Attack strategies ####
def fgsm(adj, features, labels_origin, model_path, n_attack_epochs, idx_opt, output_path=None, n_nodes = 500):
    time_start = str(datetime.datetime.now())
    log_path = os.path.join(output_path, "attack_log", "fgsm_{}".format(time_start))

    logging.basicConfig(filename=log_path, level=logging.DEBUG)
    logging.info(f"log_path: {log_path}")

    adj_norm = normalize_adj_gcn(adj)

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    le = LabelEncoder()
    labels = le.fit_transform(labels_origin)
    labels = torch.LongTensor(labels)

    features = torch.FloatTensor(features)
    features.requires_grad_(True)

    model = torch.load(model_path)

    raw_origin_pred_ = model._predict(features, adj_norm)
    raw_origin_pred = raw_origin_pred_.max(1)[1]

    # 6 attacking on the features
    turn = n_attack_epochs
    ite=1

    max_logloss = -np.inf
    min_acc = np.inf

    print("new .....")
    while turn > 0:
        output = model._predict(features, adj_norm)
        # loss_test = F.nll_loss(output[idx_opt], raw_output[idx_opt])


        tmp =  F.log_softmax(output[idx_opt],1)

        #  F.nll_loss( F.log_softmax(output[idx_opt], 1), raw_output[idx_opt]) -\
        loss_test =  F.nll_loss( tmp, raw_origin_pred[idx_opt])
        acc_test = accuracy(output[idx_opt], labels[idx_opt])

        logloss, acc = loss_test.item(), acc_test.item()
        if max_logloss < logloss:
            max_logloss = logloss

        if min_acc > acc:
            min_acc = acc

        msg1 = "iter {}: loss= {}, accuracy= {}".format (ite, logloss, acc)
        print(msg1)
        logging.info(msg1)

        turn -= 1
        ite += 1

        if turn == 0:
            break

        # 6.1 calculate the derivative on the features
        grad = torch.autograd.grad(loss_test, features, retain_graph=True)[0]

        line_base = len(features)  - n_nodes
        for k in range(n_nodes):
            line = line_base + k
            for n in range(100):
                # 6.2 renew features
                if grad[line][n] > 0:
                    if features[line, n] < 0:
                        features[line, n] = features[line, n] + 2
                    else:
                        features[line, n] = 1.9999
                elif grad[line][n] < 0:
                    if features[line, n] > 0:
                        features[line, n] = features[line, n] - 2
                    else:
                        features[line, n] = -1.9999



    # write pickle
    last_attack_pred = accuracy_origin(output[idx_opt], labels[idx_opt], le, return_preds_only=True)
    last_attack_pred_path = os.path.join(output_path, "preds", "idx_test_attack_pred.pkl")
    with open(last_attack_pred_path, 'wb') as handle:
        pickle.dump(last_attack_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # write pickle
    raw_origin_pred_path = os.path.join(output_path, "preds", "raw_pred.pkl")
    with open(raw_origin_pred_path, 'wb') as handle:
        pickle.dump(raw_origin_pred_[idx_opt], handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("store last attack predict for idx_test at", last_attack_pred_path)
    print("store raw predict for idx_test at ", raw_origin_pred_path)

    adj = adj.astype("float32").tocsr()
    features = features.cpu().detach().numpy()

    attacker_path = os.path.join(output_path, "attacker", "fgsm_{}".format(time_start) + "-" + str(max_logloss) + "-" + str(min_acc))
    os.makedirs(attacker_path)
    np.save(os.path.join(attacker_path, 'features.npy'), features)
    with open(os.path.join(attacker_path, 'adj.pkl'), 'wb') as f:
        pickle.dump(adj, f)

    # os.makedirs(path)
    # np.save(os.path.join(path, 'features.npy'), features[line_base:])
    # with open(os.path.join(path, 'adj.pkl'), 'wb') as f:
    #     pickle.dump(adj[line_base:, :], f)

    print("log path:", log_path)
    msg = "Done attack! Save at {}".format(attacker_path)
    print(msg)
    print( max_logloss, min_acc )

    logging.info("min_logloss: {}, min_acc: {}".format(max_logloss, min_acc))
    logging.info(msg)

    return attacker_path


def gradient_attack(adj, features, labels_origin, model_path, n_attack_epochs, idx_opt, \
                    output_path=None, n_nodes = 500, lr_attack =100000, test_mode = False, use_steepest=False):


    if test_mode:
        print("TEST MODE!")

    time_start = str(datetime.datetime.now())

    name = "gradient_attack_{}"
    log_path = os.path.join(output_path, "attack_log", name.format(time_start))

    logging.basicConfig(filename=log_path, level=logging.DEBUG)
    logging.info(f"log_path: {log_path}")

    adj_norm = normalize_adj_gcn(adj)

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    le = LabelEncoder()
    labels = le.fit_transform(labels_origin)
    labels = torch.LongTensor(labels)

    features = torch.FloatTensor(features)
    features.requires_grad_(True)

    model = torch.load(model_path)

    raw_origin_pred_ = model._predict(features, adj_norm)
    raw_origin_pred = raw_origin_pred_.max(1)[1]



    # 6 attacking on the features
    turn = n_attack_epochs
    ite = 1

    max_logloss = -np.inf
    min_acc = np.inf

    print("gradientnnnnnnnhhhh.....")
    lr_attack_ex = 0.99

    while turn > 0:
        output = model._predict(features, adj_norm)
        # loss_test = F.nll_loss(output[idx_opt], raw_output[idx_opt])
        tmp1 = F.log_softmax(output[idx_opt], 1)

        # tmp2 = raw_origin_pred[idx_opt].detach().numpy().reshape(-1, 1)
        # cross_entropy(tmp1, 1 - tmp2)

        loss_test = F.nll_loss( tmp1, raw_origin_pred[idx_opt])
        if not test_mode:
            acc_test = accuracy(output[idx_opt], labels[idx_opt])
            logloss, acc = loss_test.item(), acc_test.item()

            if min_acc > acc:
                min_acc = acc

        else:
            logloss = loss_test.item()
            acc  = 0

        if max_logloss < logloss:
            max_logloss = logloss

        max_ =  1.6222137
        min_ =  -1.7355031

        msg1 = "iter {}: loss= {}, accuracy= {}".format(ite, logloss, acc)
        print(msg1)
        logging.info(msg1)

        turn -= 1
        ite += 1

        if turn == 0:
            break

        # 6.1 calculate the derivative on the features
        grad = torch.autograd.grad(loss_test, features, retain_graph=True)[0]

        line_base = len(features) - n_nodes

        if use_steepest:
            norm_l2 = np.linalg.norm(grad[line_base:].numpy())
            lr_attack =  norm_l2 * 10000000
            # print("norm_l2", norm_l2)
            # print("lr_attack", lr_attack)


        for k in range(n_nodes):
            line = line_base + k
            for n in range(100):
                features[line, n] += grad[line][n] * lr_attack * lr_attack_ex

                if features[line, n] < min_:
                    features[line, n] = min_
                if features[line, n] > max_:
                    features[line, n] = max_  ##1.9999

        lr_attack_ex *= 0.95

        # write pickle
    if test_mode == False:
        last_attack_pred = accuracy_origin(output[idx_opt], labels[idx_opt], le, return_preds_only=True)
        last_attack_pred_path = os.path.join(output_path, "preds", "idx_test_attack_pred.pkl")
        with open(last_attack_pred_path, 'wb') as handle:
            pickle.dump(last_attack_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("store last attack predict for idx_test at", last_attack_pred_path)

    # write pickle
    raw_origin_pred_path = os.path.join(output_path, "preds", "raw_pred.pkl")
    with open(raw_origin_pred_path, 'wb') as handle:
        pickle.dump(raw_origin_pred_[idx_opt], handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("store raw predict for idx_test at ", raw_origin_pred_path)

    adj = adj.astype("float32").tocsr()
    features = features.cpu().detach().numpy()

    attacker_path = os.path.join(output_path, "attacker",
                                name.format(time_start) + "-" + str(max_logloss) + "-" + str(min_acc))


    os.makedirs(attacker_path)


    # np.save(os.path.join(attacker_path, 'feature.npy'), features)
    # with open(os.path.join(attacker_path, 'adj.pkl'), 'wb') as f:
    #     pickle.dump(adj, f)

    line_base = len(features) - n_nodes
    print("line_base:", line_base)
    if test_mode:
        adj_save = adj[line_base:, :]
        feature_save = features[line_base:]
    else:
        adj_save = adj
        feature_save = features

    print("adj_save.shape, feature_save.shape", adj_save.shape, feature_save.shape)
    np.save(os.path.join(attacker_path, 'feature.npy'), feature_save)
    with open(os.path.join(attacker_path, 'adj.pkl'), 'wb') as f:
        pickle.dump(adj_save, f)

    print("log path:", log_path)
    msg = "Done attack! Save at {}".format(attacker_path)
    print(msg)
    print(max_logloss, min_acc)

    logging.info("min_logloss: {}, min_acc: {}".format(max_logloss, min_acc))
    logging.info(msg)
    return attacker_path


def gradient_attack_steepest(adj, features, labels_origin, model_path, n_attack_epochs, idx_opt, output_path=None, n_nodes = 500):
    time_start = str(datetime.datetime.now())

    name = "gradient_attack_steepest_{}"
    log_path = os.path.join(output_path, "attack_log", name.format(time_start))

    logging.basicConfig(filename=log_path, level=logging.DEBUG)
    logging.info(f"log_path: {log_path}")

    adj_norm = normalize_adj_gcn(adj)

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    le = LabelEncoder()
    labels = le.fit_transform(labels_origin)
    labels = torch.LongTensor(labels)

    features = torch.FloatTensor(features)
    features.requires_grad_(True)

    model = torch.load(model_path)

    raw_output = model._predict(features, adj_norm)

    # write pickle
    raw_pred_path = os.path.join(output_path, "preds", "raw_pred_.pkl")
    with open(raw_pred_path, 'wb') as handle:
        pickle.dump(raw_output[idx_opt], handle, protocol=pickle.HIGHEST_PROTOCOL)

    raw_output = raw_output.max(1)[1]

    # write pickle
    preds_path = os.path.join(output_path, "preds", "idx_test_origin_pred_.pkl")
    with open(preds_path, 'wb') as handle:
        pickle.dump(le.inverse_transform(labels[idx_opt]), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 6 attacking on the features with FGA like method
    turn = n_attack_epochs
    ite=1

    max_logloss = -np.inf
    min_acc = np.inf
    while turn > 0:
        output = model._predict(features, adj_norm)
        loss_test = F.nll_loss(output[idx_opt], raw_output[idx_opt])

        acc_test = accuracy(output[idx_opt], labels[idx_opt])

        # 6.1 calculate the derivative on the features
        grad = torch.autograd.grad(loss_test, features, retain_graph=True)[0]

        line_base = len(features)  - n_nodes

        norm_l2 = np.linalg.norm(grad[line_base:].numpy())
        lr_attack = norm_l2*norm_l2

        for k in range(n_nodes):
            line = line_base + k
            for n in range(100):
                tmp = grad[line][n] * lr_attack
                features[line, n] += tmp

                if features[line, n] < -2:
                    features[line, n] = -1.9999
                if features[line, n] > 2:
                    features[line, n] = 1.9999
            if k%50 == 0:
                print(tmp)


        logloss, acc = loss_test.item(), acc_test.item()

        if max_logloss < logloss:
            max_logloss = logloss

        if min_acc > acc:
            min_acc = acc

        msg1 = "iter {}: loss= {}, accuracy= {};  lr_attack={}".format(ite, logloss, acc, lr_attack)
        print(msg1)
        logging.info(msg1)
        turn -= 1
        ite+=1

     # write pickle
    last_attack_pred = accuracy_origin(output[idx_opt], labels[idx_opt], le, return_preds_only=True)

    last_attack_pred_path = os.path.join(output_path, "preds", "idx_test_attack_pred_.pkl")
    with open(last_attack_pred_path, 'wb') as handle:
        pickle.dump(last_attack_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("store last attack predict for idx_test at", last_attack_pred_path)
    print("store raw predict for idx_test at ", raw_pred_path)
    print("store origin predict for idx_test at ", preds_path)

    adj = adj.astype("float32").tocsr()
    features = features.cpu().detach().numpy()

    attacker_path = os.path.join(output_path, "attacker", name.format(time_start) + "-" + str(max_logloss) + "-" + str(min_acc))
    os.makedirs(attacker_path)
    np.save(os.path.join(attacker_path, 'feature.npy'), features)
    with open(os.path.join(attacker_path, 'adj.pkl'), 'wb') as f:
        pickle.dump(adj, f)

    # os.makedirs(path)
    # np.save(os.path.join(path, 'features.npy'), features[line_base:])
    # with open(os.path.join(path, 'adj.pkl'), 'wb') as f:
    #     pickle.dump(adj[line_base:, :], f)

    msg = "Done attack! Save at {}".format(attacker_path)
    print(msg)
    print( max_logloss, min_acc )

    logging.info("min_logloss: {}, min_acc: {}".format(max_logloss, min_acc))
    logging.info(msg)

    return attacker_path

#### Run Attack

def run_attack(idx_test, data_path, n_nodes, model_path, seed, output_path, n_attack_epochs, attack_func, idx_test_half, test_mode, use_steepest):

    adj, features, labels_origin = load_raw_input(data_path)


    if test_mode:
        idx_test = list(range(609574, 609574+ 50000))
    list_nodes = pick_nodes_to_add_edges_randomly(idx_test, adj, seed = seed)

    # list_nodes = pick_nodes_to_add_edges_randomly_HALF_Cluster(idx_test_half, adj, seed=seed)

    new_adj = add_edges_to_adj(adj, list_nodes)

    random_range  = (-0.1, 0.1)
    added_features = initialize_features_randomly(n_nodes, random_range =random_range , seed=seed)
    new_feature = add_features(features, added_features)
    attacker_path = attack_func(new_adj, new_feature, labels_origin, model_path, n_attack_epochs, idx_test, output_path, test_mode=test_mode, use_steepest=use_steepest)

    return attacker_path




