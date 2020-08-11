
import os
import pickle
import scipy.sparse as sp
from GNN.GCN import *
import torch
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder

import datetime


def run_GCN(use_cuda, data_path, idx_train, idx_val, idx_test, params, model_dump_path):
    time_start = str(datetime.datetime.now())

    seed = params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if use_cuda:
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')

    # 2 read and processing data
    adj, features, labels = load_raw_input(data_path)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    prefix = 'gcn_model' + "-" + params["normalize"] + "-" + \
             str(params['n_epochs']) + "-" + time_start
    log_path = os.path.join(model_dump_path, "log", prefix + '.log')



    model = GCN(nfeat=features.shape[1], lr=params['lr'], nclass=params["num_classes"], nhid=params['num_hidden'],
                dropout=params['dropout'], weight_decay=params['weight_decay'], n_epochs = params['n_epochs'],
                early_stopping = params['early_stopping'], device=device, log_path=log_path, verbose = params['verbose'])

    model = model.to(device)
    model.fit(features, adj, labels, idx_train, idx_val, normalize=params["normalize"])
    # setattr(victim_model, 'norm_tool',  GraphNormTool(normalize=True, gm='gcn', device=device))

    # 4 validation on test data
    acc_test, loss_test = model.test(idx_test)

    model_path =  os.path.join(model_dump_path, prefix + "-" + str(acc_test) + '.pth' )

    # torch.save(model.state_dict(), model_path)
    torch.save(model, model_path)

    print("save model at ", model_path)