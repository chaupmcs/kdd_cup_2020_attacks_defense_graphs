from GNN.MyGraphSAGE import MyGraphSAGE
from GNN.MyGCN import *
from Input_processing.read_input import *
import os
from attack.attack_evaluation import *
from utils.helper_functions import *

from attack import attack

if __name__ == '__main__':

    #### READ CONFIG
    params = read_config()



    #### CREATE TRAIN/VALID/TEST DATA
    data_path = os.path.join(params["path"], "data", params["phase"])
    tmp_path = os.path.join(params["path"], "output", "tmp", params["phase"])
    model_dump_path = os.path.join(params["path"], "output", "model_dump", params["phase"])

    if params["train_valid_test_folder"] == "":
        data =  Dataset(data_path, params["seed"])
        data.create_train_valid_test(tmp_path, params["train_size"], params["valid_size"], params["test_size"])
        train_valid_test_path = os.path.join(tmp_path,
                                f"train_valid_test_{params['seed']}_{params['train_size']}_{params['valid_size']}_{params['test_size']}")
    else:
        train_valid_test_path = os.path.join(tmp_path, params["train_valid_test_folder"])

    idx_train, idx_valid, idx_test, idx_test_half = read_train_valid_test(train_valid_test_path)


    #### TRAIN GNN MODELS
    ## GraphSAGE
    # graphSAGE_params = params["graphSAGE"]
    # graphSAGE = MyGraphSAGE(data_path,idx_train, idx_valid, idx_test, graphSAGE_params)
    # graphSAGE.run_model()


    gcn_params = params["GCN"]

    ## GCN
    # # run_GCN(False, data_path, idx_train, idx_valid, idx_test, gcn_params, model_dump_path)
    #
    model_path = "/Users/chaupham/PycharmProjects/KDD_attack_defense/output/model_dump/phase_2/gcn_model-normalize_adj_gcn-500-2020-07-17 23:58:34.435969-tensor(0.5513, dtype=torch.float64).pth"
    origin_data_path = "/Users/chaupham/PycharmProjects/KDD_attack_defense/data/phase_2"
    attack_log_output = os.path.join(params["path"], "output", "attack_log")
    attack_normalize = gcn_params["normalize"]

    n_nodes = 500
    seed = 2022
    output_path = os.path.join(params["path"], "output")

    n_attack_epochs = 30
    test_mode = True
    use_steepest = False  ##2.54
    attack.run_attack(idx_test, origin_data_path, n_nodes, model_path, seed, output_path, n_attack_epochs, attack.gradient_attack, idx_test_half, test_mode, use_steepest)





