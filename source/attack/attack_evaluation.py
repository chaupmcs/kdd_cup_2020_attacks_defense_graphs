import dill
import os
import tensorflow.keras as keras
from GNN.GCN import GCN
from utils.helper_functions import path_filename
import ntpath
import torch
import logging
# def evaluate_attack(model_path, input_paths, output_paths):
#
#     with open(os.path.join(model_path, 'generator.pkl'), 'rb') as f:
#         generator = dill.load(f)
#
#     with open(os.path.join(model_path, 'test_labels.pkl'), 'rb') as f:
#         test_labels = dill.load(f)
#
#     with open(os.path.join(model_path, 'test_targets.pkl'), 'rb') as f:
#         test_targets = dill.load(f)
#
#     reconstructed_model = keras.models.load_model("./my_model2")
#     test_gen = generator.flow(test_labels.index, test_targets)
#
#     test_metrics = reconstructed_model.evaluate(test_gen)
#     print("\nTest Set Metrics:")
#     for name, val in zip(reconstructed_model.metrics_names, test_metrics):
#         print("\t{}: {:0.4f}".format(name, val))
#
#
#     return None
from source.utils.helper_functions import get_origin_metric


def evaluate_attack_GCN(model_path, idx_test, data_path, output_path, normalize, use_cuda=False):

    attack_log_filename = path_filename(data_path)
    attack_log_path = os.path.join(output_path, attack_log_filename + ".log")

    logging.basicConfig(filename=attack_log_path, level=logging.DEBUG)
    logging.info(f"model_path: {model_path}")
    logging.info(f"data_path: {data_path}")



    tmp = model_path.replace(model_path.split("-")[-1], "")[:-1]
    model_name = path_filename(tmp)

    log_path = os.path.join(tmp.replace(model_name, ""), "log", model_name) + ".log"

    # print("log_path = ", log_path)

    model = torch.load(model_path)

    if use_cuda:
        model = model.to("cuda:1")

    # model = GCN()
    # model.load_state_dict(torch.load(model_path))
    # model.eval()


    acc_after_attack, logloss_after_attack = model.test_attack(normalize, idx_test, data_path)

    acc_before_attack, logloss_before_attack = get_origin_metric(log_path)
    msg_1 = '=== acc_AFTER_attack={:.4f}, logloss_AFTER_attack={:.4f} ==='.format(acc_after_attack, logloss_after_attack)
    msg_2 = 'acc_BEFORE_attack={}, logloss_BEFORE_attack={}'.format(acc_before_attack, logloss_before_attack)

    print(msg_2)
    print(msg_1)

    logging.info(msg_2)
    logging.info(msg_1)

    return None


