import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os

class Dataset:
    def __init__(self, data_path, seed):
        self.seed = seed
        self.data_path = data_path
        self.adj, self.features, self.labels = self.load_raw_input()

    def load_raw_input(self):
        adj = np.load(os.path.join(self.data_path, 'adj.pkl'), allow_pickle=True)
        features = np.load(os.path.join(self.data_path, 'features.pkl'), allow_pickle=True)
        labels = np.load(os.path.join(self.data_path, 'train.pkl'), allow_pickle=True)
        return adj, features, labels

    def create_train_valid_test(self, _path, train_size, valid_size, test_size):
        idx = np.arange(len(self.labels))


        idx_train, idx_valid_test = train_test_split(idx,
                                                       random_state = self.seed,
                                                       train_size = train_size,
                                                       test_size = valid_size + test_size,
                                                       stratify = self.labels)

        stratify = self.labels[idx_valid_test]

        idx_valid, idx_test = train_test_split(idx_valid_test,
                                              random_state=self.seed,
                                              train_size=(valid_size / (test_size + valid_size)),
                                              test_size=(test_size / (test_size + valid_size)),
                                              stratify=stratify)

        # write pickle

        path = os.path.join(_path, f"train_valid_test_{self.seed}_{train_size}_{valid_size}_{test_size}")
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, 'idx_train.pkl'), 'wb') as handle:
            pickle.dump(idx_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'idx_valid.pkl'), 'wb') as handle:
            pickle.dump(idx_valid, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(path, 'idx_test.pkl'), 'wb') as handle:
            pickle.dump(idx_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("len of train/valid/test:", len(idx_train), len(idx_valid), len(idx_test))
        print(f"Done create_train_valid_test(). Saved file at {path}.")





