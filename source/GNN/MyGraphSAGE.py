import pandas as pd
import numpy as np
import stellargraph as sg
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE

from tensorflow.keras import layers, optimizers, losses, Model
from sklearn import preprocessing
from stellargraph import StellarGraph
import os

class MyGraphSAGE:
    def __init__(self, data_path, idx_train, idx_valid, idx_test, hyperparams):
        self.data_path = data_path
        self.idx_train = idx_train
        self.idx_valid = idx_valid
        self.idx_test = idx_test
        self.hyperparams = hyperparams


    def prepare_data_for_stellargraph(self):
        def load_raw_input():
            adj = np.load(os.path.join(self.data_path, 'adj.pkl'), allow_pickle=True)
            features = np.load(os.path.join(self.data_path, 'features.pkl'), allow_pickle=True)
            labels = np.load(os.path.join(self.data_path, 'train.pkl'), allow_pickle=True)
            return adj, features, labels

        print("Reading raw inputs...")
        adj, features, labels = load_raw_input()

        print("creating nodes...")
        adj_list = [[i, j, adj[i, j]] for i, j in zip(*adj.nonzero())]
        tmp_df = pd.DataFrame(adj_list)
        tmp_df.columns = ["source", "target", "weight"]

        print("creating edges...")
        feature_df = pd.DataFrame(features)
        feature_df.columns = [f"w{i}" for i in range(feature_df.shape[1])]

        print("creating labels...")
        label_series = pd.DataFrame({"label": labels})["label"]

        my_graph = StellarGraph(
            {"paper": feature_df}, {"cites": tmp_df}
        )

        print(my_graph.info())
        return my_graph, label_series

    def get_train_valid_test(self, labels):
        train_labels, valid_labels, test_labels = labels[self.idx_train], labels[self.idx_valid], labels[self.idx_test]

        target_encoding = preprocessing.LabelBinarizer()
        train_targets = target_encoding.fit_transform(train_labels)
        valid_targets = target_encoding.transform(valid_labels)
        test_targets = target_encoding.transform(test_labels)

        return train_targets, valid_targets, test_targets, train_labels, valid_labels, test_labels

    def run_model(self):
        graph_sampled, label_series_sampled = self.prepare_data_for_stellargraph()
        train_targets, valid_targets, test_targets, train_labels, valid_labels, test_labels = self.get_train_valid_test(label_series_sampled)

        batch_size = self.hyperparams["batch_size"]
        num_samples = self.hyperparams["num_samples"]
        generator = GraphSAGENodeGenerator(graph_sampled, batch_size, num_samples)
        train_gen = generator.flow(train_labels.index, train_targets, shuffle=True)
        graphsage_model = GraphSAGE(
            layer_sizes=self.hyperparams["layer_sizes"], generator=generator, bias=self.hyperparams["bias"], dropout=self.hyperparams["dropout"],
        )
        x_inp, x_out = graphsage_model.in_out_tensors()
        prediction = layers.Dense(units=train_targets.shape[1], activation="softmax")(x_out)

        model = Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=optimizers.Adam(lr=self.hyperparams["lr"]),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )

        valid_gen = generator.flow(valid_labels.index, valid_targets)

        history = model.fit(
            train_gen, epochs=self.hyperparams["n_epochs"], validation_data=valid_gen, verbose=self.hyperparams["verbose"],
            shuffle=True, use_multiprocessing=True,
        )

        sg.utils.plot_history(history)

        test_gen = generator.flow(test_labels.index, test_targets)
        test_metrics = model.evaluate(test_gen)
        print("\nTest Set Metrics:")
        for name, valid in zip(model.metrics_names, test_metrics):
            print("\t{}: {:0.4f}".format(name, valid))


