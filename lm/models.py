import json
from pathlib import Path

from keras import optimizers
from keras.callbacks import TensorBoard
from keras.layers import (
    Activation,
    Bidirectional,
    CuDNNLSTM,
    Dense,
    Dropout,
    Embedding,
    initializers,
    Input,
    GRU,
    LSTM,
    TimeDistributed,
)
from keras.models import Sequential
import numpy as np

from layers import TiedEmbeddingsTransposed


class LanguageModel:
    """

    """

    def __init__(self, vocab_size, embed_dim, log_dir="./logs"):
        self.log_dir = log_dir
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embed_dim, input_length=None))
        self.model.add(Bidirectional(GRU(256, return_sequences=True)))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(GRU(256, return_sequences=True)))
        self.model.add(Dropout(0.3))
        self.model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(self.model.summary())

    @property
    def fname_model(self):
        return "model.json"

    @property
    def fname_weights(self):
        return "weights.json"

    def fit(self, loader_train, loader_valid=None, bs=64, epochs=10):

        self.model.fit_generator(
            iter(loader_train),
            steps_per_epoch=len(loader_train) // 10,
            epochs=epochs,
            validation_data=iter(loader_valid),
            validation_steps=len(loader_valid),
            callbacks=[TensorBoard(log_dir=self.log_dir)],
        )

    def eval(self, loader_test):

        self.model.evaluate_generator(iter(loader_test), steps=len(loader_test))

    def save(self, path):
        _path = Path(path)
        if _path.is_dir():
            with open(_path / self.fname_model, "w") as f:
                f.write(self.model.to_json())
            self.model.save_weights(_path / self.fname_weights)


class AWDLSTMLanguageModel:
    """ AWD-LSTM Language Model, based on Merity, Keskar and Socher. 2017

    Notes:

        In order to be able to use the same embedding weights both at input and output (weight tying), the last
        layer from the encoder must have an output dimensionality matching the embeddings.

    Refs:

        * [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182)

    """

    def __init__(self, vocab_size, embed_dim, hidden_size=256, log_dir="./logs"):
        self.log_dir = log_dir

        init_embed = initializers.RandomUniform(minval=-0.1, maxval=0.1, seed=None)
        init_lstm_norm = 1 / np.sqrt(hidden_size)
        init_lstm = initializers.RandomUniform(minval=-init_lstm_norm, maxval=init_lstm_norm)

        self.model = Sequential()
        self.model.add(
            Embedding(vocab_size, embed_dim, input_length=None, name="embedding", embeddings_initializer=init_embed)
        )
        self.model.add(Dropout(0.2))
        self.model.add(
            Bidirectional(
                CuDNNLSTM(
                    hidden_size, return_sequences=True, kernel_initializer=init_lstm, recurrent_initializer=init_lstm
                )
            )
        )
        self.model.add(Dropout(0.3))
        self.model.add(
            Bidirectional(
                CuDNNLSTM(
                    hidden_size, return_sequences=True, kernel_initializer=init_lstm, recurrent_initializer=init_lstm
                )
            )
        )
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(CuDNNLSTM(embed_dim // 2, return_sequences=True)))
        self.model.add(Dropout(0.3))
        self.model.add(
            TimeDistributed(
                TiedEmbeddingsTransposed(tied_to=self.model.get_layer(name="embedding"), activation="softmax")
            )
        )
        self.optim = optimizers.SGD(lr=1.0, decay=1e-4, momentum=0.9, nesterov=True, clipnorm=0.5)
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer=self.optim, metrics=["accuracy"])

        print(self.model.summary())

    @property
    def fname_model(self):
        return "model.json"

    @property
    def fname_optim(self):
        return "optim.json"

    @property
    def fname_weights(self):
        return "weights.json"

    def fit(self, loader_train, loader_valid=None, bs=64, epochs=10):

        self.model.fit_generator(
            iter(loader_train),
            steps_per_epoch=len(loader_train) // 20,
            epochs=epochs,
            validation_data=iter(loader_valid),
            validation_steps=len(loader_valid),
            callbacks=[TensorBoard(log_dir=self.log_dir)],
        )

    def eval(self, loader_test):

        self.model.evaluate_generator(iter(loader_test), steps=len(loader_test))

    def save(self, path):
        _path = Path(path)
        if _path.is_dir():
            with open(_path / self.fname_model, "w") as f:
                f.write(self.model.to_json())
            with open(_path / self.fname_optim, "w") as f:
                f.write(self.optim.to_json())
            self.model.save_weights(_path / self.fname_weights)
