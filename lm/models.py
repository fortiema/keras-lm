from keras.callbacks import TensorBoard
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, Input, GRU, TimeDistributed
from keras.models import Sequential


class LanguageModel:
    """

    """

    def __init__(self, vocab_size, embed_dim):
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, embed_dim, input_length=None))
        self.model.add(Bidirectional(GRU(128, return_sequences=True)))
        self.model.add(Dropout(0.25))
        self.model.add(Bidirectional(GRU(128, return_sequences=True)))
        self.model.add(Dropout(0.25))
        self.model.add(TimeDistributed(Dense(vocab_size, activation="softmax")))
        self.model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        print(self.model.summary())

    def fit(self, loader_train, loader_valid=None, bs=128, epochs=10):

        self.model.fit_generator(
            iter(loader_train),
            steps_per_epoch=len(loader_train) // 100,
            epochs=epochs,
            validation_data=iter(loader_valid),
            validation_steps=len(loader_valid),
            callbacks=[TensorBoard()],
        )
