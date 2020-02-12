#!/user/bin/env python3
# -*- coding: utf-8 -*-

from keras import Input, Model, losses, Sequential
from keras.activations import relu, sigmoid
from keras.layers import Dense, Bidirectional, LSTM, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
def build_model(maxlen):
    model = Sequential()
    # BiRNN train
    model.add(Bidirectional(LSTM(64), input_shape=(maxlen, 768)))
    # model.add(Flatten())
    # model.add(Dense(32, activation=relu,   input_dim=maxlen))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation=sigmoid))
    model.compile(loss=losses.binary_crossentropy, optimizer=Adam(1e-5), metrics=['accuracy'])
    return model
if __name__ == '__main__':
    print(build_model(maxlen=100).summary())