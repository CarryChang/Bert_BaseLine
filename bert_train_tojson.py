# encoding = utf - 8
import codecs
import numpy as np
import os
from keras.preprocessing import sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from model import build_model
from keras.models import model_from_json
class CTokenizer(Tokenizer):
    def _tokenize(self, text):
        tokenize_dic = []
        for character in text:
            if character in self._token_dict:
                tokenize_dic.append(character)
            elif self._is_space(character):
                tokenize_dic.append('[unused1]')
            else:
                tokenize_dic.append('[UNK]')
        return tokenize_dic
def get_token_dict(dict_path):
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict
def load_data():
    pos = []
    neg = []
    with codecs.open('data/pos_all.txt', 'r', 'utf-8') as reader:
        for line in reader:
            pos.append(line.strip())
    with codecs.open('data/neg_all.txt', 'r', 'utf-8') as reader:
        for line in reader:
            neg.append(line.strip())
    return pos[:train_number], neg[:train_number]
def get_encode(pos, neg, token_dict):
    data_encoder = pos + neg
    tokenizer = CTokenizer(token_dict)
    one_hot_encoder = []
    position_encoder = []
    for line in data_encoder:
        one_hot, position = tokenizer.encode(first=line)
        one_hot_encoder.append(one_hot)
        position_encoder.append(position)
    one_hot_encoder = sequence.pad_sequences(one_hot_encoder, maxlen=maxlen, padding='post', truncating='post')
    position_encoder = sequence.pad_sequences(position_encoder, maxlen=maxlen, padding='post', truncating='post')
    return [one_hot_encoder, position_encoder]
def model_train(bertvec,y):
    model = build_model(maxlen)
    model.summary()
    # checkpoit = ModelCheckpoint(filepath=os.path.join('model/check_point/', 'model-{epoch:02d}.h5'))
    best_model_path = 'model/weight/keras_bert.hdf5'
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    saveBestModel = ModelCheckpoint(best_model_path, save_weights_only=True, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    tensorboard = TensorBoard(log_dir='tensorboard', histogram_freq=0, write_graph=True, write_grads=False, write_images=True)
    model.fit(bertvec, y, batch_size=64, epochs=5, validation_split=0.2, shuffle=True,
              callbacks=[tensorboard, earlyStopping, saveBestModel])
if __name__ =='__main__':
    '''
    Bert+ BiLSTM to make text_classify
    '''
    base_path = 'D:/bert_textcls/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12'
    config_path = '{}/bert_config.json'.format(base_path)
    checkpoint_path = '{}/bert_model.ckpt'.format(base_path)
    dict_path = '{}/vocab.txt'.format(base_path)
    maxlen = 100
    train_number = 1000
    pos, neg = load_data()
    token_dict = get_token_dict(dict_path)
    # get_encode()
    encoder = get_encode(pos, neg, token_dict)
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    bert_vec = bert_model.predict(encoder)
    # label make
    y = np.concatenate((np.ones(train_number, dtype=int), np.zeros(train_number, dtype=int)))
    print(len(y))
    print(len(bert_vec))
    # model train
    model_train(bert_vec, y)