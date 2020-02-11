# encoding = utf - 8
import codecs
import numpy as np
from keras.preprocessing import sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from model import build_model
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
    with codecs.open('data/pos.txt', 'r', 'utf-8') as reader:
        for line in reader:
            pos.append(line.strip())
    with codecs.open('data/neg.txt', 'r', 'utf-8') as reader:
        for line in reader:
            neg.append(line.strip())
    return pos[:train_number], neg[:train_number]
# 得到编码
def get_encode(content, token_dict):
    tokenizer = CTokenizer(token_dict)
    onehot_encoding = []
    postion_encoding = []
    onehot, postion = tokenizer.encode(first=content)
    onehot_encoding.append(onehot)
    postion_encoding.append(postion)
    onehot_encoding = sequence.pad_sequences(onehot_encoding, maxlen=maxlen, padding='post', truncating='post')
    postion_encoding = sequence.pad_sequences(postion_encoding, maxlen=maxlen, padding='post', truncating='post')
    return [onehot_encoding, postion_encoding]
def model_train(bertvec,y):
    model = build_model(maxlen)
    model.summary()
    model.fit(bertvec, y, batch_size=32, epochs=10, validation_split=0.2, shuffle=True)
    model.save('model/keras_bert.h5')
if __name__ =='__main__':
    '''
    Bert+ BiLSTM to make text_classify
    '''
    base_path = 'D:/bert_textcls/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12'
    config_path = '{}/bert_config.json'.format(base_path)
    checkpoint_path = '{}/bert_model.ckpt'.format(base_path)
    dict_path = '{}/vocab.txt'.format(base_path)
    maxlen = 100
    train_number = 10000
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