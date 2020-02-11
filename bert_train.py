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
    return pos[:3000], neg[:3000]
# 得到编码
def get_encode(pos, neg, token_dict):
    all_data = pos + neg
    tokenizer = OurTokenizer(token_dict)
    X1 = []
    X2 = []
    for line in all_data:
        # character level train
        x1, x2 = tokenizer.encode(first=line)
        X1.append(x1)
        X2.append(x2)
    # x1 one-hot,x2 is position encoder
    X1 = sequence.pad_sequences(X1, maxlen=maxlen, padding='post', truncating='post')
    X2 = sequence.pad_sequences(X2, maxlen=maxlen, padding='post', truncating='post')
    return [X1, X2]
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
    pos, neg = load_data()
    token_dict = get_token_dict(dict_path)
    # get_encode()
    encoder = get_encode(pos, neg, token_dict)
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    bert_vec = bert_model.predict(encoder)
    # label make
    y = np.concatenate((np.ones(3000, dtype=int), np.zeros(3000, dtype=int)))
    print(len(y))
    print(len(bert_vec))
    # model train
    model_train(bert_vec, y)