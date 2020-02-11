# encoding = utf - 8
import codecs
import numpy as np
import yaml
import keras
from keras import Input, Model, losses, Sequential
from keras.activations import relu, sigmoid
from keras.layers import Dense, Bidirectional, LSTM
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from model import build_model

def get_token_dict(dict_path):
    '''
    :param: dict_path: 是bert模型的vocab.txt文件
    :return:将文件中字进行编码
    '''
    # 将bert模型中的 字 进行编码
    # 目的是 喂入模型的是这些编码，不是汉字
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

class OurTokenizer(Tokenizer):
    '''
    关键在  Tokenizer 这个类，要实现这个类中的方法，其实不实现也是可以的
    目的是 扩充 vocab.txt文件的
    '''
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')
            else:
                R.append('[UNK]')
        return R
def get_data():
    '''
    读取数据的函数
    :return: list  类型的 数据
    '''
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
        # 本数据集是  都是按照第一句，即一行数据即是一句，也就是第一句
        # 返回的x1,是经过编码过后得到，纯整数集合
        # 返回的x2,源码中segment_ids，表示区分第一句和第二句的位置。结果为：[0]*first_len+[1]*sencond_len
        # 本数据集中，全是以字来分割的。
        x1, x2 = tokenizer.encode(first=line)
        X1.append(x1)
        X2.append(x2)
    # x1 one-hot,x2 is position encoder
    X1 = sequence.pad_sequences(X1, maxlen=maxlen, padding='post', truncating='post')
    X2 = sequence.pad_sequences(X2, maxlen=maxlen, padding='post', truncating='post')
    return [X1, X2]

def load_bert_model(X1,X2):
    '''
    :param X1:经过编码过后的集合
    :param X2:经过编码过后的位置集合
    :return:模型
    '''
    # 加载  Google 预训练好的模型bert模型
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    # config_path 是Bert模型的参数，checkpoint_path 是Bert模型的最新点，即训练的最新结果
    bertvec = bert_model.predict([X1, X2])
    return bertvec

def model_train(bertvec,y):
    model = build_model(maxlen)
    model.summary()
    model.fit(bertvec, y, batch_size=32, epochs=10, validation_split=0.2, shuffle=True)
    # yaml_string = model.to_yaml()
    # with open('test_keras_bert.yml', 'w') as f:
    #     f.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save('model/keras_bert.h5')

if __name__ =='__main__':
    '''
    Bert+ BiLSTM to make text_classify
    '''
    base_path = 'D:/bert_textcls/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12'
    config_path = '{}/bert_config.json'.format(base_path)
    checkpoint_path = '{}/bert_model.ckpt'.format(base_path)
    dict_path = '{}/vocab.txt'.format(base_path)
    # 定义句子的最大长度，padding要用的
    maxlen = 100
    # load data
    pos, neg = get_data()
    token_dict = get_token_dict(dict_path)
    # get_encode()
    [X1, X2] = get_encode(pos, neg, token_dict)
    # bert vec
    bert_vec = load_bert_model(X1, X2)
    # label make
    y = np.concatenate((np.ones(3000, dtype=int), np.zeros(3000, dtype=int)))
    print(len(y))
    print(len(bert_vec))
    # model train
    model_train(bert_vec, y)