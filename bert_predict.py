# encoding = utf - 8
import codecs
from keras.preprocessing import sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from keras.models import load_model
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

def predict(bertvec):
    model = load_model('model/keras_bert.h5')
    # model.summary()
    return model.predict(bertvec)

if __name__ =='__main__':
    '''
    Bert + BiLSTM  to predict
    '''
    base_path = 'D:/bert_textcls/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12'
    config_path = '{}/bert_config.json'.format(base_path)
    checkpoint_path = '{}/bert_model.ckpt'.format(base_path)
    dict_path = '{}/vocab.txt'.format(base_path)
    maxlen = 100
    # load data
    content = '房间的环境可以啊'
    token_dict = get_token_dict(dict_path)
    # get_encode()
    encoder = get_encode(content, token_dict)
    # bert vec
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=maxlen)
    bert_vec = bert_model.predict(encoder)
    # model predict
    print(content)
    print(predict(bert_vec)[0][0])