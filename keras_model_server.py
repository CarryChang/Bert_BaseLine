import codecs
import json
from keras.preprocessing import sequence
from keras_bert import Tokenizer, load_trained_model_from_checkpoint
from keras.models import load_model
from flask import request, Flask, jsonify
import tensorflow as tf
app = Flask(__name__)
def global_():
	# 全局定义和全局加载模型，提升inference速度
	global model, graph, bert_model, maxlen, dict_path, config_path, checkpoint_path, token_dict
	base_path = 'D:/bert_textcls/chinese_L-12_H-768_A-12/chinese_L-12_H-768_A-12'
	config_path = '{}/bert_config.json'.format(base_path)
	checkpoint_path = '{}/bert_model.ckpt'.format(base_path)
	dict_path = '{}/vocab.txt'.format(base_path)
	maxlen = 100
	bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=maxlen)
	model = load_model('model/keras_bert.h5')
	token_dict = {}
	with codecs.open(dict_path, 'r', 'utf-8') as reader:
		for line in reader:
			token = line.strip()
			token_dict[token] = len(token_dict)
	graph = tf.get_default_graph()
class OurTokenizer(Tokenizer):
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
def get_encode(content, token_dict):
	tokenizer = OurTokenizer(token_dict)
	onehot_encoding = []
	postion_encoding = []
	onehot, postion = tokenizer.encode(first=content)
	onehot_encoding.append(onehot)
	postion_encoding.append(postion)
	onehot_encoding = sequence.pad_sequences(onehot_encoding, maxlen=maxlen, padding='post', truncating='post')
	postion_encoding = sequence.pad_sequences(postion_encoding, maxlen=maxlen, padding='post', truncating='post')
	return [onehot_encoding, postion_encoding]

global_()
@app.route("/sentiment_analysis_api", methods=['POST'])
def predict():
	data = json.loads(request.get_data().decode('utf-8'))
	content = data['content']
	encoder = get_encode(content, token_dict)
	# set default
	with graph.as_default():
		result = {}
		bert_vec = bert_model.predict(encoder)
		result["content"] = content
		result["sa"] = '%f.4' % model.predict(bert_vec)[0][0]
	return jsonify(result)
if __name__ == "__main__":
	app.run()