#!/user/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/2/12 0012 12:09
# @Author  : CarryChang
# @Software: PyCharm
# @email: coolcahng@gmail.com
# @web ：CarryChang.top
from keras.models import load_model
# 不使用save_best_only=False的结果就是保留全部模型，打开就是保留权重
def load_whole_moel():
    # 使用整个model
    model = load_model('model/keras_bert.h5')
    print(model.summary())
def load_weigt():
    weight_path = 'model/weight/keras_bert.json'
    # 从json文件中加载模型,json只是保存权重参数
    with open(weight_path, 'r', encoding='utf-8') as file:
        model_json = file.read()
    print(model_json)
if __name__ == '__main__':
    load_weigt()
