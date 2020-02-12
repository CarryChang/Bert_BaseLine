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
    from model import build_model
    model = build_model(maxlen=100)
    # 按照层数进行
    model.load_weights('model/keras_bert_weight.h5')
    print(model.summary())
if __name__ == '__main__':
    # load_whole_moel()
    # 只加载weight
    load_weigt()