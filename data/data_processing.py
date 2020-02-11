#!/user/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/2/11 0011 16:10
# @Author  : CarryChang
# @Software: PyCharm
# @email: coolcahng@gmail.com
# @web ：CarryChang.top
# pos
def pos_():
    pos_file = open('pos_all.txt', 'w', encoding='utf-8')
    count_pos = []
    for content_pos in open('pos1.txt', 'r', encoding='utf-8').readlines():
        count_pos.append(content_pos.strip())
    for content_pos1 in open('pos.txt', 'r', encoding='utf-8').readlines():
        count_pos.append(content_pos1.strip())
    for filter in set(count_pos):
        pos_file.write(filter.strip() + '\n')
# neg
def neg_():
    neg_file = open('neg_all.txt', 'w', encoding='utf-8')
    count_neg = []
    for content_neg in open('neg1.txt', 'r', encoding='utf-8').readlines():
        count_neg.append(content_neg.strip())
    for content_neg1 in open('neg.txt', 'r', encoding='utf-8').readlines():
        count_neg.append(content_neg1.strip())
    for filter in set(count_neg):
        neg_file.write(filter.strip() + '\n')
if __name__ == '__main__':
    pos_()
    neg_()