#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 2018/3/2
import collections
import json
import re
from multiprocessing import Pool
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score

def read_data(filename):
    """
    数据读取
    输出：
        多行文本字符串中的一行文本字符串
    功能：
        一百万行的大数据使用生成器模式逐行读取
    """
    with open(filename) as file1:  # 打开文本文件
        # str1 = file1.read().split(' ')  # 将文章按照空格划分开
        for line in file1:
            yield line

def data_clean():
    """
    数据清洗
    输出：
        拼接成的列表
    功能：
        读取文本字符串中的text所对应字符串
        清洗数据，删除#，@后字符串和标点符号
    """
    texts = []
    for line in read_data('data.txt'):
        text = json.loads(line)['text']
        # print(text)
        text = re.sub('(#.*?)\s+', '', text)
        text = re.sub('(@.*?)\s+', '', text)
        text = re.sub('[!.,?:]', '', text)
        texts.append(text)
    return texts


def words_frequency(data):
    """
    词频统计
    输出：
        各个单词以及出现次数，字典形式
    """
    b = collections.Counter(data.split())
    print("\n各单词出现的次数：\n %s" % b)



def create_vocal_list(data):
    """创建词汇表，词汇表包含了所有训练样本中出现的词（不包含停止词）"""
    vocab_list = set(data)
    return vocab_list


def text_to_vec(vocab_list, text):
    """把一组词转换成词向量，词向量的长度等于词汇表的长度"""
    text_vec = [0]*len(vocab_list)
    for word in text:
        if word in vocab_list:
            text_vec[vocab_list.index(word)] = 1 # 伯努利模型，不考虑重复出现的词，对于一条数据中的每个词，如果改词存在于词汇表中，则把词向量对应位置设为1
#             words_vec[vocab_list.index(word)] += 1 # 多项式模型，考虑重复出现的词
    return text_vec


def train(texts_vec, labels):
    # text_train, text_test, labels_train, labels_test = train_test_split(texts_vec, labels, test_size=0.3)
    svm = SVC()
    # svm.fit(text_train, labels_train)
    # svm.predict(text_test)

    cross_val_score(svm, texts_vec, labels) #3折交叉验证 输出一个[1x3]的矩阵


if __name__ == '__main__':
    import time
    start = time.time()
    text_list = data_clean() #数据清洗变成列表
    vocal_list_string = ' '.join(text_list)  # 将列表以空格连接起来
    words_frequency(vocal_list_string) #词频统计
    vocal_list = create_vocal_list(vocal_list_string) #列表去重
    print('列表去重', vocal_list_string)


    print('run time: ', time.time()-start)
