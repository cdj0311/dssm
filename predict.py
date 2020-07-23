# coding:utf-8
#############################################
# FileName: predict.py
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: get sentence vector
#############################################
import numpy as np
import codecs
import json
import os
from scipy.spatial import distance
from tensorflow.contrib import predictor
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 读取字典文件
vocab = json.load(codecs.open("./char.json", "r", "utf-8"))
# 这里只用query做测试
model_name = "./query_model"
# 句子最大长度
max_size = 50

def sent2id(sent):
    # 将句子转换为id序列
    sent = [vocab.get(c, 1) for c in sent]
    sent = sent[:max_size] + [0] * (max_size - len(sent))
    return sent

def load_model():
    # 读取模型
    model_time = str(max([int(i) for i in os.listdir(model_name) if len(i)==10]))
    model = predictor.from_saved_model(os.path.join(model_name,model_time))
    return model

def get_vector(sentence, model):
    # 输入句子并转换为向量
    feed_dict = {"query_char": [sent2id(sentence)]}
    vector = model(feed_dict)
    return vector["query_vector"][0]

def similar_index(sentence, file_path, max_sentence_num=10000, topn=10):
    # 输入一个句子和包含一系列句子的文件，找到文件中跟该句子最相似的N条
    model = load_model()
    source_vec = get_vector(sentence, model)

    target_vec = dict()
    with codecs.open(file_path, "r", "utf-8") as fr:
        for line in tqdm(fr):
            line = line.strip().split("\t")
            vec = get_vector(line[0], model)
            target_vec[line[0]] = 1.0 - distance.cosine(source_vec, vec)
    rank = sorted(target_vec.items(), key=lambda e:e[1], reverse=True)
    print("source: %s"%sentence)
    print("target: \n")
    for i in rank[:topn]:
        print("%s\t%s"%(round(i[1],6), i[0]))

similar_index(u"赵丽颖冯绍峰在拍女儿国的时候真的超级甜了",
              "./data/data.txt",
              10000,
              10)
