# coding:utf-8
#############################################
# FileName: convert_data.py
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: convert tfrecord
#############################################
import json
import codecs
import numpy as np
import tensorflow as tf
from tqdm import tqdm

"""
转换为tfrecord格式
"""
def create_int_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def sent2id(sent, vocab, max_size):
    sent = [vocab[c] for c in sent if c in vocab]
    sent = sent[:max_size] + [0]*(max_size - len(sent))
    return sent

def convert_tfrecord(in_file, out_file, vocab_path, query_size=50, doc_size=200):
    vocab = json.load(codecs.open(vocab_path, "r", "utf-8"))
    writer = tf.io.TFRecordWriter(out_file)
    icount = 0
    with codecs.open(in_file, "r", "utf-8") as fr:
        for line in tqdm(fr):
            icount += 1
            line = line.strip().split("\t")
            query = sent2id(line[0], vocab, query_size)
            doc = sent2id(line[1], vocab, doc_size)

            feed_dict = {"query_char": create_int_feature(query),
                         "doc_char": create_int_feature(doc),
                         "label": create_int_feature([1])}
            example = tf.train.Example(features=tf.train.Features(feature=feed_dict))
            serialized = example.SerializeToString()
            writer.write(serialized)
    print(icount)
    writer.close()



if __name__ == "__main__":
    convert_tfrecord("./data/data.txt", 
                     "./data/train.tfrecord", 
                     "./char.json", 
                     query_size=50, 
                     doc_size=200)
