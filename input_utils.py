#encoding:utf-8
#############################################
# FileName: input_utils
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: input utils
#############################################
import os, json, codecs
import tensorflow as tf
import config

FLAGS = config.FLAGS

def parse_exp(example):
    features_def = dict()
    features_def["label"] = tf.io.FixedLenFeature([1], tf.int64)
    features_def["query_char"] = tf.io.FixedLenFeature([FLAGS.query_max_char_length], tf.int64)
    features_def["doc_char"] = tf.io.FixedLenFeature([FLAGS.doc_max_char_length], tf.int64)
    features = tf.io.parse_single_example(example, features_def)
    label = features.pop("label")
    return features, label


def train_input_fn(filenames=None,
                   batch_size=128,
                   shuffle_buffer_size=1000):
    # 集群上训练需要切分数据
    if FLAGS.run_on_cluster:
        files_all = tf.gfile.Glob(filenames)
        train_worker_num = len(FLAGS.worker_hosts.split(","))
        hash_id = FLAGS.task_index if FLAGS.job_name == "worker" else train_worker_num - 1
        files_shard = [files for i, files in enumerate(files_all) if i % train_worker_num == hash_id]
        dataset = tf.data.TFRecordDataset(files_shard)
    else:
        files = tf.gfile.Glob(filenames)
        dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.shuffle(batch_size*10)
    dataset = dataset.map(parse_exp, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=True).repeat().prefetch(1)
    return dataset

def eval_input_fn(filenames=None,
                  batch_size=128):
    files = tf.gfile.Glob(filenames)
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(parse_exp, num_parallel_calls=4)
    dataset = dataset.batch(batch_size, drop_remainder=True).repeat()
    return dataset

