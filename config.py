# coding:utf-8
#############################################
# FileName: config.py
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: configuration parameters
#############################################

import json, os, re, codecs
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_boolean("run_on_cluster", False, "Whether the cluster info need to be passed in as input")
flags.DEFINE_boolean("export_query_model", False, "Whether export model or not")
flags.DEFINE_boolean("export_doc_model", False, "Whether export model or not")
flags.DEFINE_boolean("train_eval", False, "Whether train and evaluate model or not")

flags.DEFINE_string("train_dir", "", "")
flags.DEFINE_string("data_dir", "", "")
flags.DEFINE_string("log_dir", "", "")
flags.DEFINE_string("ps_hosts", "","Comma-separated list of hostname:port pairs, you can also specify pattern like ps[1-5].example.com")
flags.DEFINE_string("worker_hosts", "","Comma-separated list of hostname:port pairs, you can also specify worker[1-5].example.co")
flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

flags.DEFINE_string("model_dir", "./ckpt/", "Base directory for the model.")
flags.DEFINE_string("query_model_path", "./query_model/", "Saved model.")
flags.DEFINE_string("doc_model_path", "./doc_model/", "Saved model.")

flags.DEFINE_string("train_data", "./train/*.tfrecord", "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "./eval/*.tfrecord", "Path to the evaluation data.")
flags.DEFINE_string("gpuid", "1", "gpuid")

flags.DEFINE_integer("rnn_hidden_size",64, "rnn_hidden_size")
flags.DEFINE_integer("query_max_char_length",50, "max_char_sentence_length")
flags.DEFINE_integer("doc_max_char_length",200, "max_char_sentence_length")
flags.DEFINE_integer("char_embedding_size",128, "char embedding size")
flags.DEFINE_integer("last_hidden_size",64, "last hidden size")

flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("char_vocab_size",10000, "char_vocab_size")
flags.DEFINE_integer("NEG",50, "negative samples")
flags.DEFINE_integer("train_steps",1000000, "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 512, "Training batch size")
flags.DEFINE_integer("save_checkpoints_steps", 500000, "Save checkpoints every this many steps")

FLAGS = flags.FLAGS



