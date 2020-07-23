# encoding:utf-8
#############################################
# FileName: main.py
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: train and predict model
#############################################
import os
import json
import math
import numpy as np
import tensorflow as tf
import input_utils
import model
import config

FLAGS = config.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpuid

# 如果在集群上训练，需要导入环境变量
if FLAGS.run_on_cluster:
    cluster = json.loads(os.environ["TF_CONFIG"])
    task_index = int(os.environ["TF_INDEX"])
    task_type = os.environ["TF_ROLE"]


def main(unused_argv):
    classifier = tf.estimator.Estimator(model_fn=model.model_fn,
                                        config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir,
                                                                      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                                                                      keep_checkpoint_max=3),
                                        params={}
                                        )
    def train_eval_model():
        train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_utils.train_input_fn(FLAGS.train_data, FLAGS.batch_size),
                                            max_steps=FLAGS.train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_utils.eval_input_fn(FLAGS.eval_data, FLAGS.batch_size),
                                          start_delay_secs=60,
                                          throttle_secs = 30,
                                          steps=1000)
        tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    def train_model():
        #from tensorflow.python import debug as tf_debug
        #debug_hook = tf_debug.LocalCLIDebugHook()
        classifier.train(input_fn=lambda: input_utils.train_input_fn(FLAGS.train_data, FLAGS.batch_size), max_steps=FLAGS.train_steps)

    def export_model(feed_dict, export_dir):
        feature_map = dict()
        for key, value in feed_dict.items():
            feature_map[key] = tf.placeholder(dtype=tf.int64, shape=[None, value], name=key)
        serving_input_recevier_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_map)
        export_dir = classifier.export_saved_model(export_dir, serving_input_recevier_fn)
        print("pb model exported to %s"%export_dir)

    # 训练
    if FLAGS.train_eval:
        train_eval_model()

    # 导出query和doc模型，如果在集群上导出，需要设置在chief节点上
    if FLAGS.run_on_cluster:
        if task_type == "chief":
            if FLAGS.export_query_model:
                feed_dict = {"query_char": FLAGS.query_max_char_length}
                export_model(feed_dict, FLAGS.query_model_path)
            if FLAGS.export_doc_model:
                FLAGS.export_query_model = False
                feed_dict = {"doc_char": FLAGS.doc_max_char_length}
                export_model(feed_dict, FLAGS.doc_model_path)
    else:
        if FLAGS.export_query_model:
            feed_dict = {"query_char": FLAGS.query_max_char_length}
            export_model(feed_dict, FLAGS.query_model_path)
        if FLAGS.export_doc_model:
            FLAGS.export_query_model = False
            feed_dict = {"doc_char": FLAGS.doc_max_char_length}
            export_model(feed_dict, FLAGS.doc_model_path)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
