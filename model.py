# coding:utf-8
#############################################
# FileName: model.py
# Author: ChenDajun
# CreateTime: 2020-06-12
# Descreption: model
#############################################

import random
import tensorflow as tf
import config

FLAGS = config.FLAGS

def compute_seq_length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

def rnn_encoder(inputs, reuse, scope_name):
    with tf.variable_scope(scope_name, reuse=reuse):
        GRU_cell_fw = tf.contrib.rnn.GRUCell(FLAGS.rnn_hidden_size)
        GRU_cell_bw = tf.contrib.rnn.GRUCell(FLAGS.rnn_hidden_size)
        ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                             cell_bw=GRU_cell_bw,
                                                                             inputs=inputs,
                                                                             sequence_length=compute_seq_length(inputs),
                                                                             dtype=tf.float32)
        outputs = tf.concat((fw_outputs, bw_outputs), axis=2)
        outputs = tf.nn.tanh(outputs)
        return outputs

def attention_layer(inputs, reuse, scope_name, outname):
    with tf.variable_scope(scope_name, reuse=reuse):
        u_context = tf.Variable(tf.truncated_normal([FLAGS.rnn_hidden_size * 2]), name=scope_name+ '_u_context')
        h = tf.contrib.layers.fully_connected(inputs, FLAGS.rnn_hidden_size * 2, activation_fn=tf.nn.tanh)
        alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keepdims=True), axis=1)
    attn_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1, name=outname)
    # attn_output = tf.nn.tanh(attn_output)
    return attn_output

def word_embedding(inputs, reuse=None, vocab_size=10000, embedding_size=128, scope_name="char_embedding"):
    with tf.variable_scope(scope_name, reuse=reuse):
        embedding_matrix = tf.Variable(tf.truncated_normal((vocab_size, embedding_size)))
        embedding = tf.nn.embedding_lookup(embedding_matrix, inputs, name=scope_name + "_layer")
    embedding = tf.nn.tanh(embedding)
    return embedding

def sentence_embedding(inputs, reuse=None, max_sentence_length=50, scope_name="char_sent"):
    with tf.variable_scope(scope_name, reuse=reuse):
        embedding = tf.reshape(inputs, [-1, max_sentence_length, FLAGS.char_embedding_size])
        word_encoder = rnn_encoder(embedding, reuse, scope_name=scope_name + "_encoder_layer")
        sent_encoder = attention_layer(word_encoder, reuse=reuse, scope_name=scope_name+"_attention_layer", outname=scope_name+"_vec")
        return sent_encoder

def build_query_model(features, mode):
    # 输入shape: [batch_size, sentence_size]
    char_input = tf.reshape(features["query_char"], [-1, FLAGS.query_max_char_length])
    char_embed = word_embedding(char_input, None, FLAGS.char_vocab_size, FLAGS.char_embedding_size, "char_embedding")
    sent_encoder = sentence_embedding(char_embed,
                                      None,
                                      FLAGS.query_max_char_length,
                                      "char_sent")
    sent_encoder = tf.layers.dense(sent_encoder, units=FLAGS.last_hidden_size, activation=tf.nn.tanh, name="query_encoder")
    sent_encoder = tf.nn.l2_normalize(sent_encoder)
    return sent_encoder

def build_doc_model(features, mode):
    # 输入shape: [batch_size, sentence_size]
    char_input = tf.reshape(features["doc_char"], [-1, FLAGS.doc_max_char_length])
    char_embed = word_embedding(char_input, True, FLAGS.char_vocab_size, FLAGS.char_embedding_size, "char_embedding")
    sent_encoder = sentence_embedding(char_embed,
                                      True if mode==tf.estimator.ModeKeys.TRAIN else tf.AUTO_REUSE,
                                      FLAGS.doc_max_char_length,
                                      "char_sent")
    sent_encoder = tf.layers.dense(sent_encoder, units=FLAGS.last_hidden_size, activation=tf.nn.tanh, name="doc_encoder")
    sent_encoder = tf.nn.l2_normalize(sent_encoder)
    return sent_encoder


def model_fn(features, labels, mode, params):
    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        if FLAGS.export_query_model:
            query_encoder = build_query_model(features, mode)
            predictions = {"query_vector": query_encoder}
        elif FLAGS.export_doc_model:
            doc_encoder = build_doc_model(features, mode)
            predictions = {"doc_vector": doc_encoder}
        export_outputs = {"predictions": tf.estimator.export.PredictOutput(outputs=predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    query_encoder = build_query_model(features, mode)
    doc_encoder = build_doc_model(features, mode)

    with tf.name_scope("fd-rotate"):
        tmp = tf.tile(doc_encoder, [1, 1])
        doc_encoder_fd = doc_encoder
        for i in range(FLAGS.NEG):
            rand = random.randint(1, FLAGS.batch_size + i) % FLAGS.batch_size
            s1 = tf.slice(tmp, [rand, 0], [FLAGS.batch_size - rand, -1])
            s2 = tf.slice(tmp, [0, 0], [rand, -1])
            doc_encoder_fd = tf.concat([doc_encoder_fd, s1, s2], axis=0)
        query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_encoder), axis=1, keepdims=True)), [FLAGS.NEG + 1, 1])
        doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_encoder_fd), axis=1, keepdims=True))
        query_encoder_fd = tf.tile(query_encoder, [FLAGS.NEG + 1, 1])
        prod = tf.reduce_sum(tf.multiply(query_encoder_fd, doc_encoder_fd), axis=1, keepdims=True)
        norm_prod = tf.multiply(query_norm, doc_norm)
        cos_sim_raw = tf.truediv(prod, norm_prod)
        cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [FLAGS.NEG + 1, -1])) * 20

    with tf.name_scope("loss"):
        prob = tf.nn.softmax(cos_sim)
        hit_prob = tf.slice(prob, [0, 0], [-1, 1])
        loss = -tf.reduce_mean(tf.log(hit_prob))
        correct_prediction = tf.cast(tf.equal(tf.argmax(prob, 1), 0), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    # Eval
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={})

    # Train
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_global_step()
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=global_step)
        #train_op = tf.train.AdagradOptimizer(FLAGS.learning_rate).minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
