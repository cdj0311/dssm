#!/bin/sh
#############################################
# FileName: train_local.sh
# Author: cdj0311
# CreateTime: 2020-06-12
# Descreption: train script
#############################################


ckpt_dir=./ckpt
query_model_dir=./query_model
doc_model_dir=./doc_model

train_data=./data/*.tfrecord
eval_data=./data/*.tfrecord  # 这里评估数据没有用，直接设置为训练数据
train_steps=100000  # 训练步骤
batch_size=512
learning_rate=0.001
save_steps=100000
char_embedding_size=200
rnn_hidden_size=128
query_max_char_length=50
doc_max_char_length=200
last_hidden_size=64
char_vocab_size=18672
NEG=50
gpuid=3

python main.py \
     --train_data=${train_data} \
     --eval_data=${eval_data} \
     --model_dir=${ckpt_dir} \
     --query_model_path=${query_model_dir} \
     --doc_model_path=${doc_model_dir} \
     --train_steps=${train_steps} \
     --save_checkpoints_steps=${save_steps} \
     --learning_rate=${learning_rate} \
     --batch_size=${batch_size} \
     --char_embedding_size=${char_embedding_size} \
     --rnn_hidden_size=${rnn_hidden_size} \
     --query_max_char_length=${query_max_char_length} \
     --doc_max_char_length=${doc_max_char_length} \
     --last_hidden_size=${last_hidden_size} \
     --char_vocab_size=${char_vocab_size} \
     --NEG=${NEG} \
     --gpuid=${gpuid} \
     --train_eval=True \
     --export_query_model=True \
     --export_doc_model=True \
     --run_on_cluster=False
