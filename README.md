# dssm
A BiGRU-Attention DSSM implementation with tensorflow estimator.

之前使用Keras和paddlepaddle实现过DSSM文本表示模型，(https://github.com/cdj0311/keras_bert_classification/blob/master/bert_dssm.py， https://github.com/cdj0311/paddledssm)
由于Keras做分布式计算比较麻烦，而paddlepaddle早已弃用，现在用tensorflow的高级API tf.estimator重写一遍，其中表示层使用双向GRU+Attention，最终输出为64维的向量。

python == 3.6

tensorflow == 1.13.1


训练步骤如下：
1. 将文本数据转换为tfrecord格式：

   python convert_data.py
   
   data目录的data.txt中包含了10000条训练数据，格式为：title\tcontent，train.tfrecord是转换完成的tfrecord数据。

2. 模型训练：

   sh train_local.sh
   
   模型训练完后会分别导出query和doc的pb格式模型，可根据需要进行选择。
   
3. 模型预测：
    
   python predict.py
   
   给定一个句子得到向量，并获取最相似的N个句子，例如：
   
   输入： 赵丽颖冯绍峰在拍女儿国的时候真的超级甜了
   
   输出：
      
          0.801103	女神赵丽颖李沁都爱穿黄毛衣，但差距真的蛮大的
          0.744942	街拍：喜欢第二位俏皮可爱的小姐姐，和她在一起不会觉得无聊！
          0.722599	杜江霍思燕夫妇甜蜜现身 牵手依偎恩爱甜到发腻
          0.719018	还在情侣穿搭烦恼，看街拍情侣都是怎么搭配的
          0.707306	赵丽颖，应是绿肥红瘦，剧照
          0.701783	她的闺蜜则穿了一件白色的蕾丝连衣裙，尽显女人味
          0.70024	国民妖精十元女神可爱撩人瞬间合集！出色的不只是时尚穿搭
          0.691073	图集：#杨幂#赵丽颖暗斗时尚穿同款婚纱谁更美
          0.687201	赵丽颖 路人抓拍下的颖宝，这颜值可以说是完美的纯天然美女了～
          
 4. 分布式训练
 
    设置run_on_cluster=True， 提交到job中即可训练，由于每个公司的分布式训练提交命令不一样，这里就不贴出来了。
    
 该项目是基于字符做Embedding，实际使用中我们一般会将字和词同时作为输入进行训练。
 
   
