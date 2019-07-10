#!/usr/env/bin python
#coding=utf-8

import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from CommonLibs import ModelUtils

# Frobenius（F范数）
def frobenius(tensor):
    input_size = tf.to_float(tf.shape(tensor)[0])
    ret = tf.reduce_sum(tf.reduce_sum(tf.square(tensor), 2), 1)
    print ret.get_shape()
    return tf.div(tf.reduce_sum(tf.add(tf.sqrt(ret), 1e-10), 0), input_size)

class StructuredSentEmbedding:
    def __init__(self,
                 vocab_size,
                 max_seq_length,
                 d_emb,
                 d_hiddens,
                 d_attention,
                 attention_channels,
                 d_mlp,
                 d_fc,
                 class_num=2,
                 frobenius_norm=1.0):
        self.vocab_size = vocab_size
        self.d_emb = d_emb
        self.d_hiddens = d_hiddens
        self.d_attention = d_attention
        self.attention_channels = attention_channels
        self.seq_length = max_seq_length
        self.class_num = 2
        self.frobenius_weight = frobenius_norm
        self.d_mlp = d_mlp
        self.d_fc = d_fc

        self.input_a = tf.placeholder(tf.int32, [None, self.seq_length], name="input_ids_a")
        self.input_b = tf.placeholder(tf.int32, [None, self.seq_length], name="input_ids_b")
        self.input_y = tf.placeholder(tf.float32, [None, self.class_num], name="input_y")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")

        self.embeddings = ModelUtils.get_token_embedding(self.vocab_size,
                                                         self.d_emb,
                                                         scope="embeddings")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

    def build_lstm_layer(self):
        self.input_embeddings_a = tf.nn.embedding_lookup(self.embeddings, self.input_a)
        self.input_embeddings_b = tf.nn.embedding_lookup(self.embeddings, self.input_b)

        outputs_a = self.input_embeddings_a # [batch_size, seq_length, d_embedding]
        outputs_b = self.input_embeddings_b # [batch_size, seq_length, d_embedding]

        with tf.variable_scope("bi_lstm_layers", reuse=tf.AUTO_REUSE):
            for layer_id, hidden_size in enumerate(self.d_hiddens):
                with tf.variable_scope("layer_%d" % layer_id, reuse=tf.AUTO_REUSE):
                    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=(1-self.dropout_prob))
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=(1-self.dropout_prob))
                    outputs_a, states_a = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, outputs_a, dtype=tf.float32)
                    outputs_a = tf.concat(outputs_a, axis=2)    # [batch_size, seq_length, d_lstm_hidden*2]
                    outputs_b, states_b = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, outputs_b, dtype=tf.float32)
                    outputs_b = tf.concat(outputs_b, axis=2)    # [batch_size, seq_length, d_lstm_hidden*2]
        print "lstm outputs size: {}".format(outputs_a.get_shape())
        return outputs_a, outputs_b

    def build_attention_layer(self, input_a, input_b):
        with tf.variable_scope("attention_layer", reuse=tf.AUTO_REUSE):
            input_a = tf.transpose(input_a, [0, 2, 1])
            input_b = tf.transpose(input_b, [0, 2, 1])
            w_s1 = tf.Variable(tf.truncated_normal(shape=[self.d_attention, 2*self.d_hiddens[-1]],
                                                   stddev=0.1, dtype=tf.float32), name="w_s1")
            w_ = tf.tile(w_s1, [tf.shape(input_a)[0], 1])
            w_s1 = tf.reshape(w_, [tf.shape(input_a)[0], w_s1.get_shape()[0], w_s1.get_shape()[1]])
            print "w_s1 size: {}".format(w_s1.get_shape())
            w_s3 = tf.reshape(w_, [self.d_attention, self.d_hiddens[-1], 2])
            output_a = tf.nn.tanh(tf.matmul(w_s1, input_a))     # [batch_size, ]
            output_b = tf.nn.tanh(tf.matmul(w_s1, input_b))
            print "attention inside output size: {}".format(output_a.get_shape())

            w_s2 = tf.Variable(tf.truncated_normal(shape=[self.attention_channels, self.d_attention],
                                                   stddev=0.1, dtype=tf.float32), name="w_s2")
            w_ = tf.tile(w_s2, [tf.shape(output_a)[0], 1])
            w_s2 = tf.reshape(w_, [tf.shape(output_a)[0], w_s2.get_shape()[0], w_s2.get_shape()[1]])
            output_a = tf.nn.softmax(tf.matmul(w_s2, output_a), name="attention_a")
            output_b = tf.nn.softmax(tf.matmul(w_s2, output_b), name="attention_b")
        print "attention output metrix size: {}".format(output_a.get_shape())
        return output_a, output_b

    def build_output_layer(self):
        pooled_dim = self.attention_channels * self.d_hiddens[-1] * 2
        pooled_a = tf.reshape(self.outputs_a, [tf.shape(self.outputs_a)[0], pooled_dim], name="pooled_a")
        pooled_b = tf.reshape(self.outputs_b, [tf.shape(self.outputs_b)[0], pooled_dim], name="pooled_b")
        # pooled_a = tf.concat(self.encoder_output_a, axis=-1, name="pooled_a")
        # pooled_b = tf.concat(self.encoder_output_b, axis=-1, name="pooled_b")
        print "pooled_a size: {}".format(pooled_a.get_shape())
        output = tf.concat([pooled_a, pooled_b], axis=-1, name="pooled_ab")
        print "pooled_ab size: {}".format(output.get_shape())
        d_ins = [int(output.get_shape()[-1])] + self.d_mlp
        with tf.variable_scope("mlp_layer", reuse=tf.AUTO_REUSE):
            for layer_id, d_out in enumerate(self.d_mlp):
                d_in = d_ins[layer_id]
                with tf.variable_scope("layer_%d" % layer_id, reuse=tf.AUTO_REUSE):
                    w = tf.Variable(tf.truncated_normal(shape=[d_in, d_out],
                                                        stddev=0.02, dtype=tf.float32), name="w")
                    b = tf.Variable(tf.constant(value=0.1, shape=[d_out], dtype=tf.float32), name="b")
                    output = tf.nn.relu(tf.nn.xw_plus_b(output, w, b))
        print "mlp output size: {}".format(output.get_shape())
        with tf.variable_scope("output_layer", reuse=tf.AUTO_REUSE):
            d_in = self.d_mlp[-1]
            w = tf.Variable(tf.truncated_normal(shape=[d_in, self.class_num],
                                                stddev=0.02, dtype=tf.float32), name="w")
            b = tf.Variable(tf.constant(value=0.1, shape=[self.class_num], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(output, w, b)
            self.probs = tf.nn.softmax(self.logits, name="probs")
            self.predictions = tf.argmax(self.logits, 1, name="prediction")
            self.labels = tf.argmax(self.input_y, 1, name="labels")
        print "logits size: {}".format(self.logits.get_shape())

    def build_loss(self):
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.logits)
            self.softmax_score_losses = tf.reduce_mean(losses, name="pred_losses")
            self.l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                      name="l2_losses") * 0.0000001
            self.frobenius_loss = frobenius(self.m_a) + frobenius(self.m_b)
            self.loss = tf.add(self.softmax_score_losses, self.l2_losses, name="loss")
            self.frobenius_loss = frobenius(self.m_a) + frobenius(self.m_b)
            self.loss = tf.add(self.loss, self.frobenius_loss, name="loss")

    def build(self):
        # lstm层
        h_a, h_b = self.build_lstm_layer()
        # attention层
        self.m_a, self.m_b = self.build_attention_layer(h_a, h_b)
        # output
        self.outputs_a = tf.matmul(self.m_a, h_a, name="outputs_a")
        self.outputs_b = tf.matmul(self.m_b, h_b, name="outputs_b")
        self.build_output_layer()
        # loss
        self.build_loss()


if __name__ == "__main__":
    vocab_size = 100
    emb_size = 16
    max_seq_len = 20
    hidden_sizes = [32]
    fc_size = 256
    d_mlp = [33, 22, 11]

    sess = tf.Session()
    with sess.as_default():
        model = StructuredSentEmbedding(vocab_size=vocab_size,
                                        max_seq_length=max_seq_len,
                                        d_emb=emb_size,
                                        d_hiddens=hidden_sizes,
                                        d_attention=13,
                                        attention_channels=5,
                                        d_mlp=d_mlp,
                                        d_fc=13)
        model.build()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        inputs_a = np.random.randint(0, vocab_size, size=[5, max_seq_len])
        inputs_b = np.random.randint(0, vocab_size, size=[5, max_seq_len])
        inputs_y = np.array([[0, 1], [1, 0], [0, 1], [0, 1], [1, 0]])
        print inputs_a
        print inputs_b

        # inputs_a = np.ones([1, emb_size])
        # inputs_b = np.ones([1, emb_size])
        run_args = [model.probs, model.labels, model.outputs_a, model.outputs_b, model.m_a, model.m_b, model.loss]
        scores, raw_scores, fc_outputs_a, fc_outputs_b, attention_a, attention_b, loss = sess.run(run_args,
                                                                  feed_dict={model.input_a: inputs_a,
                                                                  model.input_b: inputs_b,
                                                                  model.dropout_prob: 0.0,
                                                                  model.input_y: inputs_y,
                                                                  })
        print scores
        print raw_scores
        # print attention_a
        # print attention_b
        print loss

        # tensor_a = tf.constant(1, shape=[3, 10, 20], dtype=tf.float32)
        # print frobenius(tensor_a)

