#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test_ssa.py
# @Author: zhangxiang
# @Date  : 2019/7/9
# @Desc  :

import os
import sys
import tensorflow as tf
import numpy as np
import logging
import json

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../.."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../../.."))

import data_helper as dh
import hyperparams
import Modeling

from CommonLibs import OtherUtils

prj_name = sys.argv[1]
hp = hyperparams.Hyperparams(prj_name)

model_id = sys.argv[2]

my_dh = dh.TrainDataHelper(hp.model_params["max_seq_len"])
my_dh.initialize()

def trans_line_to_inputs(line):
    datas = []
    splits = line.strip('\r\n').split('\t')
    q_b = splits[-2]
    label = int(float(splits[-1]))
    q_id_b = my_dh.trans_query_to_input_id(q_b)

    q_a = ",".join(splits[:-2])
    datas.append([my_dh.trans_query_to_input_id(q_a), q_id_b])
    for q in splits[:-2]:
        q_id = my_dh.trans_query_to_input_id(q)
        datas.append([q_id, q_id_b])
    return datas, splits[:-2], q_b, label


graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        saver = tf.train.import_meta_graph("%s/model-%s.meta" % (hp.train_params["model_path"], model_id))
        saver.restore(sess, "%s/model-%s" % (hp.train_params["model_path"], model_id))

        input_a_ids = graph.get_operation_by_name("input_ids_a").outputs[0]
        input_b_ids = graph.get_operation_by_name("input_ids_b").outputs[0]
        dropout_prob = graph.get_operation_by_name("dropout_prob").outputs[0]
        output_probs = graph.get_operation_by_name("output_layer/probs").outputs[0]

        def predict_step(batch_ids_a, batch_ids_b):
            feed_dict = {
                input_a_ids: batch_ids_a,
                input_b_ids: batch_ids_b,
                dropout_prob: 0.0,
            }
            probs = sess.run([output_probs], feed_dict=feed_dict)
            return probs[0]

        wp = open("result.txt", 'w')
        for line in sys.stdin:
            batch, q_a_s, q_b, label = trans_line_to_inputs(line)
            batch_ids_a = [my_dh.get_input_ids(data[0]) for data in batch]
            batch_ids_b = [my_dh.get_input_ids(data[1]) for data in batch]
            probs = predict_step(batch_ids_a, batch_ids_b)
            prob_raw = probs[0][1]
            if prob_raw < 0.5:
                continue
            scores = []
            for i in range(len(q_a_s)):
                prob = probs[i+1][1]
                scores.append(prob*3)
                # scores.append(-abs(prob-prob_raw))
            scores = sess.run(tf.nn.softmax(scores))
            print "*" * 50
            print [prob[1] for prob in probs]
            print "\t".join(["%s|%.3f" % (q, s) for q,s in zip(q_a_s, scores)]) + "\n"
            print >> wp, ",".join(q_a_s)
            print >> wp, "\t".join(["%s|%.3f" % (q, s) for q,s in zip(q_a_s, scores)])
        wp.close()