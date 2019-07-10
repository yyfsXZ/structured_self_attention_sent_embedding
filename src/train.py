#!/usr/env/bin python
#coding=utf-8

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

from CommonLibs import ModelUtils
from CommonLibs import Metrix
from CommonLibs import OtherUtils

prj_name = sys.argv[1]
hp = hyperparams.Hyperparams(prj_name)

def train():
    # 初始化日志和模型路径
    OtherUtils.initPaths(hp.train_params["model_path"], hp.train_params["log_path"])

    # 初始化输入文件
    train_data_helper = dh.TrainDataHelper(hp.model_params["max_seq_len"])
    train_data_helper.initialize()
    vocab_size = train_data_helper.get_vocab_size() # 词汇量大小

    train_datas = train_data_helper.read_input_file(hp.train_params["train_file"], type="train")
    train_data_size = len(train_datas)
    valid_datas = train_data_helper.read_input_file(hp.train_params["valid_file"], type="valid")
    valid_data_size = len(valid_datas)

    logging.info("Train start")

    # Build a graph and rnn object
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            log_device_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            model = Modeling.StructuredSentEmbedding(vocab_size=vocab_size,
                                                     max_seq_length=hp.model_params["max_seq_len"],
                                                     d_emb=hp.model_params["embedding_size"],
                                                     d_hiddens=hp.model_params["d_hidden_lstm"],
                                                     d_attention=hp.model_params["d_attention"],
                                                     attention_channels=hp.model_params["attention_channels"],
                                                     frobenius_norm=hp.train_params["frobenius_norm"],
                                                     d_mlp=hp.model_params["mlp_size"],
                                                     d_fc=hp.model_params["fc_size"])
            model.build()

            # 获取train_operator
            train_op = ModelUtils.train_step(model.loss,
                                             hp.train_params["learning_rate"],
                                             model.global_step,
                                             decay=False)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=hp.train_params["num_checkpoints"])

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            def train_step(batch_ids_a, batch_ids_b, batch_labels):
                batch_labels = np.array(batch_labels)
                feed_dict = {
                    model.input_a: batch_ids_a,
                    model.input_b: batch_ids_b,
                    model.input_y: batch_labels,
                    model.dropout_prob: hp.train_params["dropout_rate"],
                }
                _, loss, scores, preds, labels = sess.run(
                    [train_op, model.loss, model.probs, model.predictions, model.labels], feed_dict=feed_dict
                )
                print "actucal-{} predict-{}".format(labels[:32], preds[:32])
                tp, fp, tn, fn = Metrix.get_accu(scores[:, 1], batch_labels[:, 1], hp.train_params["accu_threshold"])
                return loss, tp, fp, tn, fn

            def validation_step(batch_ids_a, batch_ids_b, batch_labels):
                batch_labels = np.array(batch_labels)
                feed_dict = {
                    model.input_a: batch_ids_a,
                    model.input_b: batch_ids_b,
                    model.input_y: batch_labels,
                    model.dropout_prob: 0.0,
                }
                loss, scores = sess.run(
                    [model.softmax_score_losses, model.probs],
                    feed_dict=feed_dict
                )
                tp, fp, tn, fn = Metrix.get_accu(scores[:, 1], batch_labels[:, 1], hp.train_params["accu_threshold"])
                return loss, tp, fp, tn, fn

            with tf.device("/gpu:0"):
                batch_per_epoch = train_data_size / hp.train_params["batch_size"]
                if train_data_size % hp.train_params["batch_size"] != 0:
                    batch_per_epoch += 1
                valid_batch_sum = valid_data_size / hp.train_params["batch_size"]
                if valid_data_size % hp.train_params["batch_size"] != 0:
                    valid_batch_sum += 1

                best_val_loss = 1000
                best_val_accu = 0.0
                best_val_recall = 0.0
                best_val_prec = 0.0
                best_val_f1 = -1
                best_epoch = -1
                for epoch in range(hp.train_params["epochs"]):
                    total_loss = 0.0
                    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
                    batches = train_data_helper.batch_iter(train_datas, hp.train_params["batch_size"], shuffle=True)
                    for idx, batch in enumerate(batches):
                        batch_ids_a = [train_data_helper.get_input_ids(data[0]) for data in batch]
                        batch_ids_b = [train_data_helper.get_input_ids(data[1]) for data in batch]
                        batch_labels = [data[2:] for data in batch]
                        _loss, _tp, _fp, _tn, _fn = train_step(batch_ids_a, batch_ids_b, batch_labels)
                        total_loss += _loss
                        tp += _tp
                        tn += _tn
                        fp += _fp
                        fn += _fn
                        if idx !=0 and idx % (batch_per_epoch/10) == 0:
                            tmp_loss = total_loss / idx
                            tmp_accu = (tp+tn) / (tp+tn+fp+fn)
                            per = idx/(batch_per_epoch/10)
                            mess = "Epoch: %d, percent: %d0%%, loss: %f, accu: %f" % (epoch, per, tmp_loss, tmp_accu)
                            logging.info(mess)
                            logging.info("Epoch: %d, percent: %d0%%, tp=%d, tn=%d, fp=%d, fn=%d" % (epoch, per, int(tp), int(tn), int(fp), int(fn)))


                    total_loss = total_loss / batch_per_epoch
                    accu = (tp+tn) / (tp+tn+fp+fn)
                    mess = "Epoch %d: train result - loss %f, accu %f"%(epoch, total_loss, accu)
                    logging.info(mess)


                    total_loss = 0.0
                    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
                    batches = train_data_helper.batch_iter(valid_datas, hp.train_params["batch_size"], shuffle=False)
                    # for batch_ids_a, batch_ids_b, batch_labels in zip(batches_ids_a, batches_ids_b, batches_labels):
                    for batch in batches:
                        batch_ids_a = [train_data_helper.get_input_ids(data[0]) for data in batch]
                        batch_ids_b = [train_data_helper.get_input_ids(data[1]) for data in batch]
                        batch_labels = [data[2:] for data in batch]
                        loss_, _tp, _fp, _tn, _fn = validation_step(batch_ids_a, batch_ids_b, batch_labels)
                        total_loss += loss_
                        tp += _tp
                        tn += _tn
                        fp += _fp
                        fn += _fn
                    total_loss = total_loss / valid_batch_sum
                    accu, recall, f1, prec = Metrix.eva(tp, tn, fp, fn)
                    mess = "Evaluation: loss %f, acc %f, recall %f, precision %f, f1 %f" % \
                           (total_loss, accu, recall, prec, f1)
                    logging.info(mess)
                    logging.info("Evaluation: tp=%d, tn=%d, fp=%d, fn=%d" % (int(tp), int(tn), int(fp), int(fn)))

                    # checkpoint_prefix = "%s/model" % FLAGS.model_path
                    # path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                    # print("Saved model checkpoint to {0}".format(path))
                    if best_val_loss > total_loss:
                        best_val_loss = total_loss
                        best_val_accu = accu
                        best_val_recall = recall
                        best_val_prec = prec
                        best_val_f1 = f1
                        best_epoch = epoch
                        checkpoint_prefix = "%s/model" % hp.train_params["model_path"]
                        path = saver.save(sess, checkpoint_prefix, global_step=epoch)
                        print("Saved model checkpoint to {0}".format(path))
                        model_conf = {"epoch": 0,
                                      "maxSeqLength": hp.model_params["max_seq_len"],
                                      "model_size": json.dumps(hp.model_params),
                                      "vocabDic": "vocab.txt",
                                      "encoderModelPath": "encoder",
                                      "similarityModelPath": "similairity",
                                      "hiddenSize": hp.model_params["d_hidden_lstm"][-1],
                                      "attentionChannelSize": hp.model_params["attention_channels"]}
                        train_data_helper.save_vocab_file("%s/%s" % (hp.train_params["model_path"], model_conf["vocabDic"]))
                        model_conf_file = "%s/model.conf" % hp.train_params["model_path"]
                        with open(model_conf_file, 'w') as wp:
                            print >> wp, json.dumps(model_conf, ensure_ascii=False)
                            wp.close()
                logging.info("Best epoch=%d, loss=%f, accu=%.4f, recall=%.4f, prec=%.4f, f1=%.4f",
                             best_epoch, best_val_loss, best_val_accu, best_val_recall, best_val_prec, best_val_f1)
        logging.info("Training done")

if __name__ == "__main__":
    train()