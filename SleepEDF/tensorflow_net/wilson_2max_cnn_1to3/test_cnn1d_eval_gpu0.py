#! /usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,-1"
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

from datetime import datetime
import h5py
from scipy.io import loadmat,savemat

from cnn_1d_sleep import CNN1DSleep
from cnn_1d_sleep_config import Config

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

from datagenerator_from_list_v2 import DataGenerator


# Parameters
# ==================================================

#tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# My Parameters
tf.app.flags.DEFINE_string("eeg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eeg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("eog_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_train_data", "../train_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_eval_data", "../data/eval_data_1.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("emg_test_data", "../test_data.mat", "Point to directory of input data")
tf.app.flags.DEFINE_string("out_dir", "./output/", "Point to output directory")
tf.app.flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "Point to checkpoint directory")

tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_integer("num_filter", 500, "Number of filters per filter size (default: 400)")

tf.app.flags.DEFINE_string("eeg_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")
tf.app.flags.DEFINE_string("eog_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")
tf.app.flags.DEFINE_string("emg_pretrainedfb_path", "./output/filterbank.mat", "Point to the pretrainedfb mat file")


FLAGS = tf.app.flags.FLAGS
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()): # python3
    print("{}={}".format(attr.upper(), value))
print("")

out_path = os.path.abspath(os.path.join(os.path.curdir,FLAGS.out_dir))
checkpoint_path = os.path.abspath(os.path.join(out_path,FLAGS.checkpoint_dir))

config = Config()
config.dropout_keep_prob = FLAGS.dropout_keep_prob
config.num_filters = FLAGS.num_filter

eeg_active = ((FLAGS.eeg_train_data != "") & (FLAGS.eeg_test_data != ""))
eog_active = ((FLAGS.eog_train_data != "") & (FLAGS.eog_test_data != ""))
emg_active = ((FLAGS.emg_train_data != "") & (FLAGS.emg_test_data != ""))

if (eeg_active):
    print("eeg active")
    # Initalize the data generator seperately for the training, test sets
    # We load the training data here just for normalization purpose
    eeg_train_gen = DataGenerator(os.path.abspath(FLAGS.eeg_train_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    eeg_test_gen = DataGenerator(os.path.abspath(FLAGS.eeg_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    eeg_eval_gen = DataGenerator(os.path.abspath(FLAGS.eeg_eval_data), data_shape=[config.time_length, config.freq_length], shuffle=False, test_mode=True)

    # load pretrained filterbank and do filtering first
    eeg_filter = loadmat(FLAGS.eeg_pretrainedfb_path)
    Wfb = eeg_filter['Wfb']
    eeg_train_gen.filter_with_filterbank(Wfb)
    eeg_test_gen.filter_with_filterbank(Wfb)
    eeg_eval_gen.filter_with_filterbank(Wfb)
    del Wfb

if (eog_active):
    print("eog active")
    # Initalize the data generator seperately for the training, test sets
    eog_train_gen = DataGenerator(os.path.abspath(FLAGS.eog_train_data), data_shape=[config.time_length, config.freq_length], shuffle = False)
    eog_test_gen = DataGenerator(os.path.abspath(FLAGS.eog_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    eog_eval_gen = DataGenerator(os.path.abspath(FLAGS.eog_eval_data), data_shape=[config.time_length, config.freq_length], shuffle=False, test_mode=True)
    # do filtering first
    eog_filter = loadmat(FLAGS.eog_pretrainedfb_path)
    Wfb = eog_filter['Wfb']
    eog_train_gen.filter_with_filterbank(Wfb)
    eog_test_gen.filter_with_filterbank(Wfb)
    eog_eval_gen.filter_with_filterbank(Wfb)
    del Wfb

if (emg_active):
    print("emg active")
    # Initalize the data generator seperately for the training, test sets
    emg_train_gen = DataGenerator(os.path.abspath(FLAGS.emg_train_data), data_shape=[config.time_length, config.freq_length], shuffle = False)
    emg_test_gen = DataGenerator(os.path.abspath(FLAGS.emg_test_data), data_shape=[config.time_length, config.freq_length], shuffle = False, test_mode=True)
    emg_eval_gen = DataGenerator(os.path.abspath(FLAGS.emg_eval_data), data_shape=[config.time_length, config.freq_length], shuffle=False, test_mode=True)
    # do filtering first
    emg_filter = loadmat(FLAGS.emg_pretrainedfb_path)
    Wfb = emg_filter['Wfb']
    emg_train_gen.filter_with_filterbank(Wfb)
    emg_test_gen.filter_with_filterbank(Wfb)
    emg_eval_gen.filter_with_filterbank(Wfb)
    del Wfb

# eeg always active
train_generator = eeg_train_gen
test_generator = eeg_test_gen
eval_generator = eeg_eval_gen

# concatenate different channels
if (eog_active):
    train_generator.X = np.concatenate((train_generator.X, eog_train_gen.X), axis=-1)
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.concatenate((test_generator.X, eog_test_gen.X), axis=-1)
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.concatenate((eval_generator.X, eog_eval_gen.X), axis=-1)
    eval_generator.data_shape = eval_generator.X.shape[1:]

if (emg_active):
    train_generator.X = np.concatenate((train_generator.X, emg_train_gen.X), axis=-1)
    train_generator.data_shape = train_generator.X.shape[1:]
    test_generator.X = np.concatenate((test_generator.X, emg_test_gen.X), axis=-1)
    test_generator.data_shape = test_generator.X.shape[1:]
    eval_generator.X = np.concatenate((eval_generator.X, emg_eval_gen.X), axis=-1)
    eval_generator.data_shape = eval_generator.X.shape[1:]

del eeg_train_gen
del eeg_test_gen
del eeg_eval_gen
if (eog_active):
    del eog_train_gen
    del eog_test_gen
    del eog_eval_gen
if (emg_active):
    del emg_train_gen
    del emg_test_gen
    del emg_eval_gen


# data normalization here
X = train_generator.X
X = np.reshape(X,(train_generator.data_size*train_generator.data_shape[0], train_generator.data_shape[1]))
meanX = X.mean(axis=0)
stdX = X.std(axis=0)
X = (X - meanX) / stdX
train_generator.X = np.reshape(X, (train_generator.data_size, train_generator.data_shape[0], train_generator.data_shape[1]))

X = test_generator.X
X = np.reshape(X,(test_generator.data_size*test_generator.data_shape[0], test_generator.data_shape[1]))
X = (X - meanX) / stdX
test_generator.X = np.reshape(X, (test_generator.data_size, test_generator.data_shape[0], test_generator.data_shape[1]))

X = eval_generator.X
X = np.reshape(X,(eval_generator.data_size*eval_generator.data_shape[0], eval_generator.data_shape[1]))
X = (X - meanX) / stdX
eval_generator.X = np.reshape(X, (eval_generator.data_size, eval_generator.data_shape[0], eval_generator.data_shape[1]))

# update the frequency dimension after filtering
config.freq_length = train_generator.data_shape[1]

test_batches_per_epoch = np.floor(len(test_generator.data_index) / config.batch_size).astype(np.int16)
train_batches_per_epoch = np.floor(len(train_generator.data_index) / config.batch_size).astype(np.int16)
eval_batches_per_epoch = np.floor(len(eval_generator.data_index) / config.batch_size).astype(np.int16)

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    session_conf.gpu_options.allow_growth = True
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = CNN1DSleep(
            time_length=config.time_length,
            freq_length=config.freq_length,
            num_classes=config.num_classes,
            filter_sizes=list(map(int, config.filter_sizes.split(","))),
            num_filters=config.num_filters,
            l2_reg_lambda=config.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        def dev_step(x_batch, y_batch1, y_batch2, y_batch3):
            feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y1: y_batch1,
                  cnn.input_y2: y_batch2,
                  cnn.input_y3: y_batch3,
                  cnn.dropout_keep_prob: 1.0
                }
            loss, acc1, pred_Y1, score1_, \
                acc2, pred_Y2, score2_,\
                acc3, pred_Y3, score3_ = sess.run(
                   [cnn.loss, cnn.accuracy1, cnn.predictions1, cnn.scores1,
                    cnn.accuracy2, cnn.predictions2, cnn.scores2,
                    cnn.accuracy3, cnn.predictions3, cnn.scores3],
                   feed_dict)
            return acc1, acc2, acc3, pred_Y1, pred_Y2, pred_Y3, score1_, score2_, score3_

        saver = tf.train.Saver(tf.all_variables()) #wilson: tf.global_variables()

        # Load saved model to continue training or initialize all variables
        best_dir = os.path.join(checkpoint_path, "best_model_acc")
        saver.restore(sess, best_dir)
        print("Model loaded")

        # wilson: test set metrics
        test_yhat1 = np.zeros_like(test_generator.data_index)
        score1 = np.zeros([len(test_generator.data_index),config.num_classes])
        test_yhat2 = np.zeros_like(test_generator.data_index)
        score2 = np.zeros([len(test_generator.data_index),config.num_classes])
        test_yhat3 = np.zeros_like(test_generator.data_index)
        score3 = np.zeros([len(test_generator.data_index),config.num_classes])
        test_step = 1
        while test_step < test_batches_per_epoch:
            x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = test_generator.next_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, _, _, test_yhat1_, test_yhat2_, test_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            test_yhat1[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat1_
            score1[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score1_
            test_yhat2[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat2_
            score2[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score2_
            test_yhat3[(test_step-1)*config.batch_size : test_step*config.batch_size] = test_yhat3_
            score3[(test_step-1)*config.batch_size : test_step*config.batch_size,:] = score3_
            test_step += 1
        if test_generator.pointer < len(test_generator.data_index):
            actual_len, x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = test_generator.rest_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, _, _, test_yhat1_, test_yhat2_, test_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            test_yhat1[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat1_
            score1[(test_step-1)*config.batch_size : len(test_generator.data_index),:] = score1_
            test_yhat2[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat2_
            score2[(test_step-1)*config.batch_size : len(test_generator.data_index),:] = score2_
            test_yhat3[(test_step-1)*config.batch_size : len(test_generator.data_index)] = test_yhat3_
            score3[(test_step-1)*config.batch_size : len(test_generator.data_index),:] = score3_
        test_yhat1 = test_yhat1 + 1
        test_yhat2 = test_yhat2 + 1
        test_yhat3 = test_yhat3 + 1
        test_fscore1 = f1_score(test_generator.label[:-1], test_yhat1[1:], average='macro')
        test_acc1 = accuracy_score(test_generator.label[:-1], test_yhat1[1:])
        test_kappa1 = cohen_kappa_score(test_generator.label[:-1], test_yhat1[1:])
        test_fscore2 = f1_score(test_generator.label, test_yhat2, average='macro')
        test_acc2 = accuracy_score(test_generator.label, test_yhat2)
        test_kappa2 = cohen_kappa_score(test_generator.label, test_yhat2)
        test_fscore3 = f1_score(test_generator.label[1:], test_yhat3[:-1], average='macro')
        test_acc3 = accuracy_score(test_generator.label[1:], test_yhat3[:-1])
        test_kappa3 = cohen_kappa_score(test_generator.label[1:], test_yhat3[:-1])
        savemat(os.path.join(out_path, "test_ret_model_acc.mat"), dict(acc1=test_acc1,
                                                                       fscore1=test_fscore1,
                                                                       kappa1=test_kappa1,
                                                                       acc2=test_acc2,
                                                                       fscore2=test_fscore2,
                                                                       kappa2=test_kappa2,
                                                                       acc3=test_acc3,
                                                                       fscore3=test_fscore3,
                                                                       kappa3=test_kappa3,
                                                                       yhat1=test_yhat1,
                                                                       yhat2=test_yhat2,
                                                                       yhat3=test_yhat3,
                                                                       score1 = score1,
                                                                       score2 = score2,
                                                                       score3 = score3))
        test_generator.reset_pointer()

        # wilson: train set metrics
        train_yhat1 = np.zeros_like(train_generator.data_index)
        score1 = np.zeros([len(train_generator.data_index),config.num_classes])
        train_yhat2 = np.zeros_like(train_generator.data_index)
        score2 = np.zeros([len(train_generator.data_index),config.num_classes])
        train_yhat3 = np.zeros_like(train_generator.data_index)
        score3 = np.zeros([len(train_generator.data_index),config.num_classes])
        train_step = 1
        while train_step < train_batches_per_epoch:
            x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = train_generator.next_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, _, _, train_yhat1_, train_yhat2_, train_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            train_yhat1[(train_step-1)*config.batch_size : train_step*config.batch_size] = train_yhat1_
            score1[(train_step-1)*config.batch_size : train_step*config.batch_size,:] = score1_
            train_yhat2[(train_step-1)*config.batch_size : train_step*config.batch_size] = train_yhat2_
            score2[(train_step-1)*config.batch_size : train_step*config.batch_size,:] = score2_
            train_yhat3[(train_step-1)*config.batch_size : train_step*config.batch_size] = train_yhat3_
            score3[(train_step-1)*config.batch_size : train_step*config.batch_size,:] = score3_
            train_step += 1
        if train_generator.pointer < len(train_generator.data_index):
            actual_len, x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = train_generator.rest_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, _, _, train_yhat1_, train_yhat2_, train_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            train_yhat1[(train_step-1)*config.batch_size : len(train_generator.data_index)] = train_yhat1_
            score1[(train_step-1)*config.batch_size : len(train_generator.data_index),:] = score1_
            train_yhat2[(train_step-1)*config.batch_size : len(train_generator.data_index)] = train_yhat2_
            score2[(train_step-1)*config.batch_size : len(train_generator.data_index),:] = score2_
            train_yhat3[(train_step-1)*config.batch_size : len(train_generator.data_index)] = train_yhat3_
            score3[(train_step-1)*config.batch_size : len(train_generator.data_index),:] = score3_
        train_yhat1 = train_yhat1 + 1
        train_yhat2 = train_yhat2 + 1
        train_yhat3 = train_yhat3 + 1
        train_fscore1 = f1_score(train_generator.label[:-1], train_yhat1[1:], average='macro')
        train_acc1 = accuracy_score(train_generator.label[:-1], train_yhat1[1:])
        train_kappa1 = cohen_kappa_score(train_generator.label[:-1], train_yhat1[1:])
        train_fscore2 = f1_score(train_generator.label, train_yhat2, average='macro')
        train_acc2 = accuracy_score(train_generator.label, train_yhat2)
        train_kappa2 = cohen_kappa_score(train_generator.label, train_yhat2)
        train_fscore3 = f1_score(train_generator.label[1:], train_yhat3[:-1], average='macro')
        train_acc3 = accuracy_score(train_generator.label[1:], train_yhat3[:-1])
        train_kappa3 = cohen_kappa_score(train_generator.label[1:], train_yhat3[:-1])
        savemat(os.path.join(out_path, "train_ret_model_acc.mat"), dict(acc1=train_acc1,
                                                                       fscore1=train_fscore1,
                                                                       kappa1=train_kappa1,
                                                                       acc2=train_acc2,
                                                                       fscore2=train_fscore2,
                                                                       kappa2=train_kappa2,
                                                                       acc3=train_acc3,
                                                                       fscore3=train_fscore3,
                                                                       kappa3=train_kappa3,
                                                                       yhat1=train_yhat1,
                                                                       yhat2=train_yhat2,
                                                                       yhat3=train_yhat3,
                                                                       score1 = score1,
                                                                       score2 = score2,
                                                                       score3 = score3))
        train_generator.reset_pointer()

        # eval(dev) set metrics
        eval_yhat1 = np.zeros_like(eval_generator.data_index)
        score1 = np.zeros([len(eval_generator.data_index),config.num_classes])
        eval_yhat2 = np.zeros_like(eval_generator.data_index)
        score2 = np.zeros([len(eval_generator.data_index),config.num_classes])
        eval_yhat3 = np.zeros_like(eval_generator.data_index)
        score3 = np.zeros([len(eval_generator.data_index),config.num_classes])
        eval_step = 1
        while eval_step < eval_batches_per_epoch:
            x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = eval_generator.next_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, _, _, eval_yhat1_, eval_yhat2_, eval_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            eval_yhat1[(eval_step-1)*config.batch_size : eval_step*config.batch_size] = eval_yhat1_
            score1[(eval_step-1)*config.batch_size : eval_step*config.batch_size,:] = score1_
            eval_yhat2[(eval_step-1)*config.batch_size : eval_step*config.batch_size] = eval_yhat2_
            score2[(eval_step-1)*config.batch_size : eval_step*config.batch_size,:] = score2_
            eval_yhat3[(eval_step-1)*config.batch_size : eval_step*config.batch_size] = eval_yhat3_
            score3[(eval_step-1)*config.batch_size : eval_step*config.batch_size,:] = score3_
            eval_step += 1
        if eval_generator.pointer < len(eval_generator.data_index):
            actual_len, x_batch, y_batch1, label_batch1_,\
                y_batch2, label_batch2_,\
                y_batch3, label_batch3_ = eval_generator.rest_batch(config.batch_size)
            x_batch = np.expand_dims(x_batch,axis=3)
            _, _, _, eval_yhat1_, eval_yhat2_, eval_yhat3_,\
                        score1_, score2_, score3_ = dev_step(x_batch, y_batch1, y_batch2, y_batch3)
            eval_yhat1[(eval_step-1)*config.batch_size : len(eval_generator.data_index)] = eval_yhat1_
            score1[(eval_step-1)*config.batch_size : len(eval_generator.data_index),:] = score1_
            eval_yhat2[(eval_step-1)*config.batch_size : len(eval_generator.data_index)] = eval_yhat2_
            score2[(eval_step-1)*config.batch_size : len(eval_generator.data_index),:] = score2_
            eval_yhat3[(eval_step-1)*config.batch_size : len(eval_generator.data_index)] = eval_yhat3_
            score3[(eval_step-1)*config.batch_size : len(eval_generator.data_index),:] = score3_
        eval_yhat1 = eval_yhat1 + 1
        eval_yhat2 = eval_yhat2 + 1
        eval_yhat3 = eval_yhat3 + 1
        eval_fscore1 = f1_score(eval_generator.label[:-1], eval_yhat1[1:], average='macro')
        eval_acc1 = accuracy_score(eval_generator.label[:-1], eval_yhat1[1:])
        eval_kappa1 = cohen_kappa_score(eval_generator.label[:-1], eval_yhat1[1:])
        eval_fscore2 = f1_score(eval_generator.label, eval_yhat2, average='macro')
        eval_acc2 = accuracy_score(eval_generator.label, eval_yhat2)
        eval_kappa2 = cohen_kappa_score(eval_generator.label, eval_yhat2)
        eval_fscore3 = f1_score(eval_generator.label[1:], eval_yhat3[:-1], average='macro')
        eval_acc3 = accuracy_score(eval_generator.label[1:], eval_yhat3[:-1])
        eval_kappa3 = cohen_kappa_score(eval_generator.label[1:], eval_yhat3[:-1])
        savemat(os.path.join(out_path, "eval_ret_model_acc.mat"), dict(acc1=eval_acc1,
                                                                       fscore1=eval_fscore1,
                                                                       kappa1=eval_kappa1,
                                                                       acc2=eval_acc2,
                                                                       fscore2=eval_fscore2,
                                                                       kappa2=eval_kappa2,
                                                                       acc3=eval_acc3,
                                                                       fscore3=eval_fscore3,
                                                                       kappa3=eval_kappa3,
                                                                       yhat1=eval_yhat1,
                                                                       yhat2=eval_yhat2,
                                                                       yhat3=eval_yhat3,
                                                                       score1 = score1,
                                                                       score2 = score2,
                                                                       score3 = score3))
        eval_generator.reset_pointer()
