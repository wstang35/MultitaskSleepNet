import tensorflow as tf
import numpy as np


class CNN1DSleep(object):
    """
    A CNN for audio event classification.
    Uses a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, time_length, freq_length, num_classes, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.float32, [None, time_length, freq_length,1], name="input_x")
        self.input_y1 = tf.placeholder(tf.float32, [None, num_classes], name="input_y1")
        self.input_y2 = tf.placeholder(tf.float32, [None, num_classes], name="input_y2")
        self.input_y3 = tf.placeholder(tf.float32, [None, num_classes], name="input_y3")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # dim expansion
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.device('/gpu:0'), tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, freq_length, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.input_x,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled_max = tf.nn.max_pool(h,
                                            ksize=[1, np.ceil((time_length - filter_size + 1) / 2), 1, 1], # wilson: 2-maxpool: ksize = [1，ceil((time_length - filter_size + 1)/2), 1, 1]
                                            strides=[1, np.floor((time_length - filter_size + 1) / 2), 1, 1], # wilson: 2-maxpool: stride = [1, floor((time_length - filter_size + 1)/2), 1, 1]
                                            padding='VALID',
                                            name="pool")
                pooled_outputs.append(pooled_max)
                # tf.summary.histogram("weights", W)
                # tf.summary.histogram("biases", b)
                # tf.summary.histogram("activations", h)
                # tf.summary.histogram("pooled_max", pooled_max)

        # Combine all the pooled features
        num_maxp = 2
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total*num_maxp])

        # Adaptation weights
        with tf.name_scope("adaptation"):
            Wa = tf.Variable(tf.truncated_normal([num_filters_total*num_maxp], stddev=0.1), name="Wa")
            ba = tf.Variable(tf.constant(0.1, shape=[num_filters_total*num_maxp]), name="ba")
            Wra = tf.tanh(tf.multiply(self.h_pool_flat, Wa) + ba)
            Wra = tf.expand_dims(Wra, -1)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            self.h_drop = tf.expand_dims(self.h_drop, 1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output1"):
            W1 = tf.Variable(tf.truncated_normal([num_filters_total*num_maxp, num_classes], stddev=0.1), name="W1")
            W1_ra = tf.multiply(W1, Wra)
            b1 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b1")
            l2_loss += tf.nn.l2_loss(W1_ra)
            l2_loss += tf.nn.l2_loss(b1)
            self.scores1 = tf.nn.xw_plus_b(self.h_drop, W1_ra, b1, name="scores1")
            self.scores1 = tf.reshape(self.scores1, shape=[-1, 5])
            self.predictions1 = tf.argmax(self.scores1, 1)
            self.predictions1 = tf.reshape(self.predictions1, shape=[-1], name="predictions1")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output2"):
            W2 = tf.Variable(tf.truncated_normal([num_filters_total*num_maxp, num_classes], stddev=0.1), name="W2")
            W2_ra = tf.multiply(W2, Wra)
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            l2_loss += tf.nn.l2_loss(W2_ra)
            l2_loss += tf.nn.l2_loss(b2)
            self.scores2 = tf.nn.xw_plus_b(self.h_drop, W2_ra, b2, name="scores2")
            self.scores2 = tf.reshape(self.scores2, shape=[-1, 5])
            self.predictions2 = tf.argmax(self.scores2, 1)
            self.predictions2 = tf.reshape(self.predictions2, shape=[-1], name="predictions2")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output3"):
            W3 = tf.Variable(tf.truncated_normal([num_filters_total*num_maxp, num_classes], stddev=0.1), name="W3")
            W3_ra = tf.multiply(W3, Wra)
            b3 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b3")
            l2_loss += tf.nn.l2_loss(W3_ra)
            l2_loss += tf.nn.l2_loss(b3)
            self.scores3 = tf.nn.xw_plus_b(self.h_drop, W3_ra, b3, name="scores3")
            self.scores3 = tf.reshape(self.scores3, shape=[-1, 5])
            self.predictions3 = tf.argmax(self.scores3, 1)
            self.predictions3 = tf.reshape(self.predictions3, shape=[-1], name="predictions3")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses1 = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y1, logits=self.scores1)
            losses2 = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y2, logits=self.scores2)
            losses3 = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y3, logits=self.scores3)
            self.loss = tf.reduce_mean(losses1) + tf.reduce_mean(losses3) + tf.reduce_mean(losses2) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions1 = tf.equal(self.predictions1, tf.argmax(self.input_y1, 1))
            self.accuracy1 = tf.reduce_mean(tf.cast(correct_predictions1, "float"), name="accuracy1")

            correct_predictions2 = tf.equal(self.predictions2, tf.argmax(self.input_y2, 1))
            self.accuracy2 = tf.reduce_mean(tf.cast(correct_predictions2, "float"), name="accuracy2")

            correct_predictions3 = tf.equal(self.predictions3, tf.argmax(self.input_y3, 1))
            self.accuracy3 = tf.reduce_mean(tf.cast(correct_predictions3, "float"), name="accuracy3")

            tf.summary.scalar("accuracy1", self.accuracy1)
            tf.summary.scalar("accuracy2", self.accuracy2)
            tf.summary.scalar("accuracy3", self.accuracy3)

        self.summ = tf.summary.merge_all()
