# -*- coding: utf-8 -*-


import os
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.externals import joblib

from model.hebr import HEBR_Cell


class HEBR_TSC():
    def __init__(self):
        self.param = None

    def set_configuration(self, params):
        self.param = params

    def build_model(self, is_training):

        self.x = tf.placeholder(tf.float32, [None, self.param.his_len, self.param.input_dim], name='input_raw_sequence')
        self.y = tf.placeholder(tf.int32, [None, ], name='input_label')

        y_clf = tf.one_hot(self.y, self.param.n_event)
        user_x = self.x[:, :, :3]
        area_x = self.x[:, :, 3][:, :, np.newaxis]
        climate_x = self.x[:, :, 4:]

        model = HEBR_Cell()
        model.set_configuration(self.param.his_len, self.param.dims, is_training=is_training)
        emb_logits, attention_logits = model.get_embedding(user_x, area_x, climate_x)

        self.attention_score = tf.nn.softmax(attention_logits)

        # output
        patterns = tf.reduce_mean(tf.reshape(emb_logits, [-1, self.param.his_len, self.param.dims['user_area_climate_hidden']]), axis=1)
        net = tf.layers.dense(patterns, 512, activation=tf.nn.relu, name="outnet_fc1")
        out_logits = tf.layers.dense(net, self.param.n_event, activation=tf.nn.relu, name='outnet_fc2')

        # loss and train
        self.net_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y_clf, out_logits))
        self.net_optim = tf.train.AdamOptimizer(self.param.learning_rate, beta1=0.7).minimize(self.net_loss)

        # classification test
        patterns_x = tf.reduce_mean(self.x, axis=1)
        self.patterns = tf.concat([patterns, patterns_x], axis=-1)

        self.clf = XGBClassifier(n_estimators=500, max_depth=6, n_jobs=self.param.n_jobs, scale_pos_weight=70)

        self.saver = tf.train.Saver()

        return 0

    def fit(self, sess, dataloader=None):

        if not dataloader:
            raise Exception('no data input')

        loss = []
        global_features = []
        for _ in range(dataloader.num_batch):
            x_batch, y_batch = dataloader.next_batch()
            feeds = {
                self.y: y_batch,
                self.x: x_batch
            }
            _, loss_net, feature_ = sess.run([self.net_optim, self.net_loss, self.patterns], feed_dict=feeds)
            loss.append(loss_net)
            global_features.append(feature_)

        global_features = np.concatenate(global_features)
        self.clf.fit(global_features, dataloader.y)

        return np.mean(loss)

    def predict(self, sess, dataloader=None):

        if not dataloader:
            raise Exception('no data input')

        global_features = []
        for _ in range(dataloader.num_batch):
            x_batch, _ = dataloader.next_batch()
            feeds = {
                self.x: x_batch
            }
            feature_ = sess.run(self.patterns, feed_dict=feeds)
            global_features.append(feature_)
        global_features = np.concatenate(global_features)
        return self.clf.predict(global_features), self.clf.predict_proba(global_features)

    def store(self, path, sess=None):
        save_model_name = "hebr_{}".format(self.param.data_name)
        if not os.path.exists(os.path.join(path, save_model_name)):
            os.makedirs(os.path.join(path, save_model_name))

        self.saver.save(sess, os.path.join(path, save_model_name, 'model'))
        joblib.dump(self.clf, os.path.join(path, save_model_name, "clf.model"))

    def restore(self, path, sess=None):
        save_model_name = "hebr_{}".format(self.param.data_name)
        self.saver.restore(sess, os.path.join(path, save_model_name, 'model'))
        self.clf = joblib.load(os.path.join(path, save_model_name, "clf.model"))


if __name__ == '__main__':
    pass