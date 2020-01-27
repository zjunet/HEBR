# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

from data_factory import BatchLoader
from configs import BasicParam
from model import HEBR_TSC


def getmetric(y, y_pred):
    return {'acc': metrics.accuracy_score(y, y_pred),
            'pre': metrics.precision_score(y, y_pred),
            're': metrics.recall_score(y, y_pred),
            'f1': metrics.f1_score(y, y_pred)}

def splitTrainTest(datax, datay):

    def norm(data):
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-20)

    df = pd.DataFrame({'time': [pd.to_datetime('2018-11-1')]})
    splittamp = (pd.to_datetime(df['time']).astype(int)/ 1e9).values[0]

    trainx, trainy, testx, testy = [], [], [], []
    for i in range(len(datax)):
        x = datax[i][-180:, [1,3,5,6,7,8]]    # Total，on-peak，off-peak，ntl，high temperature，low temperature
        x = norm(x)                           # shape: [his_len, dim]
        y = datay[i]
        if (y[1] >= splittamp) and (y[1] < splittamp+30*24*60*60):
            testx.append(x)
            testy.append(y[0])
        else:
            trainx.append(x)
            trainy.append(y[0])

    trainx = np.asarray(trainx, dtype=np.float32)
    trainy = np.asarray(trainy, dtype=np.int32)
    testx = np.asarray(testx, dtype=np.float32)
    testy = np.asarray(testy, dtype=np.int32)

    return trainx, trainy, testx, testy

if __name__ == '__main__':

    configs = BasicParam()
    configs.dims['user_area'] = 2 * configs.dims['user_hidden']
    configs.dims['user_climate'] = 2 * configs.dims['user_hidden']
    configs.dims['user_area_climate'] = 2 * configs.dims['user_area_hidden']

    os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu

    datax = np.load('./repo/data/months/datax.npy', allow_pickle=True)
    datay = np.load('./repo/data/months/datay.npy', allow_pickle=True)

    trainx, trainy, testx, testy = splitTrainTest(datax, datay)
    print(trainx.shape, trainy.shape, np.sum(trainy), (trainy.shape[0] - np.sum(trainy)) / np.sum(trainy))
    print(testx.shape, testy.shape, np.sum(testy), (testy.shape[0] - np.sum(testy)) / np.sum(testy))

    # establish dataloader
    trainloader = BatchLoader(configs.batch_size)
    trainloader.load_data(trainx, trainy, shuffle=True)
    testloader = BatchLoader(configs.batch_size)
    testloader.load_data(testx, testy, shuffle=False)

    model = HEBR_TSC()
    model.set_configuration(configs)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        model.build_model(is_training=True)
        init_vars = tf.global_variables_initializer()
        sess.run(init_vars)

        bestP = 0.0
        for i in range(100):
            loss = model.fit(sess, trainloader)
            y_pred, _ = model.predict(sess, testloader)
            results = getmetric(testy, y_pred)
            logstr = 'Epochs {:d}, loss {:f}, Accuracy {:f}, Precision {:f}, Recall {:f}, F1 {:f}'.format(i, loss, results['acc'], results['pre'], results['re'], results['f1'])
            print(logstr)
            p = 1.25 * results['pre'] * results['re'] / (0.25 * results['pre'] + results['re'])
            if p > bestP:
                model.store(configs.save_path, sess=sess)
                bestP = p
                print('epoch {} store.'.format(i))

    print("model testing...")
    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        model.build_model(is_training=False)
        model.restore(configs.save_path, sess=sess)
        y_pred, _ = model.predict(sess, testloader)
        results = getmetric(testy, y_pred)
        print('Accuracy {:f}, Precision {:f}, Recall {:f}, F1 {:f}'.format(results['acc'], results['pre'], results['re'], results['f1']))
        print(results['matrix'])


