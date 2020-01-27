# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
import argparse

from data_factory import dataprocessing
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


def getparams():
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--data_file', type=str, default='data/', help='path of input file')
    parser.add_argument('-o', '--output_file', type=str, default='data/result.csv', help='path of output file')
    parser.add_argument('-hl', '--history_lenght', type=int, default=180,
                        help='the historical length for observed data, default value is 180')
    parser.add_argument('-b', '--batch_size', type=int, default=1000,
                        help='the number of samples in each batch, default value is 1000')
    parser.add_argument('-e', '--num_epoch', type=int, default=100, help='number of epoch, default value is 100')
    parser.add_argument('-n', '--cpu_jobs', type=int, default=os.cpu_count(),
                        help='number of cpu jobs, default value is maximum number of cpu kernel')
    parser.add_argument('-g', '--gpu_id', type=int, default=0,
                        help='index of gpu, default value is 0')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.001)


    parser.add_argument('-iu', '--user_dims', type=int, default=16,
                        help='dimension of micro-level memory matrix, default value is 16')
    parser.add_argument('-il', '--ntl_dims', type=int, default=4,
                        help='dimension of meso-level memory matrix, default value is 4')
    parser.add_argument('-ie', '--climate_dims', type=int, default=8,
                        help='dimension of macro-level memory matrix, default value is 8')
    parser.add_argument('-iul', '--user_ntl_dims', type=int, default=64,
                        help='dimension of user-area memory matrix, default value is 64')
    parser.add_argument('-iue', '--user_climate_dims', type=int, default=64,
                        help='dimension of user-climate memory matrix, default value is 64')
    parser.add_argument('-iule', '--user_ntl_climate_dims', type=int, default=256,
                        help='dimension of user_ntl_climate memory matrix, default value is 256')

    args = parser.parse_args()

    configs = BasicParam()
    configs.data_path = args.data_file
    configs.his_len = args.history_file
    configs.n_jobs = args.cpu_jobs
    configs.gpu = str(args.gpu_id)
    configs.batch_size = args.batch_size
    configs.learning_rate = args.learning_rate
    configs.epoch = args.num_epoch

    configs.dims['user_hidden'] = args.user_dims
    configs.dims['area_hidden'] = args.ntl_dims
    configs.dims['climate_hidden'] = args.climate_dims
    configs.dims['user_area_hidden'] = args.user_ntl_dims
    configs.dims['user_climate_hidden'] = args.user_climate_dims
    configs.dims['user_area_climate_hidden'] = args.user_ntl_climate_dims

    configs.dims['user_area'] = 2 * configs.dims['user_hidden']
    configs.dims['user_climate'] = 2 * configs.dims['user_hidden']
    configs.dims['user_area_climate'] = 2 * configs.dims['user_area_hidden']

    return configs


if __name__ == '__main__':

    configs = getparams()
    os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu

    # data processing
    dataprocessing(configs.data_path)

    # load data
    datax = np.load(os.path.join(configs.data_path, 'datax.npy'), allow_pickle=True)
    datay = np.load(os.path.join(configs.data_path, 'datay.npy'), allow_pickle=True)

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


