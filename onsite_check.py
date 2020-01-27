# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd
import tensorflow as tf

from data_factory import BatchLoader
from configs import BasicParam
from model import HEBR_TSC

def runmodel(dataloader, configs):

    model = HEBR_TSC()
    model.set_configuration(configs)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(gpu_options=gpu_options)

    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        model.build_model(is_training=False)
        model.restore(configs.save_path, sess=sess)
        y_pred, y_proba = model.predict(sess, dataloader)
        results = pd.DataFrame(np.concatenate([dataloader.y, y_pred[:, np.newaxis], y_proba], axis=1), dtype=np.float64, columns=['tqid', 'userid', 'is_theft', 'prob0', 'prob1'])
        results[['tqid', 'userid']] = results[['tqid', 'userid']].astype(int)
        return results

def loaddata(datapath):

    def norm(data):
        data = np.nan_to_num(data)
        return (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-20)

    datax = np.load(os.path.join(datapath, 'datax.npy'))
    datainfo = np.load(os.path.join(datapath, 'datainfo.npy'))

    data, info = [], []

    for d, i in zip(datax, datainfo):
        x = d[-180:, [1, 3, 5, 6, 7, 8]]     # Total，on-peak，off-peak，ntl，high temperature，low temperature
        x = norm(x)                          # shape: [his_len, dim]
        data.append(x)
        info.append(i)

    data, info = np.asarray(data, dtype=np.float32), np.asarray(info)
    print(data.shape, len(np.unique(info[:, 0])))

    return data, info


if __name__ == '__main__':

    configs = BasicParam()
    configs.dims['user_area'] = 2 * configs.dims['user_hidden']
    configs.dims['user_climate'] = 2 * configs.dims['user_hidden']
    configs.dims['user_area_climate'] = 2 * configs.dims['user_area_hidden']
    configs.batch_size = 52

    os.environ["CUDA_VISIBLE_DEVICES"] = configs.gpu

    # establish dataloader
    data, info = loaddata('../repo/data/hangzhou/')
    loader = BatchLoader(configs.batch_size)
    loader.load_data(data, info, shuffle=False)

    # output list
    result = runmodel(loader, configs)
    result.to_csv('../repo/data/hangzhou/result.csv', index=False)

