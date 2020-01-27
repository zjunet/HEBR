# -*- coding: utf-8 -*-


import sys, os
import pandas as pd
import numpy as np
from data_factory.temperature_spider import getTemperatureData


def loadNTL(path):
    lineloss = pd.read_csv(path)
    lineloss['Date'] = pd.to_datetime(lineloss['Date'])
    lineloss = lineloss.sort_values(['AreaID', 'Date'])
    lineloss['Date'] = lineloss['Date'].astype(int)
    lineloss['Date'] = (lineloss['Date'] / 1e9).astype(int)
    return lineloss

def loadUser(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        userdata = []
        for file in files:
            temp = pd.read_csv(os.path.join(path, file))
            userdata.append(temp)
        userdata = pd.concat(userdata, ignore_index=True)
    else:
        userdata = pd.read_csv(path)

    userdata['Date'] = pd.to_datetime(userdata['Date'])
    userdata = userdata.sort_values(['AreaID', 'UserID', 'Date'])
    userdata['Date'] = userdata['Date'].astype(int)
    userdata['Date'] = (userdata['Date'] / 1e9).astype(int)

    return userdata

def loadTemperature(path, need_spider=False, **kwargs):
    if need_spider:
        city_ids = kwargs['cities']
        years = kwargs['years']
        months = kwargs['months']

        if not os.path.isdir(path):
            storepath = os.path.dirname(path)
        else:
            storepath = path

        getTemperatureData(city_ids, years, months, storepath)

    temperaturedata = pd.read_csv(path)
    temperaturedata['date'] = pd.to_datetime(temperaturedata['Date'], format='%Y年%m月%d日')
    temperaturedata['high'] = temperaturedata['temperature_high'].map(lambda x: x.split('℃')[0])
    temperaturedata['low'] = temperaturedata['temperature_low'].map(lambda x: x.split('℃')[0])
    temperaturedata = temperaturedata[['date', 'high', 'low']].astype(int)
    temperaturedata = temperaturedata[['date', 'high', 'low']].astype(int)
    temperaturedata['date'] = (temperaturedata['date'] / 1e9).astype(int)

    return temperaturedata

# abstract data
def abstractData(userdata, ntldata, tempdata):

    datax = []
    datainfo = []

    tq_ids = pd.unique(ntldata['AreaID'])
    num = 0

    length = len(tq_ids)
    bar_length = 20
    percent = 1.0
    for tq_id in tq_ids:
        # 进度条
        hashes = '#' * int(percent / length * bar_length)
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, int(percent * 100 / length)))
        sys.stdout.flush()
        percent += 1

        tdata = ntldata[ntldata['AreaID'] == tq_id][['Date', 'NTL']].values
        tempuser = userdata[userdata['AreaID'] == tq_id]
        if tempuser.shape[0] == 0:
            num += 1

        user_ids = pd.unique(tempuser['UserID'])
        for user_id in user_ids:
            udata = tempuser[tempuser['UserID'] == user_id][['Date', 'Total', 'Top', 'Peak', 'Flat', 'Valley']].values

            datax.append([udata, tdata, tempdata.values])
            datainfo.append([tq_id, user_id])
    sys.stdout.write('\n')
    print("Num_area: {}, Num_Outlier: {}".format(length, num))
    return datax, datainfo

# establish dataset
def establishDataset(datax, starttime, endtime):
    def getdata(d, starttime):
        tmpd = d[(d[:, 0] < starttime + 24 * 60 * 60) & (d[:, 0] >= starttime)]
        if tmpd.shape[0] == 0:
            tmpd = [starttime] + [0] * (tmpd.shape[1]-1)
        else:
            tmpd = np.max(tmpd, axis=0)
        return tmpd

    tempx = []
    length = len(datax)
    bar_length = 20
    percent = 1.0

    for i in range(length):
        d = datax[i]
        # 进度条
        hashes = '#' * int(percent / length * bar_length)
        spaces = ' ' * (bar_length - len(hashes))
        sys.stdout.write("\rPercent: [%s] %d%%" % (hashes + spaces, int(percent * 100 / length)))
        sys.stdout.flush()
        percent += 1

        ele = np.array(d[0], dtype=float, copy=True)
        ntl = np.array(d[1], dtype=float, copy=True)
        temperature = np.array(d[2], dtype=float, copy=True)

        startstamp = (pd.DataFrame([pd.to_datetime(starttime)]).astype(int) / 1e9).astype(int).values[0, 0]
        endstamp = (pd.DataFrame([pd.to_datetime(endtime)]).astype(int) / 1e9).astype(int).values[0, 0]

        temp = []
        while startstamp < endstamp:
            tmpele = getdata(ele, startstamp)
            tmpll = getdata(ntl, startstamp)
            tmpt = getdata(temperature, startstamp)
            tmp = np.concatenate([tmpele, tmpll[1:], tmpt[1:]])  # time, user(5), ntl, climate(2)

            temp.append(tmp)
            startstamp += 24 * 60 * 60
        temp = np.asarray(temp)
        tempx.append(temp)
    sys.stdout.write('\n')
    tempx = np.asarray(tempx, dtype=float)
    tempx = np.nan_to_num(tempx)
    return tempx


if __name__ == '__main__':
    ntldata = loadNTL('../repo/data/hangzhou/ntl.csv')
    userdata = loadUser('../repo/data/hangzhou/user.csv')
    tempdata = loadTemperature('../repo/data/hangzhou/weather_33401.csv')

    datax, datainfo = abstractData(userdata, ntldata, tempdata)
    data = establishDataset(datax, starttime='2019-6-1', endtime='2019-12-1')

    np.save('../repo/data/hangzhou/datax.npy', data)
    np.save('../repo/data/hangzhou/datainfo.npy', datainfo)

    print(data.shape, np.shape(datainfo))