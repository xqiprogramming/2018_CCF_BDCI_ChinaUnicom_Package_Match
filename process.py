# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:26:36 2021

@author: Admin
"""
import pandas as pd
# import numpy as np
import multiprocessing
from gensim.models import Word2Vec
# from gensim.models.word2vec import LineSentence

# 转换格式:


def change_dtype(data):
    data = data.replace('\\N', 0)
    float_type = ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee']
    int_type = ['gender', 'age']
    for i in float_type:
        data[i] = data[i].astype('float64')

    for j in int_type:
        data[j] = data[j].astype('int64')

    return data

# 处理total_fee为负数的情况


def deal_total_minus_fee(data):
    for i in range(2, 5):
        index = data[data['{0}_total_fee'.format(i)] < 0].index
        data.loc[index, '{0}_total_fee'.format(i)] = \
            data.loc[index, '{0}_total_fee'.format(
                i-1)] - data.loc[index, '{0}_total_fee'.format(i)]
    return data

# 数据预处理


def is_int(x):
    return 1 if int(x) - x == 0 else 0


def pre_process(data,is_train=True):
    data = change_dtype(data)
    data = deal_total_minus_fee(data)
    judge_int_col = ['1_total_fee', '2_total_fee',
                     '3_total_fee', '4_total_fee']
    data['former_complaint_fee'] = data['former_complaint_fee'] / 100
    for c in judge_int_col:
        data['is_int_%s' % c] = data[c].apply(is_int)
    for c in ['1_total_fee', '2_total_fee', '3_total_fee', '4_total_fee',
              'month_traffic', 'last_month_traffic', 'local_trafffic_month',
              'local_caller_time', 'service1_caller_time', 'service2_caller_time',
              'many_over_bill', 'contract_type', 'contract_time', 'pay_num']:
        data[c] = data[c].round(4)
    # 删除compla_fee异常
    if is_train:
        complain_fee_err = data['former_complaint_fee'].quantile([0.99])[0.99]
        data.drop(index=data[data['former_complaint_fee']
                             > complain_fee_err].index, inplace=True)
    return data

###特征工程####

# 统计计数


def feature_count(data, features=[]):
    if len(set(features)) != len(features):
        print('equal feature !!!!')
        return data
    new_feature = 'count'
    for i in features:
        new_feature += '_' + i.replace('add_', '')
    try:
        del data[new_feature]
    except:
        pass
    temp = data.groupby(features)['user_id'].size(
    ).reset_index().rename(columns={'user_id': new_feature})
    data = data.merge(temp, 'left', on=features)
    return data

# word2vec


def word2vec(data, size=10, path=None):
    c = ['{0}_total_fee'.format(i) for i in range(1, 5)]
    col = ['vec%s_%s' % (j, i) for i in c for j in range(10)]
    sentence = []
    w2v_results = []
    if path:
        try:
            w2v = pd.read_csv(path)
            data = pd.concat([data, w2v], axis=1)
            return data
        except:
            print('文件读取失败！')
    for line in list(data[c].values):
        sentence.append([str(float(l)) for idx, l in enumerate(line)])
    print('word2vec开始训练...')
    model = Word2Vec(sentence, size=size, window=2, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=10)
    for s in sentence:
        w2v_results.append(model[s].flatten())
    w2v_results = pd.DataFrame(w2v_results, columns=col)
    if path:
        print('embedding向量存储...')
        w2v_results.to_csv(path, index=False)
        print('embedding存储完成...')
    data = pd.concat([data, w2v_results], axis=1)
    return data


def featured(data, w2v_vec_size=10, w2v_vec_path=None):
    # 1-4个月费用的平均值、方差等
    data['total_fee_mean'] = data.loc[:,
                                      '1_total_fee':'4_total_fee'].apply('mean', axis=1)
    data['total_fee_min'] = data.loc[:,
                                     '1_total_fee':'4_total_fee'].apply('min', axis=1)
    data['total_fee_max'] = data.loc[:,
                                     '1_total_fee':'4_total_fee'].apply('max', axis=1)
    data['total_fee_std'] = data.loc[:,
                                     '1_total_fee':'4_total_fee'].apply('std', axis=1)
    #
    data['last_month_traffic_rest'] = data['month_traffic'] - \
        data['last_month_traffic']
    data['pay_num_pertime'] = data['pay_num'] / data['pay_times']
    # 次数统计特征
    data = feature_count(data, ['1_total_fee'])
    data = feature_count(data, ['2_total_fee'])
    data = feature_count(data, ['3_total_fee'])
    data = feature_count(data, ['4_total_fee'])
    data = feature_count(data, ['former_complaint_fee'])
    data = feature_count(data, ['pay_num'])
    data = feature_count(data, ['contract_time'])
    data = feature_count(data, ['last_month_traffic'])
    data = feature_count(data, ['online_time'])
    # 次数统计特征
    data = feature_count(data, ['1_total_fee'])
    data = feature_count(data, ['2_total_fee'])
    data = feature_count(data, ['3_total_fee'])
    data = feature_count(data, ['4_total_fee'])
    data = feature_count(data, ['former_complaint_fee'])
    data = feature_count(data, ['pay_num'])
    data = feature_count(data, ['contract_time'])
    data = feature_count(data, ['last_month_traffic'])
    data = feature_count(data, ['online_time'])

    data['diff_total_fee_1'] = data['1_total_fee'] - data['2_total_fee']
    data['diff_total_fee_2'] = data['2_total_fee'] - data['3_total_fee']
    data['diff_total_fee_3'] = data['3_total_fee'] - data['4_total_fee']
    data['pay_num_1_total_fee'] = data['pay_num'] - data['1_total_fee']
    data['last_month_traffic_rest'] = data['month_traffic'] - \
        data['last_month_traffic']
    data.loc[data['last_month_traffic_rest']
             < 0, 'last_month_traffic_rest'] = 0
    # 占比
    data['total_caller_time'] = data['service2_caller_time'] + \
        data['service1_caller_time']
    data['service2_caller_ratio'] = data['service2_caller_time'] / \
        (data['total_caller_time']+0.0001)
    data['local_caller_ratio'] = data['local_caller_time'] / \
        (data['total_caller_time']+0.0001)

    data['total_month_traffic'] = data['local_trafffic_month'] + \
        data['month_traffic']
    data['month_traffic_ratio'] = data['month_traffic'] / \
        (data['total_month_traffic']+0.0001)
    data['last_month_traffic_ratio'] = data['last_month_traffic'] / \
        (data['total_month_traffic']+0.0001)

    data['1_total_fee_call_fee'] = data['1_total_fee'] - \
        data['service1_caller_time'] * 0.15
    data['1_total_fee_call2_fee'] = data['1_total_fee'] - \
        data['service2_caller_time'] * 0.15

    data = word2vec(data, size=w2v_vec_size, path=w2v_vec_path)
    return data


if __name__ == "__main__":
    TRAIN_PATH = "./operator/data_round1/train_all.csv"
    TEST_PATH = "./operator/data_round1/test_1.csv"
    W2V_PATH = "./w2v_total_fee.csv"
    data = pd.read_csv(TRAIN_PATH)
    data = pre_process(data)
    data = featured(data, 10, W2V_PATH)
    print(data.shape)
