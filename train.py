# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:52:04 2021

@author: Admin
"""
import pandas as pd
from process import pre_process, featured
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
import lightgbm as lgb


def f1_scores(estimator, X, y_true):
    y_pred = estimator.predict(X)
    score = f1_score(y_true, y_pred, average='macro')
    return score


class Trainer:
    def __init__(self):
        self.grid_searcher = {}

    def search_param(self, data, model, param, features, label, service_type):
        cat_features = ['is_mix_service', 'online_time',
                        'many_over_bill', 'contract_type',
                        'is_promise_low_consume', 'net_service',
                        'gender', 'complaint_level', 'is_int_1_total_fee',
                        'is_int_2_total_fee', 'is_int_3_total_fee',
                        'is_int_4_total_fee']
        for c in cat_features:
            data[c] = data[c].astype('category')
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
        data = data[data['service_type'] ==
                    service_type].reset_index(drop=True)
        self.grid_searcher[service_type] = GridSearchCV(
            model, param, scoring=f1_scores, cv=skf, verbose=2)
        self.grid_searcher[service_type].fit(data[features], data[label])


if __name__ == "__main__":
    TRAIN_PATH = "./operator/data_round1/train_all.csv"
    TEST_PATH = "./operator/data_round1/test_1.csv"
    W2V_PATH = "./w2v_total_fee.csv"
    data = pd.read_csv(TRAIN_PATH)
    data = pre_process(data)
    data = featured(data, 10, W2V_PATH)
    features = data.columns.to_list()
    label = 'current_service'
    remove_lst = ['service_type', '1_total_fee',
                  '2_total_fee', '3_total_fee', '4_total_fee', 'current_service',
                  'user_id']
    for c in remove_lst:
        features.remove(c)
    param = {'num_leaves': [5, 10, 15, 20], 'max_depth': [-1, 2, 5, 10]}
    lgb_model = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=200, subsample=0.65,
                                   subsample_freq=1, class_weight='balanced',
                                   objective='multiclass', metrics='None',
                                   colsample_bytree=0.65, random_state=100)
    tr = Trainer()
    tr.search_param(data, lgb_model, param, features, label, 4)
