# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:52:04 2021

@author: Admin
"""
import pandas as pd
from process import pre_process, featured
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
import lightgbm as lgb
# from lightgbm.callback import reset_parameter


def f1_scores(estimator, X, y_true):
    y_pred = estimator.predict(X)
    score = f1_score(y_true, y_pred, average='macro')
    return score


def lgb_eval_func(y_true, y_pred):
    label_ = np.unique(y_true)
    n_class = len(label_)
    y_pred = np.reshape(y_pred, (-1, n_class))
    y_pred = label_[y_pred.argmax(axis=1)]
    score = f1_score(y_true, y_pred, average='macro')
    return ('eval_f1_score:', score, True)


def stacking_predict(model_lst, X):
    def choice_best(row):
        cnts = row.value_counts().index
        if len(row) != len(cnts):
            return cnts[0]
        else:
            return cnts[0]
    resltus = {}
    for i, model in enumerate(model_lst):
        pred = model.predict(X)
        resltus['model' + str(i+1)] = pred
    resltus = pd.DataFrame(resltus)
    resltus = resltus.apply(choice_best, axis=1)
    return resltus


def valid_score(file, w2v_path, model_lst, features, label, service_type):
    data = pd.read_csv(file)
    data = pre_process(data, False)
    data = featured(data, 10, w2v_path)
    pred = stacking_predict(
        model_lst, data[data['service_type'] == service_type][features])
    score = f1_score(data[data['service_type'] == service_type][label], pred,
                     average='macro')
    return score


class Trainer:
    def __init__(self):
        self.grid_searcher = {}
        self.stacking_models = {}
        self.cv_scores = {}

    def search_param(self, data, model, param, features, label, service_type):

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
        data = data[data['service_type'] ==
                    service_type].reset_index(drop=True)
        self.grid_searcher[service_type] = GridSearchCV(
            model, param, scoring=f1_scores, cv=skf, verbose=2)
        self.grid_searcher[service_type].fit(data[features], data[label])

    def train(self, data, model, features, label, service_type):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
        data = data[data['service_type'] ==
                    service_type].reset_index(drop=True)
        models = []
        scores_ = []
        params = model.get_params()
        params.pop('random_state', None)
        for i, (train_idx, test_idx) in enumerate(skf.split(data[features], data[label])):
            model_ = lgb.LGBMClassifier(random_state=2020+i, **params)
            train_set_X = data.loc[train_idx, features]
            train_set_y = data.loc[train_idx, label]
            eval_set = (data.loc[test_idx, features],
                        data.loc[test_idx, label])
            model_.fit(train_set_X, train_set_y, verbose=50,
                       early_stopping_rounds=3, eval_set=eval_set)
            y_pred = model_.predict(eval_set[0])
            score = f1_score(eval_set[1], y_pred, average='macro')
            scores_.append(score)
            models.append(model_)
        self.stacking_models[service_type] = models
        self.cv_scores[service_type] = scores_


if __name__ == "__main__":
    TRAIN_PATH = "C:/Users/86182/prac/sam1/src/train_1.csv"
    TEST_PATH = "C:/Users/86182/prac/sam1/src/test_1.csv"
    W2V_PATH = "C:/Users/86182/prac/sam1/src/w2v_total_fee.csv"
    TEST_W2V_PATH = "C:/Users/86182/prac/sam1/src/test_w2v_total_fee.csv"
    data = pd.read_csv(TRAIN_PATH)
    data = pre_process(data)
    data = featured(data, 10, W2V_PATH)
    features = data.columns.to_list()
    label = 'current_service'
    # label = 'package_label'
    remove_lst = ['service_type', '1_total_fee',
                  '2_total_fee', '3_total_fee', '4_total_fee', 'current_service',
                  'user_id']
    for c in remove_lst:
        features.remove(c)
    # param = {'num_leaves': [5, 10, 15, 20], 'max_depth': [-1, 2, 5, 10]}
    lgb_model = lgb.LGBMClassifier(learning_rate=0.05, n_estimators=2000, subsample=0.65,
                                   subsample_freq=1, class_weight='balanced',
                                   objective='multiclass',
                                   colsample_bytree=0.65)
    tr = Trainer()
    print('training!')
    for service_type in [4,1]:
        # # tr.search_param(data, lgb_model, param, features, label, 4)
        print('===============','current_type:',service_type,'============')
        tr.train(data, lgb_model, features, label, service_type)
        print('training f1_score:', tr.cv_scores)
        print('testing!')
        test_score = valid_score(TEST_PATH, TEST_W2V_PATH,
                                 tr.stacking_models[4], features, label, service_type)
        print('test f1_score:',test_score)
