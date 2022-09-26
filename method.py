import pandas as pd
import os
from xgboost import XGBClassifier
from parse_and_create_trees import *


def analyze_features(root='/home/kirb/work/ISP-projects/xgboost_diploma/', data_type='law', n_classes=5):
    model = XGBClassifier(learning_rate=0.8,
                          n_estimators=300,
                          booster="gbtree",
                          tree_method="gpu_hist",
                          max_depth=5,
                          random_state=42,
                          verbosity=0)

    train = pd.read_csv(os.path.join(root, f'{data_type}_train.csv'))
    test = pd.read_csv(os.path.join(root, f'{data_type}_test.csv'))

    model.load_model(os.path.join(root, f'{data_type}_best_model.json'))
    dump_list = model.get_booster().get_dump()

    xgb_parser = XgbModelParser(n_classes)

    feature_to_ind = {}
    i = 0
    for feature in list(train.columns)[:-1]:
        feature_to_ind[feature] = i
        i += 1

    forest_train = xgb_parser.get_xgb_model_from_memory(dump_list, 300*n_classes, train.values, feature_to_ind)
    forest_test = xgb_parser.get_xgb_model_from_memory(dump_list, 300*n_classes, test.values, feature_to_ind)
    forest_train.fit_trees()
    forest_test.fit_trees()

    gains_train, gains_test = {}, {}
    for item in list(feature_to_ind.keys()):
        gains_train[item] = []
        gains_test[item] = []

    gains_train = forest_train.collect_gains(gains_train)
    gains_test = forest_test.collect_gains(gains_test)

    feature_avg_gain_test = [(item[0], np.mean(np.array(item[1]))) for item in list(gains_test.items())]
    feature_avg_gain_train = [(item[0], np.mean(np.array(item[1]))) for item in list(gains_train.items())]

    # Отлов NaN фичей
    nan_features = []
    for i in range(len(feature_avg_gain_train)):
        if np.isnan(feature_avg_gain_train[i][1]):
            nan_features.append(feature_avg_gain_train[i][0])

    # Отлов фичей с 0 гейном
    zero_features = []
    for i in range(len(feature_avg_gain_train)):
        if feature_avg_gain_train[i][1] == 0.0 or feature_avg_gain_test[i][1] == 0.0:
            zero_features.append(feature_avg_gain_train[i][0])

    # Отлов фичей с разницей в гейнах в 100/1000/10000 раз
    features_100 = []
    features_1000 = []
    features_10000 = []
    for i in range(len(feature_avg_gain_train)):
        if feature_avg_gain_train[i][1] != 0.0 and feature_avg_gain_test[i][1] != 0.0:
            if feature_avg_gain_train[i][1] / feature_avg_gain_test[i][1] >= 100.0 or \
                    feature_avg_gain_test[i][1] / feature_avg_gain_train[i][1] >= 100.0:
                features_100.append(feature_avg_gain_train[i][0])
            if feature_avg_gain_train[i][1] / feature_avg_gain_test[i][1] >= 1000.0 or \
                    feature_avg_gain_test[i][1] / feature_avg_gain_train[i][1] >= 1000.0:
                features_1000.append(feature_avg_gain_train[i][0])
            if feature_avg_gain_train[i][1] / feature_avg_gain_test[i][1] >= 10000.0 or \
                    feature_avg_gain_test[i][1] / feature_avg_gain_train[i][1] >= 10000.0:
                features_10000.append(feature_avg_gain_train[i][0])

    return nan_features, zero_features, features_100, features_1000, features_10000
