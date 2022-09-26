import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from ovr_metrics import check_metrics


def my_cross_validate(estimator, data_train_val: np.ndarray, data_test: np.ndarray,
                      cv: int = 10, random_state: int = 42):
    print('starting cross validate')
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    best_accuracy = 0
    best_estimator = estimator

    for train_index, val_index in kf.split(data_train_val):
        model = estimator
        train_data, val_data = data_train_val[train_index], data_train_val[val_index]
        df_train_data, df_val_data = train_data[0].copy(), val_data[0].copy()
        for item in train_data[1:]:
            df_to_append = item.copy()
            df_train_data = df_train_data.append(df_to_append, sort=False, ignore_index=True)
        for item in val_data[1:]:
            df_to_append = item.copy()
            df_val_data = df_val_data.append(df_to_append, sort=False, ignore_index=True)
        X_train, y_train = df_train_data.drop(['label'], axis=1), df_train_data['label']
        X_val, y_val = df_val_data.drop(['label'], axis=1), df_val_data['label']
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        accuracy = accuracy_score(y_val, prediction)
        if accuracy > best_accuracy:
            best_estimator = model
            best_accuracy = accuracy
            dataset_train_best = pd.concat([X_train, y_train], axis=1)

    df_test_data = data_test[0].copy()
    for item in data_test[1:]:
        df_to_append = item.copy()
        df_test_data = df_test_data.append(df_to_append, sort=False, ignore_index=True)

    X_test, y_test = df_test_data.drop(['label'], axis=1), df_test_data['label']
    dataset_test = pd.concat([X_test, y_test], axis=1)
    y_pred = best_estimator.predict(X_test)
    y_pred_proba = best_estimator.predict_proba(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_metrics = check_metrics(y_test, y_pred, y_pred_proba)
    metrics = {'best_train_accuracy': best_accuracy, 'test_accuracy': test_accuracy,
           'other_test_metrics': test_metrics}
    return best_estimator, dataset_train_best, dataset_test, metrics
