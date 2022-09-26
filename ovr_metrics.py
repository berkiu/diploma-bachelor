from __future__ import annotations
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score


def check_metrics(y_test, y_pred, y_pred_proba) -> dict:
    metrics, f1, precision, recall, roc_auc = {}, {}, {}, {}, {}
    for item in [None, 'micro', 'macro', 'weighted']:
        f1[str(f'{item}')] = f1_score(y_test, y_pred, average=item)
        precision[str(f'{item}')] = precision_score(y_test, y_pred, average=item)
        recall[str(f'{item}')] = recall_score(y_test, y_pred, average=item)
    for item in ['macro', 'weighted']:
        roc_auc[str(f'{item}')] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=item)
    metrics['f1'], metrics['precision'], metrics['recall'], metrics['roc_auc'] = f1, precision, recall, roc_auc
    return metrics
