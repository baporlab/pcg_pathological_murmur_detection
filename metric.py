from sklearn.metrics import * 
import numpy as np

def metrics_for_binary_classification(true_value, pred_proba, pos_label = 1, threshold_option = 'j_index'):
    
    fpr, tpr, thresholds = roc_curve(true_value, pred_proba, pos_label = pos_label)
    if threshold_option == 'j_index':
        J = tpr - fpr
        ix = np.argmax(J)
        thresholds = thresholds[ix]
    elif threshold_option == 'fpr_10':
        th_idx = np.where(fpr <= 0.1)[0][-1]
        thresholds = thresholds[th_idx]
        print(fpr[th_idx])
    else:
        thresholds = threshold_option
    
    pred_value = (pred_proba >= thresholds).astype(bool)
    pred_value = pred_value * 1
    cm = confusion_matrix(true_value, pred_value)
    
    tp = cm[1][1]
    fn = cm[1][0]
    fp = cm[0][1]
    tn = cm[0][0]
    
    acc = accuracy_score(true_value, pred_value)
    # bacc = balanced_accuracy_score(true_value, pred_value, sample_weight = np.zeros(len(true_value)) + (true_value*4) + 1 )
    f1 = f1_score(true_value, pred_value, pos_label = pos_label)
    # Sensitivity = recall_score(true_value, pred_value, pos_label = pos_label)
    # Specificity = recall_score(true_value, pred_value, pos_label = 1 - pos_label)
    
    Sensitivity = tp / (tp+fn)
    Specificity = tn / (fp+tn)
    bacc = (Sensitivity + Specificity)/2
    
    mcc = matthews_corrcoef(true_value, pred_value)
    
    auroc = roc_auc_score(true_value, pred_proba)
    auprc = average_precision_score(true_value, pred_proba, pos_label = 1)
    
    metrics = {'TP':tp, 'FN':fn, 'FP':fp, 'TN':tn,
               'ACC':acc, 'BACC':bacc, 'F1':f1, 'Specificity':Specificity, 'Sensitivity':Sensitivity, 'MCC':mcc,
               'AUROC':auroc, 'AUPRC':auprc,
               'Threshold':thresholds, 'CM': [tp, fn, fp, tn]}
    return metrics