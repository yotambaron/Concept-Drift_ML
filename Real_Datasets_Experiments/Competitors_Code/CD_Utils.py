import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.stats import iqr
from sklearn import metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score, roc_auc_score, classification_report
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier


def bins_disc(data, cut_points, start_zero_flag):
    data_new = data.copy()
    if start_zero_flag:
        category = 0
    else:
        category = 1
    for cut in range(len(cut_points)):
        point = cut_points[cut]
        if cut == 0:
            data_new = np.where(data <= point, category, data)
        else:
            data_new = np.where(((data <= point) & (data > cut_points[cut - 1])), category, data_new)
        category += 1
    data_new = np.where(data > cut_points[len(cut_points) - 1], category, data_new)
    return data_new


def remove_outliers(x):
    q1_quantile = np.quantile(x, 0.25)
    q3_quantile = np.quantile(x, 0.75)
    h = 1.5 * iqr(x)
    f = x.copy()
    f = f[(f > q1_quantile - h) & (f < q3_quantile + h)]
    return f


def robust_equal_bins(data, bins):
    new_data = data.copy()
    data = remove_outliers(data)
    data = data.reset_index(drop=True)
    disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    disc.fit(np.asarray(data).reshape(-1, 1))
    new_data = disc.transform(np.asarray(new_data).reshape(-1, 1)).reshape(-1)
    return new_data


def predict_competitors(x, y, jump, window_days, classes):
    random.seed(28)
    ind = jump * window_days
    break_points = np.arange(start=0, stop=y.shape[0] + jump, step=jump)
    neighbors = 11

    y_pred_adwin = np.zeros(y.shape[0])
    y_adwin_proba = np.zeros([y.shape[0], len(classes)])
    y_pred_awe = np.zeros(y.shape[0])
    y_awe_proba = np.zeros([y.shape[0], len(classes)])
    y_pred_dwm = np.zeros(y.shape[0])
    y_dwm_proba = np.zeros([y.shape[0], len(classes)])
    y_pred_lnse = np.zeros(y.shape[0])
    y_lnse_proba = np.zeros([y.shape[0], len(classes)])
    y_pred_srpc = np.zeros(y.shape[0])
    y_srpc_proba = np.zeros([y.shape[0], len(classes)])

    knn_adwin = KNNADWINClassifier(n_neighbors=neighbors)
    awe = AccuracyWeightedEnsembleClassifier(window_size=jump)
    dwm = DynamicWeightedMajorityClassifier(period=jump)  # , n_estimators=jump)
    lnse = LearnPPNSEClassifier(window_size=jump)
    srpc = StreamingRandomPatchesClassifier(n_estimators=5, subspace_size=60)

    Xt, yt = x[:neighbors], y[:neighbors]
    for j in range(neighbors):
        knn_adwin.partial_fit(Xt[j].reshape(1, -1), yt[j].reshape(1), classes=classes)

    # Predict randomly the first window days
    for i in tqdm(range(window_days * jump)):
        y_pred_adwin[i] = knn_adwin.predict(x[i].reshape(1, -1))
        if len(knn_adwin.predict_proba(x[i].reshape(1, -1))[0]) == 1:
            adwin_proba = np.zeros([1, len(classes)]) + 1 / len(classes)
        else:
            adwin_proba = np.double(knn_adwin.predict_proba(x[i].reshape(1, -1))) / sum(sum(np.double(knn_adwin.predict_proba(x[i].reshape(1, -1)))))  # [0, 1])
        y_pred_awe[i] = awe.predict(x[i].reshape(1, -1))
        # awe_proba = np.double(awe.predict_proba(x[k].reshape(1, -1)))/sum(sum(np.double(awe.predict_proba(X_day_i[k].reshape(1, -1)))))#[0, 1])
        y_pred_dwm[i] = dwm.predict(x[i].reshape(1, -1))
        # dwm_proba = np.double(dwm.predict_proba(x[k].reshape(1, -1)))/sum(sum(np.double(dwm.predict_proba(X_day_i[k].reshape(1, -1)))))#[0, 1])
        y_pred_lnse[i] = lnse.predict(x[i].reshape(1, -1))
        if len(lnse.predict_proba(x[i].reshape(1, -1))[0]) == 1:
            lnse_proba = np.zeros([1, len(classes)]) + 1 / len(classes)
        else:
            lnse_proba = np.double(lnse.predict_proba(x[i].reshape(1, -1))) / sum(sum(np.double(lnse.predict_proba(x[i].reshape(1, -1)))))  # [0, 1])
        y_pred_srpc[i] = srpc.predict(x[i].reshape(1, -1))
        if len(srpc.predict_proba(x[i].reshape(1, -1))[0]) == 1:
            srpc_proba = np.zeros([1, len(classes)]) + 1 / len(classes)
        else:
            srpc_proba = np.double(srpc.predict_proba(x[i].reshape(1, -1))) / sum(sum(np.double(srpc.predict_proba(x[i].reshape(1, -1)))))  # [0, 1])

        for c in range(len(classes)):
            y_adwin_proba[i, c] = adwin_proba[0, c]
            # y_awe_proba[i, c] = awe_proba[0, c]
            # y_dwm_proba[i, c] = dwm_proba[0, c]
            y_lnse_proba[i, c] = lnse_proba[0, c]
            y_srpc_proba[i, c] = srpc_proba[0, c]

    # Fit the day before and predict the current day
    for b in tqdm(range(2, len(break_points) - 1)):
        X_day_i_minus = x[break_points[b - window_days]:break_points[b]]
        y_day_i_minus = y[break_points[b - window_days]:break_points[b]]
        X_day_i = x[break_points[b]:break_points[b + 1]]

        # knn_adwin.fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
        # awe.fit(X_day_i_minus, y_day_i_minus)
        # dwm.fit(X_day_i_minus, y_day_i_minus)
        # lnse.fit(X_day_i_minus, np.array(y_day_i_minus),classes=[0, 1])
        # srpc.fit(X_day_i_minus, np.array(y_day_i_minus))

        for j in range(X_day_i_minus.shape[0]):
            knn_adwin.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1), classes=classes)
            awe.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1), classes=classes)
            dwm.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1), classes=classes)
            lnse.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1), classes=classes)
            srpc.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1), classes=classes)

        for k in range(X_day_i.shape[0]):
            y_pred_adwin[ind] = knn_adwin.predict(X_day_i[k].reshape(1, -1))
            if len(knn_adwin.predict_proba(X_day_i[k].reshape(1, -1))[0]) == 1:
                adwin_proba = np.zeros([1, len(classes)]) + 1 / len(classes)
            else:
                adwin_proba = np.double(knn_adwin.predict_proba(X_day_i[k].reshape(1, -1)))/sum(sum(np.double(knn_adwin.predict_proba(X_day_i[k].reshape(1, -1)))))#[0, 1])
            y_pred_awe[ind] = awe.predict(X_day_i[k].reshape(1, -1))
            # awe_proba = np.double(awe.predict_proba(X_day_i[k].reshape(1, -1)))/sum(sum(np.double(awe.predict_proba(X_day_i[k].reshape(1, -1)))))#[0, 1])
            y_pred_dwm[ind] = dwm.predict(X_day_i[k].reshape(1, -1))
            # dwm_proba = np.double(dwm.predict_proba(X_day_i[k].reshape(1, -1)))/sum(sum(np.double(dwm.predict_proba(X_day_i[k].reshape(1, -1)))))#[0, 1])
            y_pred_lnse[ind] = lnse.predict(X_day_i[k].reshape(1, -1))
            if len(lnse.predict_proba(X_day_i[k].reshape(1, -1))[0]) == 1:
                lnse_proba = np.zeros([1, len(classes)]) + 1 / len(classes)
            else:
                lnse_proba = np.double(lnse.predict_proba(X_day_i[k].reshape(1, -1)))/sum(sum(np.double(lnse.predict_proba(X_day_i[k].reshape(1, -1)))))#[0, 1])
            y_pred_srpc[ind] = srpc.predict(X_day_i[k].reshape(1, -1))
            if len(srpc.predict_proba(X_day_i[k].reshape(1, -1))[0]) == 1:
                srpc_proba = np.zeros([1, len(classes)]) + 1 / len(classes)
            else:
                srpc_proba = np.double(srpc.predict_proba(X_day_i[k].reshape(1, -1)))/sum(sum(np.double(srpc.predict_proba(X_day_i[k].reshape(1, -1)))))#[0, 1])
            for c in range(len(classes)):
                y_adwin_proba[ind, c] = adwin_proba[0, c]
                # y_awe_proba[ind, c] = awe_proba[0, c]
                # y_dwm_proba[ind, c] = dwm_proba[0, c]
                y_lnse_proba[ind, c] = lnse_proba[0, c]
                y_srpc_proba[ind, c] = srpc_proba[0, c]
            ind += 1

    return y_pred_adwin, y_pred_awe, y_pred_dwm, y_pred_lnse, y_pred_srpc, y_adwin_proba, y_awe_proba, y_dwm_proba, y_lnse_proba, y_srpc_proba


def compute_classification_scores(pred, probs, true, jump):
    break_points = np.arange(start=0, stop=true.shape[0] + jump, step=jump)
    all_results = np.zeros((len(break_points) - 1, 17))
    day = 1
    cols = ["Day", "TP", "FP", "FN", "TN", "FA", "TP_Precision", "TP_Recall", "TN_Precision", "TN_Recall", "F1_pos",
            "F1_neg", "F1_Avg", "Geometric_mean", "Balanced_Accuracy", "Y_score", "AUC"]

    for j in range(len(break_points) - 1):
        y_pred, y_true = pred[break_points[j]:break_points[j + 1]], true[break_points[j]:break_points[j + 1]]
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        auc = 0.5
        for i in range(y_true.shape[0]):
            if y_pred[i] == 1 and y_true[i] == 1:
                tp += 1
            if y_pred[i] == 0 and y_true[i] == 1:
                fn += 1
            if y_pred[i] == 0 and y_true[i] == 0:
                tn += 1
            if y_pred[i] == 1 and y_true[i] == 0:
                fp += 1
        if tp == 0:
            tp_precision = 0
            tp_recall = 0
        else:
            tp_precision = tp / (tp + fp)
            tp_recall = tp / (tp + fn)
        if tn == 0:
            tn_precision = 0
            tn_recall = 0
        else:
            tn_precision = tn / (tn + fn)
            tn_recall = tn / (tn + fp)
        if (tp_precision + tp_recall) == 0:
            f1_pos = 0
        else:
            f1_pos = 2 * tp_precision * tp_recall / (tp_precision + tp_recall)
        if (tn_precision + tn_recall) == 0:
            f1_neg = 0
        else:
            f1_neg = 2 * tn_precision * tn_recall / (tn_precision + tn_recall)
        if fp + tn == 0:
            fa_rate = 0
        else:
            fa_rate = fp / (fp + tn)

        f1_avg = (f1_pos + f1_neg) / 2
        gm = np.sqrt(tp_recall * tn_recall)
        b_accuracy = (tp_recall + tn_recall) / 2
        y_score = (tp_recall + tn_recall + tp_precision + tn_precision) / 4
        if j > 7:
            fpr, tpr, thresholds = metrics.roc_curve(y_true, probs[break_points[j]:break_points[j + 1]], pos_label=1)
            auc = metrics.auc(fpr, tpr)
        all_results[j] = [day, tp, fp, fn, tn, fa_rate, tn_precision, tp_recall, tn_precision, tn_recall, f1_pos,
                          f1_neg, f1_avg, gm, b_accuracy, y_score, auc]
        day = day + 1
    all_results = pd.DataFrame(all_results)
    all_results.columns = cols
    return all_results


def compute_multiclass_scores(y_pred, y_probs, y_true, labels, jump):
    y_true = np.array(y_true)
    time_steps = np.arange(1, int(len(y_true) / jump + 1), 1)
    break_points = np.arange(0, int(len(y_true)), jump)
    cols = ['Time_Step', 'W_Precision', 'W_Recall', 'W_F1', 'W_AUC', 'Kappa', 'Matthews']  # , 'Log_Loss']

    results = np.zeros([len(time_steps), 7])

    for i, step in enumerate(break_points):
        start = break_points[i]
        end = start + jump

        class_report = classification_report(y_true[start: end], y_pred[start: end], target_names=labels, output_dict=True)
        precision = class_report['weighted avg']['precision']
        recall = class_report['weighted avg']['recall']
        f1 = class_report['weighted avg']['f1-score']
        # auc = 0.5
        auc = roc_auc_score(y_true[start: end], y_probs[start: end], average="weighted", multi_class="ovr")
        kappa = cohen_kappa_score(y_true[start: end], y_pred[start: end])
        matthews = matthews_corrcoef(y_true[start: end], y_pred[start: end])
        # log_loss = log_loss(y_true[start: end], y_probs[start: end])

        results[i] = np.round([time_steps[i], precision, recall, f1, auc, kappa, matthews], 2)  # , log_loss], 2)

    results = pd.DataFrame(results, columns=cols)

    return results


