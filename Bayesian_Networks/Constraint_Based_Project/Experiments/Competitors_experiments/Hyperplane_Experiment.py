import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from scipy.stats import iqr
from sklearn.preprocessing import KBinsDiscretizer
from skmultiflow.data.hyper_plane_generator import HyperplaneGenerator
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import time


# -------------------------------------------------- Functions ------------------------------------------------------- #

# ----- Equal Bins Disc ----- #

def equal_bins_disc(X, rows, nodes, bins):

    X_discretized = np.zeros([rows, nodes])

    def remove_outliers(x):
        q1_quantile = np.quantile(x, 0.25)
        q3_quantile = np.quantile(x, 0.75)
        H = 1.5 * iqr(x)
        f = x.copy()
        f = f[(f > q1_quantile - H) & (f < q3_quantile + H)]
        return f

    def robust_EB(X, bins):
        X = remove_outliers(X)
        X = X.reset_index(drop=True)
        disc = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
        disc.fit(np.asarray(X).reshape(-1, 1))
        y = disc.transform(np.asarray(X).reshape(-1, 1)).reshape(-1)
        return y

    for i in range(nodes):
        X_discretized[:, i] = robust_EB(pd.Series(X[:, i]), bins=bins)

    return X_discretized + 1


def predict_competitors(X, y, jump, seed):
    random.seed(seed)
    ind1, ind2, ind3, ind4, ind5 = jump, jump, jump, jump, jump
    time_adwin, time_dwm, time_awe, time_lnse, time_srpc = 0, 0, 0, 0, 0
    break_points = np.arange(start=0, stop=y.shape[0] + jump, step=jump)
    neighbors = 5

    y_pred_dwm = np.zeros(y.shape[0])
    y_pred_adwin = np.zeros(y.shape[0])
    y_pred_awe = np.zeros(y.shape[0])
    y_pred_lnse = np.zeros(y.shape[0])
    y_pred_srpc = np.zeros(y.shape[0])

    dwm = DynamicWeightedMajorityClassifier()
    knn_adwin = KNNADWINClassifier(n_neighbors=neighbors)
    awe = AccuracyWeightedEnsembleClassifier()
    lnse = LearnPPNSEClassifier()
    srpc = StreamingRandomPatchesClassifier(n_estimators=10)

    xt, yt = X[:neighbors], y[:neighbors]
    for j in range(neighbors):
        knn_adwin.partial_fit(xt[j].reshape(1, -1), yt[j].reshape(1))

    # Predict randomly the first day #
    for i in tqdm(range(jump)):
        y_pred_dwm[i] = dwm.predict(X[i].reshape(1, -1))
        y_pred_adwin[i] = knn_adwin.predict(X[i].reshape(1, -1))
        y_pred_awe[i] = awe.predict(X[i].reshape(1, -1))
        y_pred_lnse[i] = lnse.predict(X[i].reshape(1, -1))
        y_pred_srpc[i] = srpc.predict(X[i].reshape(1, -1))

    # Fit the day before and predict the current day #
    for b in tqdm(range(1, len(break_points) - 1)):
        X_day_i_minus = X[break_points[b - 1]:break_points[b]]
        y_day_i_minus = y[break_points[b - 1]:break_points[b]]
        X_day_i = X[break_points[b]:break_points[b + 1]]

        # Fit previous day and predict current day - adwin (and save runtime)
        time_start = time.time()
        for j in range(X_day_i_minus.shape[0]):
            knn_adwin.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
        for k in range(X_day_i.shape[0]):
            y_pred_adwin[ind2] = knn_adwin.predict(X_day_i[k].reshape(1, -1))
            ind2 += 1
        time_end = time.time()
        time_adwin += time_end - time_start

        # Fit previous day and predict current day - dwm (and save runtime)
        time_start = time.time()
        for j in range(X_day_i_minus.shape[0]):
            dwm.fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
        for k in range(X_day_i.shape[0]):
            y_pred_dwm[ind1] = dwm.predict(X_day_i[k].reshape(1, -1))
            ind1 += 1
        time_end = time.time()
        time_dwm += time_end - time_start

        # Fit previous day and predict current day - awe (and save runtime)
        time_start = time.time()
        for j in range(X_day_i_minus.shape[0]):
            awe.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
        for k in range(X_day_i.shape[0]):
            y_pred_awe[ind3] = awe.predict(X_day_i[k].reshape(1, -1))
            ind3 += 1
        time_end = time.time()
        time_awe += time_end - time_start

        # Fit previous day and predict current day - lnse (and save runtime)
        time_start = time.time()
        for j in range(X_day_i_minus.shape[0]):
            lnse.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1), classes=[0, 1])
        for k in range(X_day_i.shape[0]):
            y_pred_lnse[ind5] = lnse.predict(X_day_i[k].reshape(1, -1))
            ind5 += 1
        time_end = time.time()
        time_lnse += time_end - time_start

        # Fit previous day and predict current day - srpc (and save runtime)
        time_start = time.time()
        for j in range(X_day_i_minus.shape[0]):
            srpc.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1))
        for k in range(X_day_i.shape[0]):
            y_pred_srpc[ind4] = srpc.predict(X_day_i[k].reshape(1, -1))
            ind4 += 1
        time_end = time.time()
        time_srpc += time_end - time_start

    def measure_over_time(y_pred, y_test, jumping, metric):
        b_points = np.arange(start=0, stop=y_test.shape[0] + jumping, step=jumping)
        accs = np.zeros(len(b_points) - 1)
        for i in range(len(b_points) - 1):
            y_pred_temp, y_test_temp = y_pred[b_points[i]:b_points[i + 1]], y_test[b_points[i]:b_points[i + 1]]
            if metric == 'F1':
                accs[i] = f1_score(y_test_temp, y_pred_temp)
            elif metric == 'Accuracy':
                accs[i] = np.mean(y_test_temp == y_pred_temp)
            elif metric == 'AUC':
                accs[i] = roc_auc_score(y_score=y_pred_temp, y_true=y_test_temp)
            elif metric == 'Precision':
                accs[i] = precision_score(y_pred=y_pred_temp, y_true=y_test_temp)
            elif metric == 'Recall':
                accs[i] = recall_score(y_pred=y_pred_temp, y_true=y_test_temp)
            else:
                raise Exception("Undefined metric")
        return accs

    results_adwin = measure_over_time(y_pred=y_pred_adwin, y_test=y, jumping=jump, metric='Accuracy')
    results_awe = measure_over_time(y_pred=y_pred_awe, y_test=y, jumping=jump, metric='Accuracy')
    results_dwm = measure_over_time(y_pred=y_pred_dwm, y_test=y, jumping=jump, metric='Accuracy')
    results_lnse = measure_over_time(y_pred=y_pred_lnse, y_test=y, jumping=jump, metric='Accuracy')
    results_srp = measure_over_time(y_pred=y_pred_srpc, y_test=y, jumping=jump, metric='Accuracy')

    return results_adwin, results_awe, results_dwm, results_lnse, results_srp, time_adwin, time_dwm, time_awe, time_lnse, time_srpc


# --------- Set perameters ---------- #

working_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Relevent_for_yotam\Concept_drift\CB_Project'

seed = 28
jump = 1500
window = 3000
time_steps = 20
change_point = 10
num_of_experiments = 10
bins = 5
observations = int(jump * time_steps / 3)
steps = np.arange(1, time_steps + 1, 1)
steps = pd.DataFrame(steps.reshape(-1, 1))

sample = 0
competitors = 1
save = 1
plot = 0

# --------- Generate data ---------- #

if sample:
    for p in range(1, num_of_experiments + 1):
        # Stable period
        Hyperplane_stable = HyperplaneGenerator(random_state=seed, n_features=8, n_drift_features=0, mag_change=0, noise_percentage=0.05, sigma_percentage=0)
        X, y = HyperplaneGenerator.next_sample(Hyperplane_stable, observations)
        df_stable = pd.DataFrame(np.hstack((X, np.array([y]).T)))

        # Create drift - Gradual
        Hyperplane_Drift = HyperplaneGenerator(random_state=seed, n_features=8, n_drift_features=4, mag_change=0.0005, noise_percentage=0.05, sigma_percentage=0.05)
        X, y = HyperplaneGenerator.next_sample(Hyperplane_Drift, observations)
        df_drift = pd.DataFrame(np.hstack((X, np.array([y]).T)))
        df = pd.concat([df_stable, df_drift], axis=0)

        # Create drift - Sudden
        # Hyperplane_Drift = HyperplaneGenerator(random_state=seed, n_features=60, n_drift_features=15, mag_change=0.1, noise_percentage=0.05, sigma_percentage=0.2)
        # X, y = HyperplaneGenerator.next_sample(Hyperplane_Drift, observations)
        # df_drift = pd.DataFrame(np.hstack((X, np.array([y]).T)))
        # df = pd.concat([df_stable, df_drift], axis=0)

        # Stable again on the new concept
        Hyperplane_Drift.set_params(n_drift_features=0, mag_change=0, sigma_percentage=0)
        X, y = HyperplaneGenerator.next_sample(Hyperplane_Drift, observations)
        df_stable2 = pd.DataFrame(np.hstack((X, np.array([y]).T)))
        df = pd.concat([df, df_stable2], axis=0)

        # Save data
        df.to_csv(working_path + '/Data/Hyperplane/Hyperplane_data' + str(p) + '.csv', index=False)

        # Descritization of the data and saving it again
        data = pd.read_csv(working_path + '/Data/Hyperplane/Hyperplane_data' + str(p) + '.csv')
        Class = data['8'] + 1
        del data['8']
        columns = data.columns
        X_disc = pd.DataFrame(equal_bins_disc(np.array(data), data.shape[0], data.shape[1], bins))
        X_disc = pd.concat([X_disc, Class], axis=1)
        X_disc.to_csv(working_path + '/Data/Hyperplane/Hyperplane_data' + str(p) + '_disc.csv', index=False)

        print("Finished sampling permutation: ", p)


# ------------------------------------------------- Experiment ------------------------------------------------------- #

res_adwin_mean, res_awe_mean, res_dwm_mean, res_lnse_mean, res_srp_mean = np.zeros(time_steps), np.zeros(time_steps), \
                                                                np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)
t_adwin_mean, t_awe_mean, t_dwm_mean, t_lnse_mean, t_srp_mean = 0, 0, 0, 0, 0

for r in tqdm(range(num_of_experiments)):

    # Run Competitors on generated data
    if competitors:
        df = pd.read_csv(working_path + '/Data/Hyperplane/Hyperplane_data' + str(r+1) + '.csv')
        df.columns = ['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'class']
        X = df[['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8']]
        y = df['class']
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)

        # Predict all competitors
        res_adwin, res_awe, res_dwm, res_lnse, res_srp, t_adwin, t_dwm, t_awe, t_lnse, t_srp = predict_competitors(X.values, y.values, jump, seed)

        # Cumulate acccuracy results from all permutations
        res_adwin_mean += res_adwin
        res_awe_mean += res_awe
        res_dwm_mean += res_dwm
        res_lnse_mean += res_lnse
        res_srp_mean += res_srp
        # Cumulate run time results from all permutations
        t_adwin_mean += t_adwin
        t_awe_mean += t_awe
        t_dwm_mean += t_dwm
        t_lnse_mean += t_lnse
        t_srp_mean += t_srp

# Compute average accuracy and run time results over all permutations
if competitors:
    res_adwin_mean /= num_of_experiments
    res_awe_mean /= num_of_experiments
    res_dwm_mean /= num_of_experiments
    res_lnse_mean /= num_of_experiments
    res_srp_mean /= num_of_experiments

    t_adwin_mean /= num_of_experiments
    t_awe_mean /= num_of_experiments
    t_dwm_mean /= num_of_experiments
    t_lnse_mean /= num_of_experiments
    t_srp_mean /= num_of_experiments

    res_adwin_mean = res_adwin_mean.reshape(-1, 1)
    res_awe_mean = res_awe_mean.reshape(-1, 1)
    res_dwm_mean = res_dwm_mean.reshape(-1, 1)
    res_lnse_mean = res_lnse_mean.reshape(-1, 1)
    res_srp_mean.reshape(-1, 1)

    # Save accuracy and run time results
    if save:
        Competitors_Results = pd.concat(
            [steps, pd.DataFrame(res_adwin_mean), pd.DataFrame(res_awe_mean), pd.DataFrame(res_dwm_mean),
             pd.DataFrame(res_lnse_mean), pd.DataFrame(res_srp_mean)], axis=1)
        columns = ['Step', 'adwin_Acc', 'awe_Acc', 'dwm_Acc', 'lnse_Acc', 'srp_Acc']
        Competitors_Results.columns = columns
        time_all = [t_adwin_mean, t_awe_mean, t_dwm_mean, t_lnse_mean, t_srp_mean]
        Competitors_Time = pd.DataFrame(np.array(time_all).reshape(1, -1))
        columns = ['adwin_time', 'awe_time', 'dwm_time', 'lnse_time', 'srp_time']
        Competitors_Time.columns = columns
        Competitors_Results.to_csv(working_path + '/Results/Hyperplane/Hyperplane_Competitors_Acc.csv', index=False)
        Competitors_Time.to_csv(working_path + '/Results/Hyperplane/Hyperplane_Competitors_Time.csv', index=False)


# ------------------------------------------------- Plotting ------------------------------------------------------- #

if plot:

    cddrl_res = pd.read_csv(working_path + '/Results/Hyperplane/Hyperplane_Competitors_Acc.csv', header=None)

    plt.figure(figsize=(10, 7))
    plt.plot(range(1, time_steps+1), cddrl_res)
    plt.plot(range(1, time_steps+1), res_adwin_mean)
    plt.plot(range(1, time_steps+1), res_awe_mean)
    plt.plot(range(1, time_steps+1), res_dwm_mean)
    plt.plot(range(1, time_steps+1), res_lnse_mean)
    plt.plot(range(1, time_steps+1), res_srp_mean)
    plt.legend(['CDDRL', 'KNN-ADWIN', 'AWE', 'DWM', 'LNSE', 'SRP'], loc='lower right', fontsize=12)
    plt.title('Competitors Accuracy Over Time', fontsize=14)
    plt.xlabel('Time step', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.vlines(x=change_point, ymin=0, ymax=1, color='black', linestyles='solid')
    plt.text(change_point, 0.1, "Change point", rotation=360, verticalalignment='baseline', fontsize=12)
    plt.grid()


