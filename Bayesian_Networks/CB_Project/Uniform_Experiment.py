import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import time


# -------------------------------------------------- Functions ------------------------------------------------------- #

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


# ------------------------------------------------- Sample Data ------------------------------------------------------ #

working_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Relevent_for_yotam\Concept_drift\CB_Project'

seed = 28
random.seed(seed)
n_features = 15
obs = 60000
drift_pos = 30000
drift_width = 5000
n_cat = 4
n_drift_features = 5
start_virtual = 10000
end_virtual = 20000
noise_percentage = 0.05
num_of_experiments = 10
time_steps = 20
change_point = 10
samples_for_time_step = 3000
jump = samples_for_time_step
steps = np.arange(1, time_steps + 1, 1)
steps = pd.DataFrame(steps.reshape(-1, 1))

sample = 0
virtual = 0
competitors = 1
save = 1
plot = 0

if sample:
    for p in range(1, num_of_experiments + 1):

        # initiate dataframe with correct size
        cols = range(1, n_features + 2, 1)
        data = pd.DataFrame(columns=cols)

        # fill dataframe
        for i in range(1, obs + 1):

            print("Sampling observation %s in permutation %s" % (i, p))

            data.at[i, :n_features] = np.random.randint(1, n_cat + 1, n_features)   # generate random integers to the current row

            # before drift position classify as 1 if any of the first n_drift_features is above 2.5 and 0 otherwise
            if i <= drift_pos:
                if virtual:
                    if start_virtual < i <= end_virtual:
                        # apply virtual drift to the drifted features
                        data.loc[i:i, :n_drift_features] += 1
                if np.mean(np.array(data.loc[i:i, 1:n_drift_features])) > 2.5:
                    data.at[i, n_features + 1] = 1
                else:
                    data.at[i, n_features + 1] = 0

            # if we are at the drift period
            else:
                ft = 1 / (1 + np.exp(-4 * (i - drift_pos) / drift_width))
                if ft > random.uniform(0, 1):   # new concept
                    if np.mean(np.array(data.loc[i:i, n_features - n_drift_features + 1:n_features])) <= 2.6:
                        data.at[i, n_features + 1] = 1
                    else:
                        data.at[i, n_features + 1] = 0
                else:   # first concept
                    if np.mean(np.array(data.loc[i:i, 1:n_drift_features])) > 2.5:
                        data.at[i, n_features + 1] = 1
                    else:
                        data.at[i, n_features + 1] = 0

            # apply noise
            if random.uniform(0, 1) <= noise_percentage:
                if data.at[i, n_features + 1] == 0:
                    data.at[i, n_features + 1] = 1
                else:
                    data.at[i, n_features + 1] = 0

        # save dataframe
        data.to_csv(working_path + '/Data/Uniform/Uniform_data' + str(p) + '.csv', index=False)
        print("Finished sampling permutation: ", p)


# ------------------------------------------------- Experiment ------------------------------------------------------- #

res_adwin_mean, res_awe_mean, res_dwm_mean, res_lnse_mean, res_srp_mean = np.zeros(time_steps), np.zeros(time_steps), \
                                                                np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)
t_adwin_mean, t_awe_mean, t_dwm_mean, t_lnse_mean, t_srp_mean = 0, 0, 0, 0, 0

for r in tqdm(range(num_of_experiments)):

    # Run Competitors on generated data
    if competitors:
        df = pd.read_csv(working_path + '/Data/Uniform/Uniform_data' + str(r+1) + '.csv')
        df.columns = ['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15', 'class']
        X = df[['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6', 'var_7', 'var_8', 'var_9', 'var_10', 'var_11', 'var_12', 'var_13', 'var_14', 'var_15']]
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
        Competitors_Results.to_csv(working_path + '/Results/Uniform/Uniform_Competitors_Acc.csv', index=False)
        Competitors_Time.to_csv(working_path + '/Results/Uniform/Uniform_Competitors_Time.csv', index=False)


# ------------------------------------------------- Plotting ------------------------------------------------------- #

if plot:

    cddrl_res = pd.read_csv(working_path + '/Results/Uniform/Uniform_Competitors_Acc.csv', header=None)

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


