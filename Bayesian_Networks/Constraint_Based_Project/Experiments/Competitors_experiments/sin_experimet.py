import numpy as np
from scipy.stats import binom, boltzmann, betabinom, uniform
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from tqdm import tqdm
from supervised_cd_utils import Evaluate_competitors
import matplotlib.pyplot as plt
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import random


# -------------------------------------------------- Functions ------------------------------------------------------- #

def predict_competitors(X, y, jump, seed):
    random.seed(seed)
    ind = jump
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
        for j in range(X_day_i_minus.shape[0]):
            dwm.fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
            knn_adwin.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
            awe.partial_fit(X_day_i_minus[j].reshape(1, -1), y_day_i_minus[j].reshape(1))
            srpc.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1))
            lnse.partial_fit(X_day_i_minus[j, :].reshape(1, -1), np.array(y_day_i_minus[j]).reshape(1), classes=[0, 1])

        for k in range(X_day_i.shape[0]):
            y_pred_dwm[ind] = dwm.predict(X_day_i[k].reshape(1, -1))
            y_pred_adwin[ind] = knn_adwin.predict(X_day_i[k].reshape(1, -1))
            y_pred_awe[ind] = awe.predict(X_day_i[k].reshape(1, -1))
            y_pred_lnse[ind] = lnse.predict(X_day_i[k].reshape(1, -1))
            y_pred_srpc[ind] = srpc.predict(X_day_i[k].reshape(1, -1))
            ind += 1

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

    results_adwin = measure_over_time(y_pred=y_pred_adwin, y_test=y, jumping=jump, metric='F1')
    results_awe = measure_over_time(y_pred=y_pred_awe, y_test=y, jumping=jump, metric='F1')
    results_dwm = measure_over_time(y_pred=y_pred_dwm, y_test=y, jumping=jump, metric='F1')
    results_lnse = measure_over_time(y_pred=y_pred_lnse, y_test=y, jumping=jump, metric='F1')
    results_srp = measure_over_time(y_pred=y_pred_srpc, y_test=y, jumping=jump, metric='F1')

    return results_adwin, results_awe, results_dwm, results_lnse, results_srp


# ------------------------------------------------- Experiment ------------------------------------------------------- #

num_of_experiments = 10
working_path = r'C:\Users\yotam\Desktop\yotam\python\Phenomics_PreProcessing_Scripts\Relevent_for_yotam\Concept_drift\CB_Project'
time_steps = 30
change_point = 10
samples_for_time_step = 1000
seed = 28
n_noise_features = 5
steps = np.arange(1, time_steps + 1, 1)
steps = pd.DataFrame(steps.reshape(-1, 1))
sample = 0
save = 1
competitors = 1
n, p = 4, 0.2  # binomial distribution params
a, b = 2, 1  # beta distribution params
pts = [p if t < change_point else p + 0.01*t for t in range(time_steps)]
pts = [p if p < 1 else 1 for p in pts]
mus = [1 if t < change_point else 1 / (1 + 0.1*(t - change_point + 1)) for t in range(time_steps)]
As = [2 if t < change_point else 2 + 0.1*(t - change_point + 1) for t in range(time_steps)]

res_adwin_mean, res_awe_mean, res_dwm_mean, res_lnse_mean, res_srp_mean = np.zeros(time_steps), np.zeros(time_steps), \
                                                                np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)
for r in tqdm(range(num_of_experiments)):
    if sample:
        y = pd.Series(np.random.randint(low=0, high=2, size=int(samples_for_time_step * change_point)))
        x1 = np.array(binom.rvs(n=n, p=p, size=int(change_point * samples_for_time_step)))  # say color
        x2 = np.array(boltzmann.rvs(lambda_=1, N=3, size=int(change_point * samples_for_time_step)))  # say size
        x3 = np.array(betabinom.rvs(a=a, b=b, n=n, size=int(change_point * samples_for_time_step)))  # say curvature
        x_noise = np.array(
            uniform.rvs(0, 3, size=(int(change_point * samples_for_time_step), n_noise_features))).astype(int)
        X = np.concatenate([x1.reshape(-1, 1), x2.reshape(-1, 1), x3.reshape(-1, 1), x_noise], axis=1)
        X = pd.DataFrame(X, columns=['color', 'area', 'curvature', 'noise1', 'noise2', 'noise3', 'noise4', 'noise5'])

        for t in tqdm(range(change_point, time_steps)):
            x1_temp = np.array(binom.rvs(n=n, p=pts[t], size=samples_for_time_step))
            x2_temp = np.array(boltzmann.rvs(lambda_=mus[t], N=3, size=samples_for_time_step))
            x3_temp = np.array(betabinom.rvs(a=As[t], b=b, n=n, size=samples_for_time_step))
            x_noise_temp = np.array(uniform.rvs(0, 3, size=(samples_for_time_step, n_noise_features))).astype(int)
            X_temp = np.concatenate([x1_temp.reshape(-1, 1), x2_temp.reshape(-1, 1),  x3_temp.reshape(-1, 1), x_noise_temp], axis=1)
            X_temp = pd.DataFrame(X_temp, columns=['color', 'area', 'curvature', 'noise1', 'noise2', 'noise3', 'noise4', 'noise5'])
            lowest_size = np.min(X_temp.area)
            quantile25_curvature = np.quantile(X_temp.curvature, q=0.25)
            mean_color = np.round(np.mean(X_temp.color))
            # rules_for_stress
            y_temp = np.zeros(samples_for_time_step)  # initialize y with zeros
            for s in range(samples_for_time_step):
                if (X_temp.area[s] == lowest_size) and (X_temp.curvature[s] <= quantile25_curvature):
                    y_temp[s] = 1
                elif (X_temp.area[s] == lowest_size + 1) and (X_temp.color[s] > mean_color + 1 or X_temp.color[s] < mean_color - 1):
                    y_temp[s] = 1
            X = pd.concat([X.copy(), X_temp], axis=0)
            y = pd.concat([y.copy(), pd.Series(y_temp)], axis=0)

        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        data = pd.concat([X, y], axis=1)
        data += 1
        if save:
            data.columns = ['color', 'area', 'curvature', 'noise1', 'noise2', 'noise3', 'noise4', 'noise5', 'class']
            data.to_csv(working_path + '/Data/Sin/Sin_data' + str(r+1) + '.csv', index=False)

    # Run Competitors on generated data
    if competitors:
        df = pd.read_csv(working_path + '/Data/Sin/Sin_data' + str(r+1) + '.csv')
        X = df[['color', 'area', 'curvature', 'noise1', 'noise2', 'noise3', 'noise4', 'noise5']]
        y = df['class']-1
        X.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        # eval_competitors = Evaluate_competitors(is_batchs=True, metric='F1', jump=samples_for_time_step)
        # res_adwin, res_awe, res_dwm, res_lnse, res_srp = eval_competitors.fit_predict_evaluate(X.values, y.values)
        res_adwin, res_awe, res_dwm, res_lnse, res_srp = predict_competitors(X.values, y.values, samples_for_time_step, seed)
        res_adwin_mean += res_adwin
        res_awe_mean += res_awe
        res_dwm_mean += res_dwm
        res_lnse_mean += res_lnse
        res_srp_mean += res_srp

if competitors:
    res_adwin_mean /= num_of_experiments
    res_awe_mean /= num_of_experiments
    res_dwm_mean /= num_of_experiments
    res_lnse_mean /= num_of_experiments
    res_srp_mean /= num_of_experiments

    res_adwin_mean = res_adwin_mean.reshape(-1, 1)
    res_awe_mean = res_awe_mean.reshape(-1, 1)
    res_dwm_mean = res_dwm_mean.reshape(-1, 1)
    res_lnse_mean = res_lnse_mean.reshape(-1, 1)
    res_srp_mean.reshape(-1, 1)

    Competitors_Results = pd.concat([steps, pd.DataFrame(res_adwin_mean), pd.DataFrame(res_awe_mean), pd.DataFrame(res_dwm_mean), pd.DataFrame(res_lnse_mean), pd.DataFrame(res_srp_mean)], axis=1)
    columns = ['Step', 'adwin_F1', 'awe_F1', 'dwm_F1', 'lnse_F1', 'srp_F1']
    Competitors_Results.columns = columns
    Competitors_Results.to_csv(working_path + '/Results/Sin/Sin_Competitors_F1.csv', index=False)


# ------------------------------------------------- Plotting ------------------------------------------------------- #
cddrl_res = pd.read_csv(working_path + '\\cddrl_sin.csv', header=None)

plt.figure(figsize=(10, 7))
plt.plot(range(1, time_steps+1), cddrl_res)
plt.plot(range(1, time_steps+1), res_adwin_mean)
plt.plot(range(1, time_steps+1), res_awe_mean)
plt.plot(range(1, time_steps+1), res_dwm_mean)
plt.plot(range(1, time_steps+1), res_lnse_mean)
plt.plot(range(1, time_steps+1), res_srp_mean)
plt.legend(['CDDRL', 'KNN-ADWIN', 'AWE', 'DWM', 'LNSE', 'SRP'], loc='lower right', fontsize=12)
# plt.title('F1 over time', fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('F1', fontsize=14)
plt.vlines(x=change_point, ymin=0, ymax=1, color='black', linestyles='solid')
plt.text(change_point, 0.1, "Change point", rotation=360, verticalalignment='baseline', fontsize=12)
plt.grid()

res_df = pd.DataFrame(columns=['Time step', 'CDDRL', 'KNN-Adwin', 'AWE', 'DWM', 'LNSE', 'SRP'])
res_df['Time step'] = time_steps
res_df['CDDRL'] = cddrl_res[0]
res_df['KNN-Adwin'] = res_adwin_mean
res_df['AWE'] = res_awe_mean
res_df['DWM'] = res_dwm_mean
res_df['LNSE'] = res_lnse_mean
res_df['SRP'] = res_srp_mean
res_df.to_csv(r'C:\Users\97254\Desktop\CD\Sin\sin_res.csv', index=False)

res_df = pd.read_csv(working_path + r'\sin_res.csv')
time_steps = 30
res_adwin_mean = res_df['KNN-Adwin']
res_awe_mean = res_df['AWE']
res_dwm_mean = res_df['DWM']
res_lnse_mean = res_df['LNSE']
res_srp_mean = res_df['SRP']


# --------- Plot Results --------- #

time_steps = 30
change_point = 10
F1_results = pd.read_csv(r'C:\Users\yotam\Desktop\yotam\CD_Project\Results\Sin\F1_All_0.5.csv')

ADWIN_F1 = F1_results['KNN-ADWIN']
AWE_F1 = F1_results['AWE']
DWM_F1 = F1_results['DWM']
LNSE_F1 = F1_results['LNSE']

original_F1 = F1_results['original']
robust_F1 = F1_results['robust']
dynamic_UCL_F1 = F1_results['dynamic ucl']
tabu_F1 = F1_results['tabu 5']
prior_auto_F1 = F1_results['prior dag']
prior5_F1 = F1_results['prior 0.5']
prior9_F1 = F1_results['prior 0.9']

plt.figure(figsize=(10, 7))
plt.plot(range(1, time_steps+1), original_F1)
plt.plot(range(1, time_steps+1), dynamic_UCL_F1, ls=':')
plt.plot(range(1, time_steps+1), tabu_F1, ls=':')
plt.plot(range(1, time_steps+1), robust_F1, ls=':')
plt.plot(range(1, time_steps+1), prior_auto_F1, ls=':')
plt.plot(range(1, time_steps+1), prior5_F1, ls=':')
plt.plot(range(1, time_steps+1), prior9_F1, ls=':')
plt.legend(['Original', 'Dynamic UCL', 'Tabu', 'Robust', 'Prior auto', 'prior 0.5', 'prior 0.9'], loc='lower right', fontsize=12)
plt.title('F1 CDDRL Versions', fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('F1', fontsize=14)
plt.vlines(x=change_point, ymin=0, ymax=1, color='black', linestyles='solid')
plt.vlines(x=change_point+1, ymin=0, ymax=1, color='black', linestyles=':')
plt.text(change_point-1.5, 0.95, "Change point", rotation=360, verticalalignment='baseline', fontsize=10)
plt.grid()


plt.figure(figsize=(10, 7))
plt.plot(range(1, time_steps+1), prior_auto_F1)
plt.plot(range(1, time_steps+1), ADWIN_F1, ls=':')
plt.plot(range(1, time_steps+1), AWE_F1, ls=':')
plt.plot(range(1, time_steps+1), DWM_F1, ls=':')
plt.plot(range(1, time_steps+1), LNSE_F1, ls=':')
plt.legend(['CDDRL Prior Auto', 'KNN-ADWIN', 'AWE', 'DWM', 'LNSE'], loc='lower right', fontsize=12)
plt.title('F1 CDDRL vs. Competitors', fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('F1', fontsize=14)
plt.vlines(x=change_point, ymin=0, ymax=1, color='black', linestyles='solid')
plt.vlines(x=change_point+1, ymin=0, ymax=1, color='black', linestyles=':')
plt.text(change_point-1.5, 0.95, "Change point", rotation=360, verticalalignment='baseline', fontsize=10)
plt.grid()

