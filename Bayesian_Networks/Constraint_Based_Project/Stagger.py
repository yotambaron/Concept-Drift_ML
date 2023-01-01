import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import uniform
from skmultiflow.lazy import KNNADWINClassifier
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier
from skmultiflow.meta import DynamicWeightedMajorityClassifier
from skmultiflow.meta import LearnPPNSEClassifier
from skmultiflow.meta import StreamingRandomPatchesClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from skmultiflow.data.stagger_generator import STAGGERGenerator
from sklearn.preprocessing import OneHotEncoder
import random
import time


# -------------------------------------------------- Functions ------------------------------------------------------- #

class CD_gradual:

    def __init__(self, data_path, data_size=20000, drift_position=10000, width=1000):
        self.data_size = data_size
        self.drift_position = drift_position
        self.width = width
        self.data_path = data_path

    def generate_2concepts(self, ind):
        stream1 = STAGGERGenerator(classification_function=0, balance_classes=True)
        stream2 = STAGGERGenerator(classification_function=1, balance_classes=True)
        stream1.prepare_for_use()
        stream2.prepare_for_use()
        X, y = stream1.next_sample(self.drift_position)
        for t in range(self.drift_position, self.data_size):
            ft = 1 / (1 + np.exp(-4 * (t - self.drift_position) / self.width))
            if ft > random.uniform(0, 1):
                xt, yt = stream2.next_sample()
            else:
                xt, yt = stream1.next_sample()
            X = np.concatenate([X, xt.reshape(1, -1)], axis=0)
            y = np.concatenate([y, pd.Series(yt)], axis=0)
        x_noise = np.array(uniform.rvs(0, 3, size=(int(30000), 2))).astype(int)
        X = np.concatenate([X, x_noise], axis=1)
        df = pd.DataFrame(np.concatenate([X, y.reshape(-1, 1)], axis=1), columns=['size', 'shape', 'colour', 'noise1', 'noise2', 'class'])
        df = df.astype(int) + 1
        df.to_csv(self.data_path+r'\Stagger_data' + str(ind) + '.csv', index=False)
        oht = OneHotEncoder(sparse=False)
        X = oht.fit_transform(X)
        return X, y


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


# --------------- Compute KPIs ---------------- #

def KPI_extraction(recovering_threshold, change_point, results):
    results = np.array(results)
    avg_accuarcy = np.mean(results)
    lowest_accuarcy = np.min(results[change_point:])
    high_preformance_inds = np.where(results >= recovering_threshold)[0]
    time_steps_to_recover = high_preformance_inds[high_preformance_inds > change_point][0] - change_point
    return avg_accuarcy, lowest_accuarcy, time_steps_to_recover


def save_competitors(accs_ADWIN, accs_AWE, accs_DWM, accs_LNSE, accs_SRPC, recovery_threshold, change_point, path_to_save):
    df = pd.DataFrame(columns=['Algorithm', 'Average accuarcy', 'Min accuacry', 'Recover time'])
    res_list = [accs_ADWIN, accs_AWE, accs_DWM, accs_LNSE, accs_SRPC]
    alg_names = ['KNN-Adwin', 'AWE', 'DWM', 'LNSE', 'SPRC']
    dictt = {i: val for i, val in enumerate(alg_names)}
    for ind, res in enumerate(res_list):
        avg_accuarcy, lowest_accuarcy, time_steps_to_recover = KPI_extraction(recovery_threshold, change_point, res)
        algorithm_name = dictt[ind]
        row_vals = [algorithm_name, avg_accuarcy, lowest_accuarcy, time_steps_to_recover]
        df = df.append({col: row_vals[i] for i, col in enumerate(df.columns)}, ignore_index=True)
    df.to_csv(path_to_save + '\\Stagger_Competitors_Results.csv', index=False)


# ------------------------------------------------- Experiment ------------------------------------------------------- #

seed = 28
time_steps = 30
num_of_experiments = 10
jump = 1000
d_position = 15
sample = 0
competitors = 1
save = 1
recovery_threshold = 0.9
steps = np.arange(1, time_steps + 1, 1)
steps = pd.DataFrame(steps.reshape(-1, 1))

# --------------- Gradual concept drift ---------------- #
widths = [500, 5000]
res_adwin_mean, res_awe_mean, res_dwm_mean, res_lnse_mean, res_srp_mean = np.zeros(time_steps), np.zeros(time_steps), \
                                                                np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps)
t_adwin_mean, t_awe_mean, t_dwm_mean, t_lnse_mean, t_srp_mean = 0, 0, 0, 0, 0
stagger_path = r'C:\Users\yotambar\Desktop\yotam\Concept_drift\CB_Project'

for width in tqdm(widths):

    for i in tqdm(range(num_of_experiments)):
        if sample:
            cd_gradual = CD_gradual(data_path=stagger_path + '/Data/STAGGER/' + str(width), width=width, drift_position=d_position * jump, data_size=time_steps * jump)
            X, y = cd_gradual.generate_2concepts(i + 1)

        if competitors:
            df = pd.read_csv(stagger_path + '/Data/STAGGER/' + str(width) + '/Stagger_data' + str(i + 1) + '.csv')
            X = df[['size', 'shape', 'colour', 'noise1', 'noise2']]
            y = df['class'] - 1
            X.reset_index(inplace=True, drop=True)
            y.reset_index(inplace=True, drop=True)
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
                [steps, pd.DataFrame(res_adwin_mean), pd.DataFrame(res_awe_mean), pd.DataFrame(res_dwm_mean), pd.DataFrame(res_lnse_mean), pd.DataFrame(res_srp_mean)], axis=1)
            columns = ['Step', 'adwin_Acc', 'awe_Acc', 'dwm_Acc', 'lnse_Acc', 'srp_Acc']
            Competitors_Results.columns = columns
            time_all = [t_adwin_mean, t_awe_mean, t_dwm_mean, t_lnse_mean, t_srp_mean]
            Competitors_Time = pd.DataFrame(np.array(time_all).reshape(1, -1))
            columns = ['adwin_time', 'awe_time', 'dwm_time', 'lnse_time', 'srp_time']
            Competitors_Time.columns = columns
            save_path = stagger_path + '/Results/STAGGER/' + str(width)
            Competitors_Results.to_csv(save_path + '/Stagger_Competitors_Acc.csv', index=False)
            Competitors_Time.to_csv(save_path + '/Stagger_Competitors_Time.csv', index=False)
            results = Competitors_Results.copy()
            save_competitors(results['adwin_Acc'].data, results['adwin_Acc'].data, results['adwin_Acc'].data,
                             results['adwin_Acc'].data, results['adwin_Acc'].data, recovery_threshold, d_position,
                             save_path)


# # --------------- Sudden concept drift ---------------- #
# res_adwin, res_awe, res_dwm, res_lnse, res_srp = np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps), \
#                                                  np.zeros(time_steps), np.zeros(time_steps)
# res_adwin1, res_awe1, res_dwm1, res_lnse1, res_srp1 = np.zeros(time_steps), np.zeros(time_steps), np.zeros(time_steps), \
#                                                  np.zeros(time_steps), np.zeros(time_steps)
#
# gradual_path = r'C:\Users\yotam\Desktop\yotam\CD_Project\DBs\STAGGER\gradual_drift\width=5000'
# sudden_path = r'C:\Users\yotam\Desktop\yotam\CD_Project\DBs\STAGGER\gradual_drift\width=50'
#
# for i in tqdm(range(num_of_experiments)):
#     cd_sudden = CD_sudden(data_path=sudden_path + '\\' + str(i+1))
#     X, y = cd_sudden.generate_2concepts()
#     # df = pd.read_csv(gradual_path + '\\' + str(i + 1) +  '\\' + 'df_gradual.csv')
#     X = df[['size', 'shape', 'colour']]
#     y = df['class']-1
#     X.reset_index(inplace=True, drop=True)
#     y.reset_index(inplace=True, drop=True)
#     eval_competitors = Evaluate_competitors(is_batchs=True)
#     res_adwin_t, res_awe_t, res_dwm_t, res_lnse_t, res_srp_t = predict_competitors(X.values, y.values, jump, seed)
#     #res_adwin_t, res_awe_t, res_dwm_t, res_lnse_t, res_srp_t = eval_competitors.fit_predict_evaluate(X, y)
#     res_adwin += res_adwin_t
#     res_awe += res_awe_t
#     res_dwm += res_dwm_t
#     res_lnse += res_lnse_t
#     #res_srp += res_srp
#     #eval_competitors = Evaluate_competitors(is_batchs=False)
#     #res_adwin_t1, res_awe_t1, res_dwm_t1, res_lnse_t1, res_srp_t1 = eval_competitors.fit_predict_evaluate(X, y)
#     #res_adwin1 += res_adwin_t1
#     #res_awe1 += res_awe_t1
#     #res_dwm1 += res_dwm_t1
#     #res_lnse1 += res_lnse_t1
#     #res_srp1 += res_srp_t1
#
# res_adwin /= num_of_experiments
# res_awe /= num_of_experiments
# res_dwm /= num_of_experiments
# res_lnse /= num_of_experiments
# #res_srp /= num_of_experiments
# #res_cddrl = pd.read_csv(sudden_path+'\\cddrl_stagger.csv', header=None)
# pd.DataFrame(res_adwin).to_csv(r'C:\Users\yotam\Desktop\yotam\CD_Project\Results\Stagger\Gradual_drift_5000\Competitors\Accs_adwin.csv')
# pd.DataFrame(res_awe).to_csv(r'C:\Users\yotam\Desktop\yotam\CD_Project\Results\Stagger\Gradual_drift_5000\Competitors\Accs_awe.csv')
# pd.DataFrame(res_dwm).to_csv(r'C:\Users\yotam\Desktop\yotam\CD_Project\Results\Stagger\Gradual_drift_5000\Competitors\Accs_dwm.csv')
# pd.DataFrame(res_lnse).to_csv(r'C:\Users\yotam\Desktop\yotam\CD_Project\Results\Stagger\Gradual_drift_5000\Competitors\Accs_lnse.csv')
#
# plot = Plot(path_to_save=sudden_path + '\\batch', title='Performance comparison in sudden concept drift', ylim=(0.4, 1.05))
# plot.plot_results(res_cddrl, res_adwin, res_awe, res_dwm, res_lnse, res_srp)
# kpi = KPI(path_to_save=gradual_path + '\\batch')
# kpi.save(res_cddrl, res_adwin, res_awe, res_dwm, res_lnse, res_srp)
#
# res_adwin1 /= num_of_experiments
# res_awe1 /= num_of_experiments
# res_dwm1 /= num_of_experiments
# res_lnse1 /= num_of_experiments
# res_srp1 /= num_of_experiments
# res_cddrl = pd.read_csv(sudden_path+'\\cddrl_stagger.csv', header=None)
#
# plot = Plot(path_to_save=sudden_path + '\\oneBYone', title='Performance comparison in sudden concept drift', ylim=(0.4, 1.05))
# plot.plot_results(res_cddrl, res_adwin1, res_awe1, res_dwm1, res_lnse1, res_srp1)
# kpi = KPI(path_to_save=sudden_path + '\\oneBYone')
# kpi.save(res_cddrl, res_adwin1, res_awe1, res_dwm1, res_lnse1, res_srp1)
#
#
#
#
#
#     # --------------- Recurring concept drift ---------------- #
#     res_adwin_r, res_awe_r, res_dwm_r, res_lnse_r, res_srp_r = np.zeros(time_steps), np.zeros(time_steps), np.zeros(
#         time_steps), \
#                                                                np.zeros(time_steps), np.zeros(time_steps)
#     res_adwin_r1, res_awe_r1, res_dwm_r1, res_lnse_r1, res_srp_r1 = np.zeros(time_steps), np.zeros(
#         time_steps), np.zeros(time_steps), \
#                                                                     np.zeros(time_steps), np.zeros(time_steps)
#
#     recurring_path = r'C:\Users\97254\Desktop\CD\STAGGER\recurring_drift'
#     for i in tqdm(range(num_of_experiments)):
#         cd_recurring = CD_recurring(data_path=recurring_path + '\\' + str(i + 1))
#         X, y = cd_recurring.generate_2concepts()
#         eval_competitors = Evaluate_competitors(is_batchs=True)
#         res_adwin_t, res_awe_t, res_dwm_t, res_lnse_t, res_srp_t = eval_competitors.fit_predict_evaluate(X, y)
#         res_adwin_r += res_adwin_t
#         res_awe_r += res_awe_t
#         res_dwm_r += res_dwm_t
#         res_lnse_r += res_lnse_t
#         res_srp_r += res_srp_t
#         eval_competitors = Evaluate_competitors(is_batchs=False)
#         res_adwin_t1, res_awe_t1, res_dwm_t1, res_lnse_t1, res_srp_t1 = eval_competitors.fit_predict_evaluate(X, y)
#         res_adwin_r1 += res_adwin_t1
#         res_awe_r1 += res_awe_t1
#         res_dwm_r1 += res_dwm_t1
#         res_lnse_r1 += res_lnse_t1
#         res_srp_r1 += res_srp_t1
#
#     res_adwin_r /= num_of_experiments
#     res_awe_r /= num_of_experiments
#     res_dwm_r /= num_of_experiments
#     res_lnse_r /= num_of_experiments
#     res_srp_r /= num_of_experiments
#     res_cddrl_r = pd.read_csv(recurring_path + '\\cddrl_stagger.csv', header=None)
#
#     plot = Plot(path_to_save=recurring_path + '\\batch', title='Performance comparison in recurring concept drift',
#                 ylim=(0.4, 1.05), is_recurring=True, change_point=[50, 100, 150], xlim=(0, 200))
#     plot.plot_results(res_cddrl_r, res_adwin_r, res_awe_r, res_dwm_r, res_lnse_r, res_srp_r)
#     kpi = KPI(path_to_save=recurring_path + '\\batch')
#     kpi.save(res_cddrl_r, res_adwin_r, res_awe_r, res_dwm_r, res_lnse_r, res_srp_r)
#
#     res_adwin_r1 = res_adwin_r1 / num_of_experiments
#     res_awe_r1 = res_awe_r1 / num_of_experiments
#     res_dwm_r1 = res_dwm_r1 / num_of_experiments
#     res_lnse_r1 = res_lnse_r1 / num_of_experiments
#     res_srp_r1 = res_srp_r1 / num_of_experiments
#     res_cddrl_r1 = pd.read_csv(recurring_path + '\\cddrl_stagger.csv', header=None)
#
#     plot = Plot(path_to_save=recurring_path + '\\oneBYone', title='Performance comparison in recurring concept drift',
#                 ylim=(0.4, 1.05), is_recurring=True, change_point=[50, 100, 150], xlim=(0, 200))
#     plot.plot_results(res_cddrl_r1, res_adwin_r1, res_awe_r1, res_dwm_r1, res_lnse_r1, res_srp_r1)
#     kpi = KPI(path_to_save=recurring_path + '\\oneBYone')
#     kpi.save(res_cddrl_r1, res_adwin_r1, res_awe_r1, res_dwm_r1, res_lnse_r1, res_srp_r1)
#
#
#
# # --------- Compute KPIs for the competitors ---------- #
#
#     path = r'C:\Users\yotam\Desktop\yotam\CD_Project\Results\Stagger\Gradual_drift_5000\CDDRL'
#     comp_path = r'C:\Users\yotam\Desktop\yotam\CD_Project\Results\Stagger\Gradual_drift_5000\Competitors'
#
#     accs_ADWIN = pd.read_csv(comp_path + '/Accs_adwin.csv')
#     accs_AWE = pd.read_csv(comp_path + '/Accs_awe.csv')
#     accs_DWM = pd.read_csv(comp_path + '/Accs_dwm.csv')
#     accs_LNSE = pd.read_csv(comp_path + '/Accs_lnse.csv')
#
#     save_competitors(accs_ADWIN, accs_AWE, accs_DWM, accs_LNSE, accs_LNSE, 0.9, 100, comp_path)
#
# # --------- Compute KPIs for the CDDRL versions ---------- #
#
#     def save_CDDRLs(CDDRL_dynamic, CDDRL_dynamicUCL, CDDRL_tabu5, CDDRL_tabu10,
#                     CDDRL_robust, CDDRL_prior5, CDDRL_prior9, CDDRL_prior_markov, CDDRL_prior_dag,
#                     recovery_threshold, change_point, path_to_save):
#     # def save_CDDRLs(CDDRL_dynamic, CDDRL_dynamicUCL, CDDRL_tabu5, CDDRL_robust, CDDRL_prior9, CDDRL_prior_dag,
#     #                     recovery_threshold, change_point, path):
#         df = pd.DataFrame(columns=['Algorithm', 'Average accuarcy', 'STD accuarcy', 'Min accuacry', 'Recover time'])
#         res_list = [CDDRL_dynamic['Accuracy'], CDDRL_dynamicUCL['Accuracy'], CDDRL_tabu5['Accuracy'],
#                     CDDRL_tabu10['Accuracy'], CDDRL_robust['Accuracy'], CDDRL_prior5['Accuracy'],
#                     CDDRL_prior9['Accuracy'], CDDRL_prior_markov['Accuracy'], CDDRL_prior_dag['Accuracy']]
#         alg_names = ['CDDRL_original', 'CDDRL_dynamicUCL', 'CDDRL_tabu5', 'CDDRL_tabu10', 'CDDRL_robust', 'CDDRL_prior0.5',
#                      'CDDRL_prior0.9', 'CDDRL_prior_markov', 'CDDRL_prior_dag']
#         dictt = {i: val for i, val in enumerate(alg_names)}
#         for ind, res in enumerate(res_list):
#             avg_accuarcy, std_accuarcy, lowest_accuarcy, time_steps_to_recover = KPI_extraction(recovery_threshold, change_point, res)
#             algorithm_name = dictt[ind]
#             row_vals = [algorithm_name, avg_accuarcy, std_accuarcy, lowest_accuarcy, time_steps_to_recover]
#             df = df.append({col: row_vals[i] for i, col in enumerate(df.columns)}, ignore_index=True)
#         df.to_csv(path + '\\CDDRLs_Results_Summary.csv', index=False)
#
#
# # CDDRL_static = pd.read_csv(path + '\\cddrl_static_stagger.csv')
# CDDRL_dynamic = pd.read_csv(path + '\\stagger_gradual_5000_Original.csv')
# CDDRL_dynamicUCL = pd.read_csv(path + '\\stagger_gradual_5000_DynamicUCL.csv')
# CDDRL_tabu5 = pd.read_csv(path + '\\stagger_gradual_5000_Tabu5.csv')
# CDDRL_tabu10 = pd.read_csv(path + '\\stagger_gradual_5000_Tabu10.csv')
# CDDRL_robust = pd.read_csv(path + '\\stagger_gradual_5000_Robust.csv')
# CDDRL_prior5 = pd.read_csv(path + '\\stagger_gradual_5000_Prior0.5.csv')
# CDDRL_prior9 = pd.read_csv(path + '\\stagger_gradual_5000_Prior0.9.csv')
# CDDRL_prior_markov = pd.read_csv(path + '\\stagger_gradual_5000_Prior_Markov.csv')
# CDDRL_prior_dag = pd.read_csv(path + '\\stagger_gradual_5000_Prior_dag.csv')
#
# save_CDDRLs(CDDRL_dynamic, CDDRL_dynamicUCL, CDDRL_tabu5, CDDRL_tabu10, CDDRL_robust,
#              CDDRL_prior5, CDDRL_prior9, CDDRL_prior_markov, CDDRL_prior_dag, 0.9, 100, path)
#
# #save_CDDRLs(CDDRL_dynamic, CDDRL_dynamicUCL, CDDRL_tabu5, CDDRL_robust, CDDRL_prior9, CDDRL_prior_dag, 0.95, 100, path)

