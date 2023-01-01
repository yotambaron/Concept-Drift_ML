from Concept_drift.CD_Experiments.CD_Utils import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\NYC_Subway_Traffic'
nyc_traffic_original = pd.read_csv(path + '/Traffic_First10_Stations.csv')
columns_original = nyc_traffic_original.columns
nyc_traffic = nyc_traffic_original.dropna(axis=1, inplace=False)
columns = nyc_traffic.columns
nyc_traffic = pd.DataFrame(nyc_traffic, columns=columns)

nyc_traffic['Unique ID'] = nyc_traffic['Unique ID'] + 1
del nyc_traffic['Datetime']

columns = nyc_traffic.columns
correlations = nyc_traffic.corr()
correlations.to_csv(path + '/Correlations.csv')
entries_cors = correlations['Entries']
exits_cors = correlations['Exits']
cor_threshold = 0.45

le = LabelEncoder()
length = len(nyc_traffic)
eq_freq_flag = 0
zero_flag = 0
bins = 3

keep_cols = []

for ind, col in enumerate(columns):

    if 3 <= ind <= 10:
        nyc_traffic.loc[:, col] = le.fit_transform(nyc_traffic.loc[:, col]) + 1

    if ind > 10:
        if (col == 'Entries') | (col == 'Exits'):
            bins = 3
            eq_freq_cuts = np.interp(np.linspace(0, length, bins + 1), np.arange(length), np.sort(nyc_traffic.loc[:, col]))
            eq_freq_cuts = eq_freq_cuts[1:len(eq_freq_cuts) - 1]
            nyc_traffic.loc[:, col] = bins_disc(nyc_traffic.loc[:, col], eq_freq_cuts, zero_flag)
        if (cor_threshold * (-1) > entries_cors[col]) | (entries_cors[col] > cor_threshold) | (cor_threshold * (-1) > exits_cors[col]) | (exits_cors[col] > cor_threshold):
            keep_cols.append(col)
        # if eq_freq_flag:
        #     eq_freq_cuts = np.interp(np.linspace(0, length, bins + 1), np.arange(length), np.sort(nyc_traffic.loc[:, col]))
        #     eq_freq_cuts = eq_freq_cuts[1:len(eq_freq_cuts) - 1]
        #     nyc_traffic.loc[:, col] = bins_disc(nyc_traffic.loc[:, col], eq_freq_cuts, zero_flag)

    else:
        keep_cols.append(col)

nyc_traffic = nyc_traffic[keep_cols]

nyc_traffic.to_csv(path + '/NYC_Traffic_18Vars_No_Discretization_Exists.csv', index=False)


# ---------------------------------------------- Multiclass Evaluation ----------------------------------------------- #

data_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\NYC_Subway_Traffic'
results_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\NYC_Subway_Traffic'
df = pd.read_csv(data_path + '/NYC_Traffic_No_Discretization_18Vars_Exits.csv')

target = 'Exits'
y_true = np.array(df[target])
del df[target]

# y_pred = np.array(pd.read_csv(results_path + '/Predictions.csv', header=None))
# y_probs = np.array(pd.read_csv(results_path + '/Scores.csv', header=None))

labels = ['low', 'mid', 'high']

jump = 1500
window_days = 2
steps = np.arange(1, int(len(y_true)/jump + 1), 1)     # 70
classes = [0, 1, 2]

# Evaluate CDDRL
# cddrl_results = compute_multiclass_scores(y_pred, y_probs, y_true, labels, jump)
# cddrl_results.to_csv(results_path + '/CDDRL/All_Results.csv', index=False)

# Evaluate Competitors
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.copy())

[adwin_preds, awe_preds, dwm_preds, lnse_preds, srp_preds, adwin_probs, awe_probs, dwm_probs, lnse_probs, srp_probs] = predict_competitors(df_scaled, y_true - 1, jump, window_days, classes)

# Save all algorithms' predictions
pd.DataFrame(adwin_preds).to_csv(results_path + '/' + target + '/Competitors/Predictions_KNN-ADWIN.csv', index=False)
pd.DataFrame(awe_preds).to_csv(results_path + '/' + target + '/Competitors/Predictions_AWE.csv', index=False)
pd.DataFrame(dwm_preds).to_csv(results_path + '/' + target + '/Competitors/Predictions_DWM.csv', index=False)
pd.DataFrame(lnse_preds).to_csv(results_path + '/' + target + '/Competitors/Predictions_LNSE.csv', index=False)
pd.DataFrame(srp_preds).to_csv(results_path + '/' + target + '/Competitors/Predictions_SRP.csv', index=False)

# Save all algorithms' probabilities
pd.DataFrame(adwin_probs).to_csv(results_path + '/' + target + '/Competitors/Probabilities_KNN-ADWIN.csv', index=False)
pd.DataFrame(awe_probs).to_csv(results_path + '/' + target + '/Competitors/Probabilities_AWE.csv', index=False)
pd.DataFrame(dwm_probs).to_csv(results_path + '/' + target + '/Competitors/Probabilities_DWM.csv', index=False)
pd.DataFrame(lnse_probs).to_csv(results_path + '/' + target + '/Competitors/Probabilities_LNSE.csv', index=False)
pd.DataFrame(srp_probs).to_csv(results_path + '/' + target + '/Competitors/Probabilities_SRP.csv', index=False)

# Compute all algorithms' scores
Results_adwin = compute_multiclass_scores(adwin_preds + 1, adwin_probs, y_true, labels, jump)
Results_awe = compute_multiclass_scores(awe_preds + 1, awe_probs, y_true, labels, jump)
Results_dwm = compute_multiclass_scores(dwm_preds + 1, dwm_probs, y_true, labels, jump)
Results_lnse = compute_multiclass_scores(lnse_preds + 1, lnse_probs, y_true, labels, jump)
Results_srpc = compute_multiclass_scores(srp_preds + 1, srp_probs, y_true, labels, jump)

# Save all algorithms' results
Results_adwin.to_csv(results_path + '/' + target + '/Competitors/Results_KNN-ADWIN.csv', index=False)
Results_awe.to_csv(results_path + '/' + target + '/Competitors/Results_AWE.csv', index=False)
Results_dwm.to_csv(results_path + '/' + target + '/Competitors/Results_DWM.csv', index=False)
Results_lnse.to_csv(results_path + '/' + target + '/Competitors/Results_LNSE.csv', index=False)
Results_srpc.to_csv(results_path + '/' + target + '/Competitors/Results_SRP.csv', index=False)



