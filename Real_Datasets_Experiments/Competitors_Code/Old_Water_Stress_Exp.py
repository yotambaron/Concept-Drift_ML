from Concept_drift.CD_Experiments.CD_Utils import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd


# ---------------------------------------------- Multiclass Evaluation ----------------------------------------------- #

data_path = r'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Data'
results_path = r'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Results\Multiclass\Competitors'
df = pd.read_csv(data_path + '/WS_old_no_disc.csv')
y_true = np.array(df['class'])
df = df.iloc[:, 3:]
del df['class']

labels = ['A', 'B', 'C', 'D']
jump = 120
window_days = 2
time_steps = 17
steps = range(1, time_steps + 1)
classes = [0, 1, 2, 3]

# Evaluate Competitors
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.copy())

[adwin_preds, awe_preds, dwm_preds, lnse_preds, srp_preds, adwin_probs, awe_probs, dwm_probs, lnse_probs, srp_probs] = \
    predict_competitors(df_scaled, y_true - 1, jump, window_days, classes)

# Save all algorithms' predictions
pd.DataFrame(adwin_preds).to_csv(results_path + '/Predictions_KNN-ADWIN.csv', index=False)
pd.DataFrame(awe_preds).to_csv(results_path + '/Predictions_AWE.csv', index=False)
pd.DataFrame(dwm_preds).to_csv(results_path + '/Predictions_DWM.csv', index=False)
pd.DataFrame(lnse_preds).to_csv(results_path + '/Predictions_LNSE.csv', index=False)
pd.DataFrame(srp_preds).to_csv(results_path + '/Predictions_SRP.csv', index=False)

# Save all algorithms' probabilities
pd.DataFrame(adwin_probs).to_csv(results_path + '/Probabilities_KNN-ADWIN.csv', index=False)
pd.DataFrame(awe_probs).to_csv(results_path + '/Probabilities_AWE.csv', index=False)
pd.DataFrame(dwm_probs).to_csv(results_path + '/Probabilities_DWM.csv', index=False)
pd.DataFrame(lnse_probs).to_csv(results_path + '/Probabilities_LNSE.csv', index=False)
pd.DataFrame(srp_probs).to_csv(results_path + '/Probabilities_SRP.csv', index=False)

# Compute and save multiclass accuracies
Accuracies_adwin = pd.DataFrame(np.zeros([len(steps), 1]), columns=['Multiclass Accuracy'])
Accuracies_awe = pd.DataFrame(np.zeros([len(steps), 1]), columns=['Multiclass Accuracy'])
Accuracies_dwm = pd.DataFrame(np.zeros([len(steps), 1]), columns=['Multiclass Accuracy'])
Accuracies_lnse = pd.DataFrame(np.zeros([len(steps), 1]), columns=['Multiclass Accuracy'])
Accuracies_srpc = pd.DataFrame(np.zeros([len(steps), 1]), columns=['Multiclass Accuracy'])

for day in range(len(steps)):
    start = day * jump
    Accuracies_adwin.iloc[day, 0] = np.round(np.mean(y_true[start: start + jump] - 1 == adwin_preds[start: start + jump]), 2)
    Accuracies_awe.iloc[day, 0] = np.round(np.mean(y_true[start: start + jump] - 1 == awe_preds[start: start + jump]), 2)
    Accuracies_dwm.iloc[day, 0] = np.round(np.mean(y_true[start: start + jump] - 1 == dwm_preds[start: start + jump]), 2)
    Accuracies_lnse.iloc[day, 0] = np.round(np.mean(y_true[start: start + jump] - 1 == lnse_preds[start: start + jump]), 2)
    Accuracies_srpc.iloc[day, 0] = np.round(np.mean(y_true[start: start + jump] - 1 == srp_preds[start: start + jump]), 2)

Accuracies_adwin.to_csv(results_path + '/Accuracies_KNN-ADWIN.csv', index=False)
Accuracies_awe.to_csv(results_path + '/Accuracies_AWE.csv', index=False)
Accuracies_dwm.to_csv(results_path + '/Accuracies_DWM.csv', index=False)
Accuracies_lnse.to_csv(results_path + '/Accuracies_LNSE.csv', index=False)
Accuracies_srpc.to_csv(results_path + '/Accuracies_SRP.csv', index=False)

# Compute all algorithms' scores
Results_adwin = compute_multiclass_scores(adwin_preds + 1, adwin_probs, y_true, labels, jump)
Results_awe = compute_multiclass_scores(awe_preds + 1, awe_probs, y_true, labels, jump)
Results_dwm = compute_multiclass_scores(dwm_preds + 1, dwm_probs, y_true, labels, jump)
Results_lnse = compute_multiclass_scores(lnse_preds + 1, lnse_probs, y_true, labels, jump)
Results_srpc = compute_multiclass_scores(srp_preds + 1, srp_probs, y_true, labels, jump)

# Save all algorithms' results
Results_adwin.to_csv(results_path + '/Results_KNN-ADWIN.csv', index=False)
Results_awe.to_csv(results_path + '/Results_AWE.csv', index=False)
Results_dwm.to_csv(results_path + '/Results_DWM.csv', index=False)
Results_lnse.to_csv(results_path + '/Results_LNSE.csv', index=False)
Results_srpc.to_csv(results_path + '/Results_SRP.csv', index=False)

# Evaluate CDDRL
cddrl_path = r'D:\yotam\MATLAB\Stress_Experiments\Water_Stress_Old\Results\Multiclass\CDDRL\Ensemble_BNs'
cddrl_pred = pd.read_csv(cddrl_path + '/Predictions.csv')
cddrl_probs = pd.read_csv(cddrl_path + '/Scores.csv')
for i in range(len(cddrl_probs)):
    if sum(cddrl_probs.iloc[i, :]) != 1:
        max_inds = np.where(cddrl_probs.iloc[i, :] == np.amax(cddrl_probs.iloc[i, :]))
        max_ind = max_inds[0][0]
        max_prob = cddrl_probs.iloc[i, max_ind]
        cddrl_probs.iloc[i, max_ind] = 1 - sum(cddrl_probs.iloc[i, :]) + max_prob
cddrl_results = compute_multiclass_scores(cddrl_pred, cddrl_probs, y_true, labels, jump)
cddrl_results.to_csv(cddrl_path + '/All_Multiclass_Results.csv', index=False)



