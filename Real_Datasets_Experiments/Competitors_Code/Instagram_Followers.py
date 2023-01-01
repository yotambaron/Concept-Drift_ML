import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from Concept_drift.CD_Experiments.CD_Utils import *


path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Instagram_Followers'

stable_data = pd.read_csv(path + '/training_data.csv')
gradual25_data = pd.read_csv(path + '/gradual_25_12.csv')
gradual50_data = pd.read_csv(path + '/gradual_50_8.csv')
gradual75_data = pd.read_csv(path + '/gradual_75_4.csv')
abrupt25_data = pd.read_csv(path + '/abrupt_25_12.csv')
abrupt50_data = pd.read_csv(path + '/abrupt_50_8.csv')
abrupt75_data = pd.read_csv(path + '/abrupt_75_4.csv')

stable_data = stable_data.sample(frac=1, random_state=42)

full_data = pd.concat([stable_data, gradual25_data, gradual50_data, gradual75_data, abrupt25_data, abrupt50_data, abrupt75_data], ignore_index=True)

class_col = full_data['Class']
del full_data['Class']
full_data['Class'] = class_col

del full_data['Country Block_1']
del full_data['Country Block_2']

full_data.to_csv(path + '/Instagram_Followers_Data.csv', index=False)

columns = full_data.columns
disc_columns = ['Length of Username', 'Number of Followers', 'Number of Posts', 'Number of Mutual Followers',
                'Mean Post Likes', 'Percentage of Following', 'Number of Video Posts', 'Length of Biography']

length = len(full_data)
eq_freq_flag = 1
zero_flag = 1
bins = 5

for ind, col in enumerate(columns):

    if col in disc_columns:

        if eq_freq_flag:
            eq_freq_cuts = np.interp(np.linspace(0, length, bins + 1), np.arange(length), np.sort(full_data.loc[:, col]))
            eq_freq_cuts = eq_freq_cuts[1:len(eq_freq_cuts) - 1]
            full_data.loc[:, col] = bins_disc(full_data.loc[:, col], eq_freq_cuts, zero_flag)

full_data = full_data + 1

full_data.to_csv(path + '/Instagram_Followers_Discretization.csv', index=False)


# -------------------------------------------- Classification Evaluation --------------------------------------------- #

data_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Instagram_Followers'
results_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\Instagram_Followers'
df = pd.read_csv(data_path + '/Instagram_Followers_Data.csv')
y_true = np.array(df['Class'])
del df['Class']

labels = ['bad_user', 'good_user']
jump = 1000
window_days = 1
time_steps = 70
steps = range(1, time_steps + 1)
classes = [0, 1]

# Evaluate Competitors
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.copy())

[adwin_preds, awe_preds, dwm_preds, lnse_preds, srp_preds, adwin_probs, awe_probs, dwm_probs, lnse_probs, srp_probs] = predict_competitors(df_scaled, y_true, jump, window_days, classes)

# Save all algorithms' predictions
pd.DataFrame(adwin_preds).to_csv(results_path + '/Competitors/Predictions_KNN-ADWIN.csv', index=False)
pd.DataFrame(awe_preds).to_csv(results_path + '/Competitors/Predictions_AWE.csv', index=False)
pd.DataFrame(dwm_preds).to_csv(results_path + '/Competitors/Predictions_DWM.csv', index=False)
pd.DataFrame(lnse_preds).to_csv(results_path + '/Competitors/Predictions_LNSE.csv', index=False)
pd.DataFrame(srp_preds).to_csv(results_path + '/Competitors/Predictions_SRP.csv', index=False)

# Save all algorithms' probabilities
pd.DataFrame(adwin_probs).to_csv(results_path + '/Competitors/Probabilities_KNN-ADWIN.csv', index=False)
pd.DataFrame(awe_probs).to_csv(results_path + '/Competitors/Probabilities_AWE.csv', index=False)
pd.DataFrame(dwm_probs).to_csv(results_path + '/Competitors/Probabilities_DWM.csv', index=False)
pd.DataFrame(lnse_probs).to_csv(results_path + '/Competitors/Probabilities_LNSE.csv', index=False)
pd.DataFrame(srp_probs).to_csv(results_path + '/Competitors/Probabilities_SRP.csv', index=False)

# Compute all algorithms' scores
Results_adwin = compute_classification_scores(adwin_preds, adwin_probs, y_true, jump)
Results_awe = compute_classification_scores(awe_preds, awe_probs, y_true, jump)
Results_dwm = compute_classification_scores(dwm_preds, dwm_probs, y_true, jump)
Results_lnse = compute_classification_scores(lnse_preds, lnse_probs, y_true, jump)
Results_srpc = compute_classification_scores(srp_preds, srp_probs, y_true, jump)

# Save all algorithms' results
Results_adwin.to_csv(results_path + '/Competitors/Results_KNN-ADWIN.csv', index=False)
Results_awe.to_csv(results_path + '/Competitors/Results_AWE.csv', index=False)
Results_dwm.to_csv(results_path + '/Competitors/Results_DWM.csv', index=False)
Results_lnse.to_csv(results_path + '/Competitors/Results_LNSE.csv', index=False)
Results_srpc.to_csv(results_path + '/Competitors/Results_SRP.csv', index=False)

