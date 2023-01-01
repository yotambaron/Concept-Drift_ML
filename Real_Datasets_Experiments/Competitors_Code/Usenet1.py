from Concept_drift.CD_Experiments.CD_Utils import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np


# -------------------------------------------- Classification Evaluation --------------------------------------------- #

data_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Usenet'
results_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\Usenet1'
df = pd.read_csv(data_path + '/usenet1.csv')
y_true = np.array(df['interesting'])
del df['interesting']

labels = ['not_interesting', 'interesting']
jump = 100
window_days = 1
time_steps = 15
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

