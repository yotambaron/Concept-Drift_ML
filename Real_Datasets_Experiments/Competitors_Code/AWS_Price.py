from Concept_drift.CD_Experiments.CD_Utils import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd

path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\AWSPrice'
aws_price = pd.read_csv(path + '/AWSPrice ca-central-1_processed.csv')
columns = aws_price.columns
aws_price = aws_price.dropna()
#
# aws_price['Price'] = np.where(aws_price['Price'] < 0.025, 1, np.where(aws_price['Price'] <= 0.2, 2, 3))
#
# le = LabelEncoder()
# aws_price['Instance_type'] = le.fit_transform(aws_price['Instance_type']) + 1
#
# aws_price['day'] = pd.to_datetime(aws_price['Date_time']).dt.day
# aws_price['hour'] = pd.to_datetime(aws_price['Date_time']).dt.hour
# aws_price['minute'] = pd.to_datetime(aws_price['Date_time']).dt.minute
#
# del aws_price['Date_time']
#
# aws_price['hour'] = np.where(aws_price['hour'] == 0, 24, aws_price['hour'])
#
# aws_price['minute'] = np.where(aws_price['minute'] < 5, 1,
#                       np.where(aws_price['minute'] < 10, 2,
#                       np.where(aws_price['minute'] < 15, 3,
#                       np.where(aws_price['minute'] < 20, 4,
#                       np.where(aws_price['minute'] < 25, 5,
#                       np.where(aws_price['minute'] < 30, 6,
#                       np.where(aws_price['minute'] < 35, 7,
#                       np.where(aws_price['minute'] < 40, 8,
#                       np.where(aws_price['minute'] < 45, 9,
#                       np.where(aws_price['minute'] < 50, 10,
#                       np.where(aws_price['minute'] < 55, 11, 12)))))))))))
#
# aws_price.to_csv(path + '/AWSPrice_Discretization.csv', index=False)

# ---------------------------------------------- Multiclass Evaluation ----------------------------------------------- #

data_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\AWSPrice'
results_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Results\AWS_Price'
y_pred = np.array(pd.read_csv(results_path + '/CDDRL/Tabu10+Params/Predictions.csv', header=None))
y_probs = np.array(pd.read_csv(results_path + '/CDDRL/Tabu10+Params/Scores.csv', header=None))
df = pd.read_csv(data_path + '/AWSPrice_Discretization.csv')
y_true = np.array(df['Price'])
del df['Price']

labels = ['low', 'mid', 'high']
jump = 3700
window_days = 2
time_steps = 54
steps = range(1, time_steps + 1)
classes = [0, 1, 2]

# Evaluate CDDRL
# cddrl_results = compute_multiclass_scores(y_pred, y_probs, y_true, labels, jump)
# cddrl_results.to_csv(results_path + '/CDDRL/All_Results.csv', index=False)

# Evaluate Competitors
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df.copy())

[adwin_preds, awe_preds, dwm_preds, lnse_preds, srp_preds, adwin_probs, awe_probs, dwm_probs, lnse_probs, srp_probs] = predict_competitors(df_scaled, y_true - 1, jump, window_days, classes)

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
Results_adwin = compute_multiclass_scores(adwin_preds + 1, adwin_probs, y_true, labels, jump)
Results_awe = compute_multiclass_scores(awe_preds + 1, awe_probs, y_true, labels, jump)
Results_dwm = compute_multiclass_scores(dwm_preds + 1, dwm_probs, y_true, labels, jump)
Results_lnse = compute_multiclass_scores(lnse_preds + 1, lnse_probs, y_true, labels, jump)
Results_srpc = compute_multiclass_scores(srp_preds + 1, srp_probs, y_true, labels, jump)

# Save all algorithms' results
Results_adwin.to_csv(results_path + '/Competitors/Results_KNN-ADWIN.csv', index=False)
Results_awe.to_csv(results_path + '/Competitors/Results_AWE.csv', index=False)
Results_dwm.to_csv(results_path + '/Competitors/Results_DWM.csv', index=False)
Results_lnse.to_csv(results_path + '/Competitors/Results_LNSE.csv', index=False)
Results_srpc.to_csv(results_path + '/Competitors/Results_SRP.csv', index=False)





