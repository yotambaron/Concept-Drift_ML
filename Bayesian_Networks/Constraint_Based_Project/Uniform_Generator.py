import pandas as pd
import numpy as np
import random


# ---------------------------------------------- Uniform Drift Old --------------------------------------------------- #

# random.seed(28)
# n_features = 60
# obs = 6000
# stable_pos = 4000
# drift_pos = 2000
# drift_width = 500
# n_cat = 6
# n_drift_features = 5
# virtual_mag_drift = 0.00005
# noise_percentage = 0.05
# path = r'C:\Users\yotam\Desktop\yotam\CD_Project\DBs\Uniform'
#
# # initiate dataframe with correct size
# cols = range(1, n_features + 2, 1)
# data = pd.DataFrame(columns=cols)
#
# # fill dataframe
# for i in range(1, obs + 1):
#
#     data.at[i, :n_features] = np.random.uniform(1, n_cat, n_features) # generate random uniform numbers to the current row
#
#     # before drift position classify as 1 if the mean of the drifted features is above 3.5 and 0 otherwise
#     if i <= drift_pos:
#         if np.array(np.mean(data.loc[i - 1:i, 0:n_drift_features + 1], axis=1))[0] >= 3.5:
#             data.at[i, n_features + 1] = 1
#         else:
#             data.at[i, n_features + 1] = 0
#
#     # if we are at the drift period
#     else:
#         if i <= stable_pos:
#             # apply virtual drift to the drifted features
#             # data.loc[i:i, :n_drift_features] += virtual_mag_drift * (i - drift_pos)
#             # data[data > 6] = 6
#             # decide what concept is currently happening by the drift's width
#             ft = 1 / (1 + np.exp(-4 * (i - drift_pos) / drift_width))
#             if ft > random.uniform(0, 1):   # new concept
#                 if np.array(np.mean(data.loc[i - 1:i, 0:n_drift_features + 1], axis=1))[0] <= 3.5:
#                     data.at[i, n_features + 1] = 1
#                 else:
#                     data.at[i, n_features + 1] = 0
#             else:   # first concept
#                 if np.array(np.mean(data.loc[i - 1:i, 0:n_drift_features + 1], axis=1))[0] >= 3.5:
#                     data.at[i, n_features + 1] = 1
#                 else:
#                     data.at[i, n_features + 1] = 0
#
#         # stable on the new concept
#         else:
#             data.loc[i:i, :n_drift_features] += virtual_mag_drift * (i - stable_pos)
#             if np.array(np.mean(data.loc[i - 1:i, 0:n_drift_features + 1], axis=1))[0] <= 3.5:
#                 data.at[i, n_features + 1] = 1
#             else:
#                 data.at[i, n_features + 1] = 0
#
#     # apply noise
#     if random.uniform(0, 1) <= noise_percentage:
#         if data.at[i, n_features + 1] == 0:
#             data.at[i, n_features + 1] = 1
#         else:
#             data.at[i, n_features + 1] = 0
#
# # save dataframe
# data.to_csv(path + '/df_uniform_sudden.csv', index=False)


