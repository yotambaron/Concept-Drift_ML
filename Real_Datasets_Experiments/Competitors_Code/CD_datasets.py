import pandas as pd
import numpy as np
from costcla.datasets import load_bankmarketing, load_creditscoring1, load_creditscoring2

save_path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data'

bank_data_loader = load_bankmarketing()
credit_data1_loader = load_creditscoring1()
credit_data2_loader = load_creditscoring2()

bank_desc = bank_data_loader['DESCR']
bank_features = bank_data_loader['feature_names'].tolist()
bank_features.append('Subscribed')
bank_data = bank_data_loader['data']
bank_cost_mat = bank_data_loader['cost_mat']
bank_target = bank_data_loader['target']
bank = pd.DataFrame(np.concatenate([bank_data, bank_target.reshape(-1, 1)], axis=1), columns=bank_features)
bank.to_csv(save_path + '/Bank_Marketing.csv', index=False)

credit1_desc = credit_data1_loader['DESCR']
credit1_features = credit_data1_loader['feature_names'].tolist()
credit1_features.append('Financial_Distress')
credit1_data = credit_data1_loader['data']
credit1_cost_mat = credit_data1_loader['cost_mat']
credit1_target = credit_data1_loader['target']
credit1 = pd.DataFrame(np.concatenate([credit1_data, credit1_target.reshape(-1, 1)], axis=1), columns=credit1_features)
credit1.to_csv(save_path + '/Credit_Card_Scoring1.csv', index=False)

credit2_desc = credit_data2_loader['DESCR']
credit2_features = credit_data2_loader['feature_names'].tolist()
credit2_features.append('Client_Type')
credit2_data = credit_data2_loader['data']
credit2_cost_mat = credit_data2_loader['cost_mat']
credit2_target = credit_data2_loader['target']
credit2 = pd.DataFrame(np.concatenate([credit2_data, credit2_target.reshape(-1, 1)], axis=1), columns=credit2_features)
credit2.to_csv(save_path + '/Credit_Card_Scoring2(PAKDD2009).csv', index=False)
pd.DataFrame(credit2_cost_mat).to_csv(save_path + '/Credit_Card_Scoring2_CostMat.csv', index=False)



