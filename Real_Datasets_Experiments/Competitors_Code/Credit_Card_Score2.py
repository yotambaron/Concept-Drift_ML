from Concept_drift.CD_Experiments.CD_Utils import *

path = r'D:\yotam\MATLAB\Stress_Experiments\CD_Real_Datasets\Data\Credit_Card_Scoring2_PAKDD2009'
credit_card = pd.read_csv(path + '/Credit_Card_Scoring2_(PAKDD2009)_Processed.csv')
columns = credit_card.columns

length = len(credit_card)
eq_freq_flag = 1
zero_flag = 1

bins = 5
if eq_freq_flag:
    eq_freq_cuts = np.interp(np.linspace(0, length, bins + 1), np.arange(length), np.sort(credit_card['AGE']))
    eq_freq_cuts = eq_freq_cuts[1:len(eq_freq_cuts)-1]
    credit_card['AGE'] = bins_disc(credit_card['AGE'], eq_freq_cuts, zero_flag)
else:
    credit_card['AGE'] = bins_disc(credit_card['AGE'], [20, 30, 40, 50], zero_flag)

credit_card['PAYMENT_DAY'] = bins_disc(credit_card['PAYMENT_DAY'], [10, 20], zero_flag)

bins = 4
eq_freq_cuts = np.interp(np.linspace(0, length, bins + 1), np.arange(length), np.sort(credit_card['MONTHS_IN_RESIDENCE']))
eq_freq_cuts = eq_freq_cuts[1:len(eq_freq_cuts) - 1]
credit_card['MONTHS_IN_RESIDENCE'] = bins_disc(credit_card['MONTHS_IN_RESIDENCE'], eq_freq_cuts, zero_flag)

credit_card['MONTHS_IN_THE_JOB'] = bins_disc(credit_card['MONTHS_IN_THE_JOB'], [6, 12, 36, 84], zero_flag)

credit_card['QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION'] = np.where(credit_card['QUANT_ADDITIONAL_CARDS_IN_THE_APPLICATION'] == 0, 0, 1)

bins = 5
eq_freq_cuts = np.interp(np.linspace(0, length, bins + 1), np.arange(length), np.sort(credit_card['PERSONAL_NET_INCOME']))
eq_freq_cuts = eq_freq_cuts[1:len(eq_freq_cuts) - 1]
credit_card['PERSONAL_NET_INCOME'] = bins_disc(credit_card['PERSONAL_NET_INCOME'], eq_freq_cuts, zero_flag)
credit_card['PERSONAL_NET_INCOME'].value_counts()

credit_card.to_csv(path + '/Credit_Card_Scoring2_Discretization.csv', index=False)


