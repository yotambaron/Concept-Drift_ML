import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------- Plotting Sin, STAGGER and Uniform ----------------------------------------- #

results_path = r'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Results'
time_steps = 30
change_point = 10
samples_for_time_step = 1000
steps = np.arange(1, time_steps + 1, 1)
steps = pd.DataFrame(steps.reshape(-1, 1))
num_of_experiments = 10

# Plotting parameters for current experiment
experiment = 'Sin'
metric = 'F1'
title = metric + ' Results of ' + experiment + ' Experiment'

# Load results
res_cb = pd.read_csv(results_path + '/' + experiment + '/Sin_CDDRL_CB_Results.csv')
res_ss = pd.read_csv(results_path + '/' + experiment + '/Sin_CDDRL_S&S_Results.csv')
res_comp = pd.read_csv(results_path + '/' + experiment + '/Sin_Competitors_F1.csv')
res_others = pd.read_csv(results_path + '/' + experiment + '/Sin_Rest_Results.csv')

# Load run times
run_time_comp = pd.read_csv(results_path + '/' + experiment + '/Sin_Competitors_Time.csv')
run_time_others = pd.read_csv(results_path + '/' + experiment + '/Sin_Rest_RunTimes.csv')

# Extract results of all algorithms
# CB
results_cb = res_cb['F1']
time_cb = res_cb['RunTime'][0]
tests_cb = res_cb['Tests']
# S&S
results_ss = res_ss['F1']
time_ss = res_ss['RunTime'][0]

comp_res_cols = res_comp.columns
comp_time_cols = run_time_comp.columns
# ADWIN
results_adwin = res_comp[comp_res_cols[1]]
time_adwin = run_time_comp[comp_time_cols[0]][0]
# AWE
results_awe = res_comp[comp_res_cols[2]]
time_awe = run_time_comp[comp_time_cols[1]][0]
# DWM
results_dwm = res_comp[comp_res_cols[3]]
time_dwm = run_time_comp[comp_time_cols[2]][0]
# LNSE
results_lnse = res_comp[comp_res_cols[4]]
time_lnse = run_time_comp[comp_time_cols[3]][0]
# SRP
results_srp = res_comp[comp_res_cols[5]]
time_srp = run_time_comp[comp_time_cols[4]][0]

others_res_cols = res_others.columns
others_time_cols = run_time_others.columns
# MMHC
results_mmhc = res_others[others_res_cols[1]]
time_mmhc = run_time_others[others_time_cols[1]][0]
# K2
results_k2 = res_others[others_res_cols[2]]
time_k2 = run_time_others[others_time_cols[2]][0]
# PC
results_pc = res_others[others_res_cols[3]]
time_pc = run_time_others[others_time_cols[3]][0]
tests_pc = res_others[others_res_cols[4]]


# Plot CDDRL + Others
plt.figure(figsize=(10, 7))
plt.plot(range(1, time_steps+1), results_cb, linestyle='solid')
plt.plot(range(1, time_steps+1), results_ss, linestyle='solid')
plt.plot(range(1, time_steps+1), results_mmhc, linestyle='dashed')
plt.plot(range(1, time_steps+1), results_k2, linestyle='dashed')
plt.plot(range(1, time_steps+1), results_pc, linestyle='dashed')
plt.legend(['CDDRL_CB', 'CDDRL_S&S', 'MMHC', 'K2_Random', 'PC'], loc='lower right', fontsize=12)
plt.title(title, fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel(metric, fontsize=14)
plt.vlines(x=change_point, ymin=0, ymax=1, color='black', linestyles='solid')
plt.text(change_point - 2.4, 0.3, "Change point", rotation=360, verticalalignment='baseline', fontsize=10)
plt.grid()
plt.savefig(results_path + '/' + experiment + '/CDDRL_Others_graphh')

# Plot CDDRL + Competitors
plt.figure(figsize=(10, 7))
plt.plot(range(1, time_steps+1), results_cb, linestyle='solid')
plt.plot(range(1, time_steps+1), results_ss, linestyle='solid')
plt.plot(range(1, time_steps+1), results_adwin, linestyle='dashed')
plt.plot(range(1, time_steps+1), results_awe, linestyle='dashed')
plt.plot(range(1, time_steps+1), results_dwm, linestyle='dashed')
plt.plot(range(1, time_steps+1), results_lnse, linestyle='dashed')
plt.plot(range(1, time_steps+1), results_srp, linestyle='dashed')
plt.legend(['CDDRL_CB', 'CDDRL_S&S', 'ADWIN', 'AWE', 'DWM', 'LNSE', 'SRP'], loc='lower right', fontsize=12)
plt.title(title, fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel(metric, fontsize=14)
plt.vlines(x=change_point, ymin=0, ymax=1, color='black', linestyles='solid')
plt.text(change_point - 2.4, 0.3, "Change point", rotation=360, verticalalignment='baseline', fontsize=10)
plt.grid()
plt.savefig(results_path + '/' + experiment + '/CDDRL_Comp_graphh')

# Plot Run times
run_times = [time_cb, time_ss, time_adwin, time_awe, time_dwm, time_lnse, time_srp, time_mmhc, time_k2, time_pc]
labels = ['CB', 'SS', 'ADWIN', 'AWE', 'DWM', 'LNSE', 'SRP', 'MMHC', 'K2', 'PC']
plt.figure(figsize=(10, 7))
plt.bar(labels, run_times, 0.4, 0)
plt.title('Run Time All Algorithms', fontsize=14)
plt.ylabel('Run Time (Seconds)', fontsize=14)
for i, v in enumerate(run_times):
    plt.text(i - 0.25, v + 1, str(round(v, 2)), color='black')
plt.savefig(results_path + '/' + experiment + '/RunTimes_All_Algorithms')


# --------------------------------------------- Plotting BN Experiments ---------------------------------------------- #

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.01*h, str(round(h, 2)), ha='center', va='bottom', fontsize=7)


# Plotting parameters for current experiment
experiment = 'Andes'
metric = 'SHDs Parts'
title = metric + ' Results of ' + experiment + ' Experiment'

# Load results
results_path = r'C:\Users\User\Desktop\yotam\MATLAB\Constrain-Based CDDRL\Results'
shds = pd.read_csv(results_path + '/' + experiment + '/' + experiment + '_Avg_SHDs.csv')
me = pd.read_csv(results_path + '/' + experiment + '/' + experiment + '_Avg_MEs.csv')
ee = pd.read_csv(results_path + '/' + experiment + '/' + experiment + '_Avg_EEs.csv')
wd = pd.read_csv(results_path + '/' + experiment + '/' + experiment + '_Avg_WDs.csv')
run_times = pd.read_csv(results_path + '/' + experiment + '/' + experiment + '_Avg_Runtime.csv')
ci_tests = pd.read_csv(results_path + '/' + experiment + '/' + experiment + '_Avg_CI_Tests.csv')

N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.14       # the width of the bars
x_labels = ['500', '1500', '5000', '15000', '50000']
X_axis = np.arange(len(x_labels))

CB = list(shds.iloc[0, :][1:])
SS = list(shds.iloc[1, :][1:])
MMHC = list(shds.iloc[2, :][1:])
K2_bef = list(shds.iloc[3, :][1:])
K2_aft = list(shds.iloc[4, :][1:])
PC = list(shds.iloc[5, :][1:])

plt.figure(figsize=(15, 7))
cb_plot = plt.bar(ind - 2 * width, CB, width=width,  align='center')
ss_plot = plt.bar(ind - width, SS, width=width, align='center')
mmhc_plot = plt.bar(ind, MMHC, width=width, align='center')
k2bef_plot = plt.bar(ind + width, K2_bef, width=width, align='center')
k2aft_plot = plt.bar(ind + 2 * width, K2_aft, width=width, align='center')
pc_plot = plt.bar(ind + 3 * width, PC, width=width, align='center')

autolabel(cb_plot)
autolabel(ss_plot)
autolabel(mmhc_plot)
autolabel(k2bef_plot)
autolabel(k2aft_plot)
autolabel(pc_plot)

plt.title(experiment + ' ' + metric + ' for all DB sizes', fontsize=14)
plt.legend(['CDDRL_CB', 'CDDRL_S&S', 'MMHC', 'K2_Before', 'K2_After', 'PC'], loc='upper right', fontsize=10)
plt.xticks(X_axis, x_labels)
plt.ylabel('SHD')

plt.savefig(results_path + '/' + experiment + '/SHDs_graph')


# CI plotting

CB_tests = list(ci_tests.iloc[0, :][1:])
PC_tests = list(ci_tests.iloc[1, :][1:])

plt.figure(figsize=(7, 5))
cb_plot = plt.bar(ind - 0.2, CB_tests, width=0.4, color='tab:blue', align='center')
pc_plot = plt.bar(ind + 0.2, PC_tests, width=0.4, color='tab:brown', align='center')

autolabel(cb_plot)
autolabel(pc_plot)

plt.title(experiment + ' CI tests for all DB sizes', fontsize=14)
plt.legend(['CDDRL_CB', 'PC'], loc='upper left', fontsize=12)
plt.xticks(X_axis, x_labels)
plt.ylabel('CI Tests')

plt.savefig(results_path + '/' + experiment + '/CI_Tests_graph')


# SHDs parts plotting

me = list(me.iloc[0, :][1:])
ee = list(ee.iloc[0, :][1:])
wd = list(wd.iloc[0, :][1:])

plt.figure(figsize=(8, 5))
me_plot = plt.bar(ind - width, me, width=width,  align='center')
ee_plot = plt.bar(ind, ee, width=width, align='center')
wd_plot = plt.bar(ind + width, wd, width=width, align='center')

autolabel(me_plot)
autolabel(ee_plot)
autolabel(wd_plot)

plt.title(experiment + ' ' + metric + ' for all DB sizes', fontsize=14)
plt.legend(['CB_ME', 'CB_EE', 'CB_WD'], loc='upper right', fontsize=10)
plt.xticks(X_axis, x_labels)
plt.ylabel('SHDs Mistakes')

plt.savefig(results_path + '/' + experiment + '/SHDS_parts_graph')


