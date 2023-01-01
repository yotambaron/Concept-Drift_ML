Uniform data parameters:
seed 28
60000 observations
jump 3000
window 6000
time steps 20
change point 30000
drift width 5000
number of features categories 4
features 15
n_drifted_features 5
noise precentage 5%

parameters before change point:
classify as 1 if the average of the first n_drifted_features is larger than 2.5


use drift function to decide which function to classify by


parameters after change point:
classify as 1 if the average of the last n_drifted_features is smaller than or equals to 2.6

