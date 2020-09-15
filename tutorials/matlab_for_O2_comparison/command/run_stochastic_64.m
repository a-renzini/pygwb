stochastic_paths;

input='/home/andrew.matas/repositories/stochastic_lite/matlab/O2_open_data_comparison/input/';
params=[input 'params_64.txt'];
jobs=[input 'JOBFILE.txt'];

stochastic(params,jobs,1);
