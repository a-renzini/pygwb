stochastic_paths;

input='/home/andrew.matas/repositories/stochastic_lite/matlab/O2_open_data_comparison/input/';
params=[input 'params.txt'];
jobs=[input 'JOBFILE.txt'];

stochastic(params,jobs,1);

% run post processing over example job
disp('Running postProcessScriptFull.m over example job ...')
% set up parameters
output='../pproc/a0/';
dsc=Inf; % apply delta sigma cut using user specified file rather than in post processing
largeSigmaCutoff=Inf; % do not apply large sigma cut
doRenormalize=0; % we are not applying a new optimal filter (see combineResultsFromMultipleJobs.m)
modifyFilter=0; % not changing the frequency mask
displayResults=1; % plotting errors if this is set to 0
applyBadGPSTimes='../pproc/badGPSTimes.dat'; % File with result of super cut, in this example the file is simply empty


postProcessScriptFull(params,jobs,output,dsc,largeSigmaCutoff,doRenormalize,modifyFilter,displayResults,applyBadGPSTimes);

% run compute_stats2 to get the narrowband Y and sigma
disp('Running compute_stats2.m to convert output to Y(f) and sigma(f)...')
dir='../pproc/a0/';
fileprefixes={'H1L1'};
outputFileName='narrowband_stats.mat';
h0=0.679; % hubble constant
bias=1.05; % bias factor needed for 192s segments / 0.03125 Hz bins
notches=[]; % no extra lines to notch


compute_Yf_and_sigmaf(dir,fileprefixes,outputFileName,h0,bias,notches);
