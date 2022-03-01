%
% This is an example of a startup.m file which you can run automatically 
% every time you launch matlab to set your matlab paths.  These commands will
% set up the paths needed for the stochastic pipeline.
% Just edit the subsequent lines and modify for your own home
% directory.
%

% Edit this to suit your local installation
stoch_install = '/home/stochastic/O3/stochastic-pipeline/O3StochIso';
%stoch_install = '/home/andrew.matas/repositories/stochastic-pipeline';

% Add the required paths
fprintf('loading stochastic packages...');
addpath([ stoch_install '/Utilities' ]);
addpath([ stoch_install '/Utilities/ligotools/matlab' ]);

addpath([ stoch_install '/CrossCorr/src_cc' ]);
addpath([ stoch_install '/PostProcessing' ]);

addpath([ stoch_install '/Utilities/Channel' ]);
addpath([ stoch_install '/Utilities/detgeom/matlab' ]);
addpath([ stoch_install '/Utilities/misc' ]);
addpath([ stoch_install '/Utilities/FTSeries' ]);

fprintf('done.\n');
