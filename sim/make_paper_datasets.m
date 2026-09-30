% MAKE_PAPER_DATASETS  Render all simulated datasets used in the paper into
% ../data/ (8 files, about 5 GB in total). Run from this folder:
%
%   matlab -batch "make_paper_datasets"
%
% The experiment files in ../experiments/ refer to these file names.

here = fileparts(mfilename('fullpath'));
D = fullfile(fileparts(here), 'data');

% worlds and drives
generate_dataset(struct('outFile', fullfile(D, 'hall.mat')));
generate_dataset(struct('scene', 'indoor', 'routeSeed', 3, 'routeLength', 1000, 'outFile', fullfile(D, 'indoor_r3.mat')));
generate_dataset(struct('scene', 'indoor', 'routeSeed', 5, 'routeLength', 1000, 'seed', 5, 'outFile', fullfile(D, 'indoor_r5.mat')));
generate_dataset(struct('scene', 'indoor', 'routeSeed', 7, 'routeLength', 1000, 'seed', 7, 'outFile', fullfile(D, 'indoor_r7.mat')));
generate_dataset(struct('scene', 'city', 'sceneVariety', 1, 'routeSeed', 3, 'routeLength', 1000, 'outFile', fullfile(D, 'city.mat')));

% degraded sensors (drive of indoor_r3)
generate_dataset(struct('scene', 'indoor', 'routeSeed', 3, 'routeLength', 1000, 'noiseAmplitude', 5e-6, ...
    'outFile', fullfile(D, 'indoor_r3_noise5e-6.mat')));
generate_dataset(struct('scene', 'indoor', 'routeSeed', 3, 'routeLength', 1000, 'noiseAmplitude', 1.5e-5, ...
    'outFile', fullfile(D, 'indoor_r3_noise1.5e-5.mat')));
generate_dataset(struct('scene', 'indoor', 'routeSeed', 3, 'routeLength', 1000, 'noiseAmplitude', 1.5e-5, 'maxRange', 5, ...
    'outFile', fullfile(D, 'indoor_r3_oldsensor.mat')));
