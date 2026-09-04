% startup.m — SIH26038 MATLAB engineering workflow path setup.
% Run from the matlab/ directory.
root = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(root, 'src')));
addpath(fullfile(root, 'config'));
addpath(fullfile(root, 'examples'));
fprintf('SIH26038 MATLAB path ready: %s\n', root);
