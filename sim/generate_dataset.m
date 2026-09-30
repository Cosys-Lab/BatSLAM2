function outFile = generate_dataset(opts)
% GENERATE_DATASET  Render a BatSLAM 2.0 dataset with the binaural sonar
% simulator: ground-truth poses, noisy odometry, and raw binaural echoes.
%
%   generate_dataset()                         % the hall, into ../data/hall.mat
%   generate_dataset(struct('ds', 0.2))        % override fields
%   generate_dataset(struct('scene', 'city', 'routeLength', 1000, ...
%       'outFile', '../data/city.mat'))
%
% The robot is placed directly on the path (buildRoute / buildCityRoute +
% followPath), without a controller, with one sonar pulse per path sample.
% make_paper_datasets.m lists the settings of all datasets in the paper.
%
% Output (.mat, -v7, readable with scipy.io.loadmat): see the dataset format
% in the top-level README.
arguments
    opts struct = struct()
end

here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(here);

% Sensor defaults: noise amplitude 1.5e-6 and 10 m range.
o = struct( ...
    'simRoot',        fullfile(here, 'simulator'), ...
    'species',        'HRTF_Phyllostomus_Discolor_azel', ...   % see simulator/README.md
    'scene',          'hall', ...    % 'hall' | 'city' (buildCityScene) | 'indoor' (buildIndoorScene)
    'corridorWidth',  1.5, ...       % indoor only: nominal corridor width (m)
    'routeSeed',      3, ...         % city only: random-walk seed
    'routeLength',    1000, ...      % city only: tour length (m)
    'sceneVariety',   1, ...         % city only: 0 = copy-paste grid, 1 = varied
    'wallRoughness',  0, ...         % city only: diffuse wall scattering strength (0 = specular walls)
    'ds',             0.15, ...      % m between pulses
    'speed',          1.5, ...       % m/s (sets timestamps only)
    'maxRange',       10.0, ...      % m, renderer cut-off
    'noiseAmplitude', 1.5e-6, ...    % additive sensor noise
    'callDuration',   0.003, ...
    'odomSigmaFwd',   0.03, ...      % m per m travelled (random)
    'odomSigmaLat',   0.01, ...      % m per m travelled (random)
    'odomSigmaRot',   deg2rad(1.0), ... % rad per m travelled (random)
    'odomBiasRot',    deg2rad(0.6), ... % rad per m travelled (systematic drift)
    'odomBiasScale',  0.02, ...      % fractional forward scale error
    'seed',           1, ...
    'maxPulses',      Inf, ...       % truncate for quick tests
    'useParfor',      true, ...
    'outFile',        fullfile(projectRoot, 'data', 'hall.mat'));
fn = fieldnames(opts);
for k = 1:numel(fn); o.(fn{k}) = opts.(fn{k}); end

addpath(fullfile(o.simRoot, 'src'));
addpath(here);

hrtf = HRTFModel(fullfile(o.simRoot, 'data', [o.species '.mat']));
renderer = SonarRenderer('MaxRange', o.maxRange, 'NoiseAmplitude', o.noiseAmplitude, ...
    'CallDuration', o.callDuration);
switch o.scene
    case 'hall'
        [env, sceneInfo] = buildHallScene(hrtf.FreqVec(1), hrtf.FreqVec(end));
        route = buildRoute();
    case 'city'
        [env, sceneInfo] = buildCityScene(hrtf.FreqVec(1), hrtf.FreqVec(end), 11, o.sceneVariety, o.wallRoughness);
        route = buildCityRoute(sceneInfo, o.routeSeed, o.routeLength);
    case 'indoor'
        [env, sceneInfo] = buildIndoorScene(hrtf.FreqVec(1), hrtf.FreqVec(end), 11, o.corridorWidth);
        % narrow corridors: tight corner rounding, small lateral wobble
        route = buildCityRoute(sceneInfo, o.routeSeed, o.routeLength, 120, 0.7, [0.03, 0.10]);
    otherwise
        error('generate_dataset:scene', 'unknown scene "%s"', o.scene);
end
traj = followPath(route, o.ds);

N = min(size(traj.pose, 1), o.maxPulses);
pose = traj.pose(1:N, :);
fprintf('BatSLAM 2.0 dataset: %d pulses, %.1f m path, %d reflectors\n', N, traj.s(N), env.numReflectors());

% --- Clearance check (the path is hand-made, so verify it) ---------------
robotProto = BatRobot();
minClear = inf; worst = 0;
for i = 1:N
    p = [pose(i, 1:2), robotProto.HeadHeight];
    near = find(vecnorm(env.Positions(:, 1:2) - p(1:2), 2, 2) < 2.5)';  % walls are <= 1.5 m segments
    for j = near
        q = env.nearestPoint(j, p);
        if any(isnan(q)); continue; end
        d = norm(q(1:2) - p(1:2));
        if d < minClear; minClear = d; worst = i; end
    end
end
fprintf('  min clearance to any reflector: %.2f m (pulse %d)\n', minClear, worst);
if minClear < robotProto.Radius
    warning('generate_dataset:collision', 'path passes within robot radius of a reflector');
end

% --- Render echoes -------------------------------------------------------
[callWave, Fs] = renderer.synthesizeCall(hrtf);
Nfft = 2^nextpow2(2 * numel(callWave));
L = ceil((2 * o.maxRange + robotProto.EarBaseline) / renderer.SpeedOfSound * Fs) + Nfft + 1;
echoL = zeros(N, L, 'int16'); echoR = zeros(N, L, 'int16');
scale = zeros(N, 1);
rngSeeds = o.seed * 100000 + (1:N);

tic;
if o.useParfor
    parfor i = 1:N
        [l, r, s] = renderPulse(pose(i, :), env, hrtf, renderer, L, rngSeeds(i));
        echoL(i, :) = l; echoR(i, :) = r; scale(i) = s;
    end
else
    for i = 1:N
        [l, r, s] = renderPulse(pose(i, :), env, hrtf, renderer, L, rngSeeds(i));
        echoL(i, :) = l; echoR(i, :) = r; scale(i) = s;
        if mod(i, 100) == 0; fprintf('  %d/%d\n', i, N); end
    end
end
fprintf('  rendered in %.1f s\n', toc);

% --- Odometry ------------------------------------------------------------
rng(o.seed);
gtDelta = zeros(N-1, 3); odomDelta = zeros(N-1, 3);
for i = 1:N-1
    gtDelta(i, :) = relativePose(pose(i, :), pose(i+1, :));
    d = norm(gtDelta(i, 1:2));
    noise = [o.odomSigmaFwd, o.odomSigmaLat, o.odomSigmaRot] .* sqrt(d) .* randn(1, 3);
    odomDelta(i, :) = gtDelta(i, :) .* [1 + o.odomBiasScale, 1, 1] ...
        + [0, 0, o.odomBiasRot * d] + noise;
end

% --- Export --------------------------------------------------------------
reflectors = struct();
reflectors.position = env.Positions;
reflectors.type = char(env.Type);             % char matrix (scipy-friendly)
reflectors.radius = env.Radius;
reflectors.endpoint1 = env.Endpoint1;
reflectors.endpoint2 = env.Endpoint2;
reflectors.reflectivity = env.Reflectivity;
reflectors.has_signature = double(~cellfun(@isempty, env.SpectralSignature));

meta = struct();
meta.species = o.species;
meta.scene = o.scene;
meta.scene_variety = o.sceneVariety;
meta.wall_roughness = o.wallRoughness;
meta.fs = Fs;
meta.speed_of_sound = renderer.SpeedOfSound;
meta.max_range = o.maxRange;
meta.noise_amplitude = o.noiseAmplitude;
meta.call = callWave(:)';
meta.ear_baseline = robotProto.EarBaseline;
meta.ds = o.ds;
meta.odom_sigma = [o.odomSigmaFwd, o.odomSigmaLat, o.odomSigmaRot];
meta.odom_bias_rot = o.odomBiasRot;
meta.odom_bias_scale = o.odomBiasScale;
meta.seed = o.seed;
meta.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
meta.lap_names = char([route.name]');
meta.hall = sceneInfo.hall;
meta.islands = sceneInfo.islands;
meta.alias_trap = sceneInfo.aliasTrap;

data = struct();
data.gt_pose = pose;                 % [N x 3] x y theta(rad)
data.gt_delta = gtDelta;             % [N-1 x 3] body-frame increments
data.odom_delta = odomDelta;         % [N-1 x 3] noisy body-frame increments
data.t = traj.s(1:N) / o.speed;
data.lap = traj.lap(1:N);
data.echo_left = echoL;              % int16, multiply row i by echo_scale(i)
data.echo_right = echoR;
data.echo_scale = scale;
data.reflectors = reflectors;
data.meta = meta;
data.waypoints = vertcat(route.waypoints);

outDir = fileparts(o.outFile);
if ~exist(outDir, 'dir'); mkdir(outDir); end
save(o.outFile, '-struct', 'data', '-v7');
info = dir(o.outFile);
fprintf('  wrote %s (%.1f MB)\n', o.outFile, info.bytes / 1e6);
outFile = o.outFile;
end

function [l, r, s] = renderPulse(p, env, hrtf, renderer, L, seed)
rng(seed);
robot = BatRobot();
robot.X = p(1); robot.Y = p(2); robot.Theta = rad2deg(p(3));
[sigL, sigR] = renderer.renderEcho(robot, env, hrtf);
sigL = fitLength(sigL, L, renderer.NoiseAmplitude);
sigR = fitLength(sigR, L, renderer.NoiseAmplitude);
s = max([abs(sigL); abs(sigR); eps]) / 32767;
l = int16(round(sigL' / s));
r = int16(round(sigR' / s));
end

function x = fitLength(x, L, noiseAmp)
if numel(x) >= L
    x = x(1:L);
else
    x = [x; noiseAmp * randn(L - numel(x), 1)];
end
end

function d = relativePose(a, b)
% Pose of b expressed in a's body frame: [dx dy dtheta].
c = cos(a(3)); s = sin(a(3));
dxy = [b(1) - a(1), b(2) - a(2)];
d = [c*dxy(1) + s*dxy(2), -s*dxy(1) + c*dxy(2), b(3) - a(3)];
end
