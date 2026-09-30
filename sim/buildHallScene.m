function [env, info] = buildHallScene(freqMin, freqMax, seed)
% BUILDHALLSCENE  A 16m x 10m "cave hall" for BatSLAM 2.0 test data.
%
% Layout (top view, x east, y north):
%
%   +--------------------------------------------------+  y = +5
%   |      north corridor (y ~ +3.1)                    |
%   |   +------------+           +------------+         |
%   |   | left island|  middle   |right island|         |  y = +-1.2
%   |   +------------+   gap     +------------+         |
%   |      south corridor (y ~ -3.1)                    |
%   +--------------------------------------------------+  y = -5
%  x = -8       -5    -2     0     2     5          8
%
% Two wall-bounded islands turn the hall into a ring of corridors plus a
% middle gap, so a trajectory can do outer loops (around both islands)
% and figure-8 / cloverleaf loops (around one island, through the gap).
%
% Landmarks are chosen for three kinds of place-recognition test:
%   * distinctive clusters (pillars of different radii, point clutter,
%     spectral-signature reflectors) that make most places unique;
%   * a deliberate ALIASING TRAP: one pillar cluster in the south
%     corridor and its 180-degree point-symmetric copy in the north
%     corridor. The hall walls and islands are themselves point-symmetric
%     about the origin, so eastbound-south and westbound-north views of
%     those two spots look alike - a single-view matcher will happily
%     confuse them, a sequence matcher should not (the surrounding
%     clutter is NOT symmetric);
%   * heading dependence: the same corridor traversed in the opposite
%     direction yields a different view (templates are X, Y, theta).
arguments
    freqMin (1,1) double = 25e3
    freqMax (1,1) double = 95e3
    seed (1,1) double = 7
end

rng(seed);
env = BatEnvironment();
info = struct();

% --- Hall boundary and islands -------------------------------------------
hall = [-8, 8, -5, 5];
env.addArenaWalls(hall(1), hall(2), hall(3), hall(4), 0.9);

islands = [ ...
    -5.0, -2.0, -1.2, 1.2; ...   % left island  [xmin xmax ymin ymax]
     2.0,  5.0, -1.2, 1.2 ];     % right island
for k = 1:size(islands, 1)
    b = islands(k, :);
    env.addArenaWalls(b(1), b(2), b(3), b(4), 0.8);
end
info.hall = hall;
info.islands = islands;

% --- Aliasing trap: a cluster and its point-symmetric twin ---------------
trap = [ ...  % x, y, radius, reflectivity
    -3.0, -4.2, 0.20, 0.85; ...
    -2.2, -4.4, 0.12, 0.80; ...
    -1.6, -4.1, 0.16, 0.85 ];
for k = 1:size(trap, 1)
    env.addCircle(trap(k,1), trap(k,2), trap(k,3), trap(k,4));
    env.addCircle(-trap(k,1), -trap(k,2), trap(k,3), trap(k,4)); % twin
end
info.aliasTrap = [trap(:,1:2); -trap(:,1:2)];

% --- Distinctive pillars (non-symmetric) ---------------------------------
pillars = [ ...
     1.2, -4.3, 0.25, 0.80; ...   % south corridor, east half
     4.0, -4.4, 0.10, 0.75; ...
     7.2, -2.0, 0.22, 0.80; ...   % east corridor
     7.3,  1.5, 0.12, 0.75; ...
     6.0,  4.3, 0.18, 0.80; ...   % north corridor, east half
    -1.0,  4.4, 0.28, 0.80; ...   % north corridor, west half
    -4.5,  4.2, 0.14, 0.75; ...
    -7.3,  2.2, 0.20, 0.80; ...   % west corridor
    -7.2, -1.8, 0.15, 0.75; ...
     1.0,  0.3, 0.10, 0.70 ];     % middle gap, just east of the path
for k = 1:size(pillars, 1)
    env.addCircle(pillars(k,1), pillars(k,2), pillars(k,3), pillars(k,4));
end

% --- Point clutter hugging walls (keeps corridors free) ------------------
% Random but seeded; placed within 0.6 m of a wall so the drive path
% (corridor centre-lines) stays clear.
nClutter = 26;
clutter = zeros(nClutter, 4);
k = 0;
while k < nClutter
    x = hall(1) + 0.3 + rand * (hall(2) - hall(1) - 0.6);
    y = hall(3) + 0.3 + rand * (hall(4) - hall(3) - 0.6);
    dWall = min([x - hall(1), hall(2) - x, y - hall(3), hall(4) - y]);
    dIsl = inf;
    for i = 1:size(islands, 1)
        b = islands(i, :);
        dx = max([b(1) - x, 0, x - b(2)]);
        dy = max([b(3) - y, 0, y - b(4)]);
        inside = x > b(1) && x < b(2) && y > b(3) && y < b(4);
        if inside; dIsl = -1; else; dIsl = min(dIsl, hypot(dx, dy)); end
    end
    if dIsl < 0; continue; end
    if min(dWall, dIsl) > 0.6; continue; end
    k = k + 1;
    clutter(k, :) = [x, y, 0.02 + 0.12 * rand, 0.35 + 0.3 * rand];
end
for k = 1:nClutter
    env.addReflector(clutter(k, 1:3), clutter(k, 4));
end

% --- Spectral-signature landmarks ----------------------------------------
sigs = { ...
    [ 3.4, -4.5], "notch"; ...
    [ 7.5,  3.8], "highpass"; ...
    [-6.0,  4.5], "lowpass"; ...
    [-3.5,  1.6], "resonant" };   % on the left island's north face
for k = 1:size(sigs, 1)
    p = sigs{k, 1};
    idx = env.addCircle(p(1), p(2), 0.15, 0.9);
    [fAx, g] = BatEnvironment.presetSignature(sigs{k, 2}, freqMin, freqMax);
    env.setSpectralSignature(idx, fAx, g);
end
end
