function [env, info] = buildIndoorScene(freqMin, freqMax, seed, corridorWidth)
% BUILDINDOORSCENE  Indoor office-like floor for BatSLAM 2.0: narrow corridors.
%
%   west part: a 3 x 3 grid of rooms (5-7 m blocks: notched walls,
%     chamfered corners, one open atrium with pillars instead of a room)
%     separated by ~1.5 m corridors (1.4-1.8 m, irregular) - 16 corridor
%     junctions;
%   east part: a 14 m wide pillar hall reachable through two doorways,
%     with a lane network kept clear of pillars;
%   clutter: wall-mounted items (pipes, radiators, door frames) within
%     0.05-0.2 m of the walls, plus thin posts, so the corridor centre
%     stays clear for the robot;
%   alias trap: one identical 3-item wall-mounted pattern on two
%     corridor walls with the same orientation.
%
% Walls are specular and built from segments <= 1.5 m (SonarRenderer
% range-gates walls on their midpoint distance). info has the same fields
% as buildCityScene, so buildCityRoute works unchanged.
arguments
    freqMin (1,1) double = 25e3
    freqMax (1,1) double = 95e3
    seed (1,1) double = 11
    corridorWidth (1,1) double = 1.5
end

rng(seed);
env = BatEnvironment();
info = struct();
maxSeg = 1.5;
wallRefl = 0.85;

wX = corridorWidth * [1.07, 0.93, 1.2, 1.0];     % ~1.4 .. 1.8 m for 1.5
wY = corridorWidth * [1.0, 1.13, 0.93, 1.07];
bw = [5.0, 7.0, 6.0]; bh = [4.5, 3.5, 4.2];
streetX = zeros(1, 4); streetY = zeros(1, 4);
streetX(1) = wX(1)/2; streetY(1) = wY(1)/2;
for k = 1:3
    streetX(k+1) = streetX(k) + wX(k)/2 + bw(k) + wX(k+1)/2;
    streetY(k+1) = streetY(k) + wY(k)/2 + bh(k) + wY(k+1)/2;
end
gridX = [0, streetX(end) + wX(end)/2];
gridY = [0, streetY(end) + wY(end)/2];
hallX = [gridX(2), gridX(2) + 14];
bound = [0, hallX(2), 0, gridY(2)];
info.hall = bound;
info.streetX = streetX; info.streetY = streetY;

% --- outer boundary -------------------------------------------------------
segWall(env, bound(1), bound(3), bound(2), bound(3), maxSeg, wallRefl);
segWall(env, bound(2), bound(3), bound(2), bound(4), maxSeg, wallRefl);
segWall(env, bound(2), bound(4), bound(1), bound(4), maxSeg, wallRefl);
segWall(env, bound(1), bound(4), bound(1), bound(3), maxSeg, wallRefl);
posts(env, [bound(1) bound(3); bound(2) bound(3); bound(2) bound(4); bound(1) bound(4)]);

% --- rooms ----------------------------------------------------------------
types = ["plaza", "notch", "notch", "notch", "chamfer", "chamfer", "both", "rect", "rect"];
types = types(randperm(9));
blocks = zeros(0, 4);
bi = 0;
for i = 1:3
    for j = 1:3
        bi = bi + 1;
        b = [streetX(i) + wX(i)/2, streetX(i+1) - wX(i+1)/2, streetY(j) + wY(j)/2, streetY(j+1) - wY(j+1)/2];
        blocks(end+1, :) = b; %#ok<AGROW>
        if types(bi) == "plaza"
            % open atrium: a loose pillar cluster, kept >= 1 m from the corridors
            P = zeros(0, 2);
            while size(P, 1) < 5
                p = [b(1) + 1.0 + rand * (b(2) - b(1) - 2.0), b(3) + 1.0 + rand * (b(4) - b(3) - 2.0)];
                if ~isempty(P) && min(vecnorm(P - p, 2, 2)) < 0.9; continue; end
                P(end+1, :) = p; %#ok<AGROW>
                env.addCircle(p(1), p(2), 0.1 + 0.15 * rand, 0.8);
            end
            continue
        end
        polyWall(env, blockPolygon(b, types(bi)), maxSeg, wallRefl);
    end
end
info.islands = blocks;
info.blockTypes = types;

% --- divider wall between rooms and hall, two doorways ---------------------
xd = gridX(2);
doorY = [streetY(2), streetY(3)];
dw = [wY(2), wY(3)] / 2;
edges = [bound(3), doorY(1) - dw(1); doorY(1) + dw(1), doorY(2) - dw(2); doorY(2) + dw(2), bound(4)];
for k = 1:size(edges, 1)
    segWall(env, xd, edges(k,1), xd, edges(k,2), maxSeg, wallRefl);
    posts(env, [xd edges(k,1); xd edges(k,2)]);
end

% --- hall lane network (relative to the hall) -----------------------------
laneX = hallX(1) + [5, 11];
laneY = [2.5, doorY(1), doorY(2), bound(4) - 2.5];
hallNodes = zeros(0, 2);
for yy = laneY
    for xx = laneX
        hallNodes(end+1, :) = [xx, yy]; %#ok<AGROW>
    end
end
idx = @(ix, iy) (iy - 1) * numel(laneX) + ix;
hallEdges = zeros(0, 2);
for iy = 1:numel(laneY)
    hallEdges(end+1, :) = [idx(1, iy), idx(2, iy)]; %#ok<AGROW>
end
for ix = 1:numel(laneX)
    for iy = 1:numel(laneY) - 1
        hallEdges(end+1, :) = [idx(ix, iy), idx(ix, iy + 1)]; %#ok<AGROW>
    end
end
info.hallNodes = hallNodes;
info.hallEdges = hallEdges;
info.doorNodes = [streetX(end), doorY(1); streetX(end), doorY(2)];
info.doorHallIdx = [idx(1, 2); idx(1, 3)];
laneSegs = zeros(0, 4);
for e = 1:size(hallEdges, 1)
    laneSegs(end+1, :) = [hallNodes(hallEdges(e,1), :), hallNodes(hallEdges(e,2), :)]; %#ok<AGROW>
end
for k = 1:2
    laneSegs(end+1, :) = [info.doorNodes(k, :), hallNodes(info.doorHallIdx(k), :)]; %#ok<AGROW>
end

% --- pillar forest in the hall --------------------------------------------
placed = zeros(0, 3); tries = 0;
while size(placed, 1) < 40 && tries < 20000
    tries = tries + 1;
    p = [hallX(1) + 1 + rand * (hallX(2) - hallX(1) - 2), bound(3) + 1 + rand * (bound(4) - bound(3) - 2)];
    r = 0.08 + 0.22 * rand;
    if minSegDist(p, laneSegs) < 1.2 + r; continue; end
    if ~isempty(placed) && min(vecnorm(placed(:,1:2) - p, 2, 2) - placed(:,3)) < 0.9 + r; continue; end
    placed(end+1, :) = [p, r]; %#ok<AGROW>
end
for k = 1:size(placed, 1)
    env.addCircle(placed(k,1), placed(k,2), placed(k,3), 0.6 + 0.3 * rand);
end

% --- alias trap: identical wall-mounted pattern on two south-facing walls --
solid = find(types ~= "plaza");
faces = solid([1, end]);                      % two different rooms
pattern = [0.6, 0.10, 0.06; 1.1, 0.12, 0.04; 1.9, 0.10, 0.07];   % along-face, off-face, radius
trap = zeros(0, 2);
for k = 1:2
    b = blocks(faces(k), :);
    for m = 1:size(pattern, 1)
        c = [b(1) + pattern(m, 1), b(3) - pattern(m, 2)];   % below the room's south wall
        env.addCircle(c(1), c(2), pattern(m, 3), 0.85);
        trap(end+1, :) = c; %#ok<AGROW>
    end
end
info.aliasTrap = trap;

% --- wall-mounted clutter (pipes, radiators, frames) ----------------------
nClutter = 170; k = 0; tries = 0;
while k < nClutter && tries < 100000
    tries = tries + 1;
    p = [bound(1) + 0.1 + rand * (gridX(2) - 0.2), bound(3) + 0.1 + rand * (bound(4) - 0.2)];
    dFace = inf; inside = false;
    for bb = solid
        B = blocks(bb, :);
        if p(1) > B(1) && p(1) < B(2) && p(2) > B(3) && p(2) < B(4); inside = true; break; end
        dx = max([B(1) - p(1), 0, p(1) - B(2)]); dy = max([B(3) - p(2), 0, p(2) - B(4)]);
        dFace = min(dFace, hypot(dx, dy));
    end
    if inside; continue; end
    dFace = min([dFace, p(1) - bound(1), gridX(2) - p(1), p(2) - bound(3), bound(4) - p(2)]);
    if dFace < 0.05 || dFace > 0.2; continue; end
    k = k + 1;
    env.addReflector([p, 0.02 + 0.12 * rand], 0.35 + 0.35 * rand);
end

% --- thin posts / frames just off room walls ------------------------------
for k = 1:25
    b = blocks(solid(randi(numel(solid))), :);
    u = 0.15 + 0.7 * rand; gap = 0.08 + 0.1 * rand;
    switch randi(4)
        case 1; p = [b(1) + u * (b(2) - b(1)), b(3) - gap];
        case 2; p = [b(2) + gap, b(3) + u * (b(4) - b(3))];
        case 3; p = [b(1) + u * (b(2) - b(1)), b(4) + gap];
        case 4; p = [b(1) - gap, b(3) + u * (b(4) - b(3))];
    end
    env.addCircle(p(1), p(2), 0.03 + 0.03 * rand, 0.6 + 0.3 * rand);
end

% --- spectral-signature landmarks: on room walls (north faces) ------------
kinds = ["notch", "highpass", "lowpass", "resonant"];
pick = solid(round(linspace(1, numel(solid), 4)));
for k = 1:4
    b = blocks(pick(k), :);
    id = env.addCircle(b(1) + 0.6 * (b(2) - b(1)), b(4) + 0.12, 0.06, 0.9);
    [fAx, g] = BatEnvironment.presetSignature(kinds(k), freqMin, freqMax);
    env.setSpectralSignature(id, fAx, g);
end
end

function V = blockPolygon(b, type)
% CCW polygon of a room [x1 x2 y1 y2] with optional recessed notches and/or a
% chamfered corner - all cut INTO the room, so corridors stay clear.
C = [b(1) b(3); b(2) b(3); b(2) b(4); b(1) b(4)];
notch = false(4, 1); chamf = false(4, 1);
if type == "notch" || type == "both"; notch(randi(4)) = true; end
if type == "notch" && rand < 0.5; notch(randi(4)) = true; end
if type == "chamfer" || type == "both"; chamf(randi(4)) = true; end
V = zeros(0, 2);
for i = 1:4
    P = C(i, :); Q = C(mod(i, 4) + 1, :); R = C(mod(i - 2, 4) + 1, :);
    if chamf(i)
        c = 1.0;
        V = [V; P - c * (P - R) / norm(P - R); P + c * (Q - P) / norm(Q - P)]; %#ok<AGROW>
    else
        V = [V; P]; %#ok<AGROW>
    end
    if notch(i)
        L = norm(Q - P); t = (Q - P) / L; n = [-t(2), t(1)];
        nw = 1.0 + 0.6 * rand; d = 0.5 + 0.4 * rand; margin = 1.2;
        a = margin + rand * max(0, L - 2 * margin - nw);
        V = [V; P + a*t; P + a*t + d*n; P + (a + nw)*t + d*n; P + (a + nw)*t]; %#ok<AGROW>
    end
end
end

function polyWall(env, V, maxSeg, refl)
for k = 1:size(V, 1)
    a = V(k, :); b = V(mod(k, size(V, 1)) + 1, :);
    segWall(env, a(1), a(2), b(1), b(2), maxSeg, refl);
end
posts(env, V);
end

function segWall(env, x1, y1, x2, y2, maxSeg, refl)
L = hypot(x2 - x1, y2 - y1);
n = max(1, ceil(L / maxSeg));
t = linspace(0, 1, n + 1);
for k = 1:n
    env.addWall(x1 + t(k) * (x2 - x1), y1 + t(k) * (y2 - y1), ...
                x1 + t(k+1) * (x2 - x1), y1 + t(k+1) * (y2 - y1), refl);
end
end

function posts(env, P)
for k = 1:size(P, 1)
    env.addCircle(P(k, 1), P(k, 2), 0.06, 0.85);
end
end

function d = minSegDist(p, S)
d = inf;
for k = 1:size(S, 1)
    a = S(k, 1:2); b = S(k, 3:4); ab = b - a;
    t = max(0, min(1, dot(p - a, ab) / max(dot(ab, ab), eps)));
    d = min(d, norm(p - (a + t * ab)));
end
end
