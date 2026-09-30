function [env, info] = buildCityScene(freqMin, freqMax, seed, variety, roughness)
% BUILDCITYSCENE  Large city-block world for map-collapse tests.
%
%   west part: a 3 x 3 grid of blocks separated by streets (16
%     intersections); east part: an open hall with a random pillar forest,
%     reachable through two doorways, with a lane network kept clear.
%
%   variety = 0  worst case: IDENTICAL 6 x 4 m blocks, uniform 3.5 m
%                streets, sparse clutter, and the same 3-pillar alias
%                pattern at 4 intersections. Turned out to be beyond what
%                an appearance-based matcher can disambiguate.
%   variety = 1  (default) still a grid, but no longer a copy-paste one:
%                irregular block sizes (5-7 m) and street widths (3-4 m),
%                blocks with recessed notches and chamfered corners, one
%                block replaced by an open plaza with a pillar cluster,
%                ~2.5x the clutter plus street poles, alias pattern at only
%                2 intersections.
%
%   roughness = 0 (default): walls are purely specular - a wall returns a
%                single echo from its perpendicular foot point, identical
%                all along a straight street. roughness > 0 adds diffuse
%                scattering: weak point scatterers every 10 cm along every
%                wall (reflectivity ~ roughness x lognormal), i.e. surface
%                texture that gives a place-specific echo train as the beam
%                sweeps the wall. Added last, on its own random stream, so
%                the rest of the scene is identical with and without it.
%
% All walls are built from segments <= 1.5 m: SonarRenderer range-gates
% walls on their MIDPOINT distance, so a long wall would silently vanish
% from echoes when the robot is near one of its ends.
%
% info (used by buildCityRoute and the export):
%   .hall        [xmin xmax ymin ymax] outer boundary
%   .islands     [K x 4] blocks
%   .streetX, .streetY   street centre-lines of the grid
%   .hallNodes   [M x 2] hall lane nodes, .hallEdges [E x 2] (indices)
%   .doorNodes   [2 x 2] grid-side door approach points (on street x = streetX(end))
%   .aliasTrap   [P x 2] trap pillar positions
arguments
    freqMin (1,1) double = 25e3
    freqMax (1,1) double = 95e3
    seed (1,1) double = 11
    variety (1,1) double = 1
    roughness (1,1) double = 0
end

rng(seed);
env = BatEnvironment();
info = struct();
maxSeg = 1.5;
wallRefl = 0.85;

if variety == 0
    wX = 3.5 * ones(1, 4); wY = wX;        % street widths
    bw = [6 6 6]; bh = [4 4 4];            % block widths / heights
else
    wX = [3.5, 3.0, 4.0, 3.5]; wY = [3.2, 3.8, 3.0, 3.6];
    bw = [5.0, 7.0, 6.0];      bh = [4.5, 3.5, 4.2];
end
streetX = zeros(1, 4); streetY = zeros(1, 4);
streetX(1) = wX(1)/2; streetY(1) = wY(1)/2;
for k = 1:3
    streetX(k+1) = streetX(k) + wX(k)/2 + bw(k) + wX(k+1)/2;
    streetY(k+1) = streetY(k) + wY(k)/2 + bh(k) + wY(k+1)/2;
end
gridX = [0, streetX(end) + wX(end)/2];
gridY = [0, streetY(end) + wY(end)/2];
hallX = [gridX(2), gridX(2) + 14];         % 32 .. 46
bound = [0, hallX(2), 0, gridY(2)];
info.hall = bound;
info.streetX = streetX; info.streetY = streetY;

% --- outer boundary -------------------------------------------------------
segWall(env, bound(1), bound(3), bound(2), bound(3), maxSeg, wallRefl);
segWall(env, bound(2), bound(3), bound(2), bound(4), maxSeg, wallRefl);
segWall(env, bound(2), bound(4), bound(1), bound(4), maxSeg, wallRefl);
segWall(env, bound(1), bound(4), bound(1), bound(3), maxSeg, wallRefl);
posts(env, [bound(1) bound(3); bound(2) bound(3); bound(2) bound(4); bound(1) bound(4)]);

% --- blocks ----------------------------------------------------------------
if variety == 0
    types = repmat("rect", 1, 9);
else
    types = ["plaza", "notch", "notch", "notch", "chamfer", "chamfer", "both", "rect", "rect"];
    types = types(randperm(9));
end
blocks = zeros(0, 4); plazaPillars = zeros(0, 2);
bi = 0;
for i = 1:3
    for j = 1:3
        bi = bi + 1;
        b = [streetX(i) + wX(i)/2, streetX(i+1) - wX(i+1)/2, streetY(j) + wY(j)/2, streetY(j+1) - wY(j+1)/2];
        blocks(end+1, :) = b; %#ok<AGROW>
        if types(bi) == "plaza"
            % open square: a loose pillar cluster instead of a building
            n = 0;
            while n < 5
                p = [b(1) + 1.0 + rand * (b(2) - b(1) - 2.0), b(3) + 1.0 + rand * (b(4) - b(3) - 2.0)];
                if n > 0 && min(vecnorm(plazaPillars - p, 2, 2)) < 0.9; continue; end
                n = n + 1; plazaPillars(end+1, :) = p; %#ok<AGROW>
                env.addCircle(p(1), p(2), 0.1 + 0.15 * rand, 0.8);
            end
            continue
        end
        polyWall(env, blockPolygon(b, types(bi)), maxSeg, wallRefl);
    end
end
info.islands = blocks;
info.blockTypes = types;

% --- divider wall between grid and hall, two doorways ---------------------
xd = gridX(2);
doorY = [streetY(2), streetY(3)];          % doorways centred on these streets
dw = [wY(2), wY(3)] / 2;
edges = [bound(3), doorY(1) - dw(1); doorY(1) + dw(1), doorY(2) - dw(2); doorY(2) + dw(2), bound(4)];
for k = 1:size(edges, 1)
    segWall(env, xd, edges(k,1), xd, edges(k,2), maxSeg, wallRefl);
    posts(env, [xd edges(k,1); xd edges(k,2)]);
end

% --- hall lane network (kept clear of pillars) ----------------------------
laneX = [37, 43];
laneY = [3, doorY(1), doorY(2), 23];
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
nPillars = 45; placed = zeros(0, 3); tries = 0;
while size(placed, 1) < nPillars && tries < 20000
    tries = tries + 1;
    p = [hallX(1) + 1 + rand * (hallX(2) - hallX(1) - 2), bound(3) + 1 + rand * (bound(4) - bound(3) - 2)];
    r = 0.08 + 0.22 * rand;
    if minSegDist(p, laneSegs) < 1.4 + r; continue; end
    if ~isempty(placed) && min(vecnorm(placed(:,1:2) - p, 2, 2) - placed(:,3)) < 0.9 + r; continue; end
    placed(end+1, :) = [p, r]; %#ok<AGROW>
end
for k = 1:size(placed, 1)
    env.addCircle(placed(k,1), placed(k,2), placed(k,3), 0.6 + 0.3 * rand);
end

% --- alias trap: same pillar pattern at 4 intersections -------------------
% pattern sits in the NE corner area of an intersection (off the path)
pattern = [1.50, 1.50, 0.14; 1.55, 0.95, 0.08; 0.95, 1.55, 0.10];
if variety == 0
    trapIdx = [1 2; 2 3; 3 2; 2 1];        % [ix iy] of the intersections
else
    trapIdx = [2 2; 3 3];                  % two interior intersections
end
trap = zeros(0, 2);
for k = 1:size(trapIdx, 1)
    ix = trapIdx(k, 1); iy = trapIdx(k, 2);
    half = [wX(ix), wY(iy)] / 2;           % pattern scaled to this intersection
    for m = 1:size(pattern, 1)
        c = [streetX(ix), streetY(iy)] + pattern(m, 1:2) .* half / 1.75;
        env.addCircle(c(1), c(2), pattern(m, 3), 0.85);
        trap(end+1, :) = c; %#ok<AGROW>
    end
end
info.aliasTrap = trap;

% --- sparse clutter on block faces and boundary walls ---------------------
% Deliberately sparse: this is what makes streets distinguishable at all.
nClutter = 70 + 100 * (variety > 0); k = 0; tries = 0;
while k < nClutter && tries < 50000
    tries = tries + 1;
    p = [bound(1) + 0.3 + rand * (gridX(2) - 0.6), bound(3) + 0.3 + rand * (bound(4) - 0.6)];
    % must lie 0.15..0.5 m from a block face or the boundary, never in a street centre band
    dFace = inf;
    for b = 1:size(blocks, 1)
        B = blocks(b, :);
        inside = p(1) > B(1) && p(1) < B(2) && p(2) > B(3) && p(2) < B(4);
        if inside; dFace = -1; break; end
        dx = max([B(1) - p(1), 0, p(1) - B(2)]); dy = max([B(3) - p(2), 0, p(2) - B(4)]);
        dFace = min(dFace, hypot(dx, dy));
    end
    if dFace < 0; continue; end
    dFace = min([dFace, p(1) - bound(1), gridX(2) - p(1), p(2) - bound(3), bound(4) - p(2)]);
    if dFace < 0.15 || dFace > 0.5; continue; end
    k = k + 1;
    env.addReflector([p, 0.02 + 0.12 * rand], 0.35 + 0.35 * rand);
end

% --- street poles (variety only): thin posts just off block faces ---------
if variety > 0
    solid = find(types ~= "plaza");
    for k = 1:25
        b = blocks(solid(randi(numel(solid))), :);
        u = 0.15 + 0.7 * rand; gap = 0.3 + 0.15 * rand;
        switch randi(4)
            case 1; p = [b(1) + u * (b(2) - b(1)), b(3) - gap];
            case 2; p = [b(2) + gap, b(3) + u * (b(4) - b(3))];
            case 3; p = [b(1) + u * (b(2) - b(1)), b(4) + gap];
            case 4; p = [b(1) - gap, b(3) + u * (b(4) - b(3))];
        end
        env.addCircle(p(1), p(2), 0.04 + 0.05 * rand, 0.6 + 0.3 * rand);
    end
end

% --- a few spectral-signature landmarks -----------------------------------
sigs = {[streetX(1) - 1.4, streetY(3) + 2.0], "notch"; ...
        [streetX(3) + 1.4, streetY(1) + 2.5], "highpass"; ...
        [39.5, 20.0], "lowpass"; ...
        [streetX(4) - 1.4, streetY(4) - 2.8], "resonant"};
for k = 1:size(sigs, 1)
    p = sigs{k, 1};
    id = env.addCircle(p(1), p(2), 0.15, 0.9);
    [fAx, g] = BatEnvironment.presetSignature(sigs{k, 2}, freqMin, freqMax);
    env.setSpectralSignature(id, fAx, g);
end

% --- diffuse wall scattering (optional) -----------------------------------
if roughness > 0
    stream = RandStream('mt19937ar', 'Seed', seed + 1000);
    walls = find(env.Type == "wall");
    spacing = 0.10;
    nAdded = 0;
    for w = walls'
        a = env.Endpoint1(w, 1:2); b = env.Endpoint2(w, 1:2);
        L = norm(b - a); t = (b - a) / L; n = [-t(2), t(1)];
        m = max(1, round(L / spacing));
        for k = 1:m
            u = (k - 0.5 + 0.3 * (rand(stream) - 0.5)) / m;
            p = a + u * (b - a) + 0.03 * (rand(stream) - 0.5) * n;
            refl = roughness * exp(0.8 * randn(stream));
            env.addReflector([p, 0.05 + 0.1 * (rand(stream) - 0.5)], refl);
            nAdded = nAdded + 1;
        end
    end
    info.nDiffuse = nAdded;
end
end

function V = blockPolygon(b, type)
% CCW polygon of a block [x1 x2 y1 y2], optionally with recessed notches
% and/or a chamfered corner - all cut INTO the block, so streets stay clear.
C = [b(1) b(3); b(2) b(3); b(2) b(4); b(1) b(4)];
notch = false(4, 1); chamf = false(4, 1);
if type == "notch" || type == "both"; notch(randi(4)) = true; end
if type == "notch" && rand < 0.5; notch(randi(4)) = true; end
if type == "chamfer" || type == "both"; chamf(randi(4)) = true; end
V = zeros(0, 2);
for i = 1:4
    P = C(i, :); Q = C(mod(i, 4) + 1, :); R = C(mod(i - 2, 4) + 1, :);
    if chamf(i)
        c = 1.2;
        V = [V; P - c * (P - R) / norm(P - R); P + c * (Q - P) / norm(Q - P)]; %#ok<AGROW>
    else
        V = [V; P]; %#ok<AGROW>
    end
    if notch(i)
        L = norm(Q - P); t = (Q - P) / L; n = [-t(2), t(1)];
        nw = 1.2 + 0.6 * rand; d = 0.6 + 0.4 * rand; margin = 1.4;
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
    env.addCircle(P(k, 1), P(k, 2), 0.12, 0.85);
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
