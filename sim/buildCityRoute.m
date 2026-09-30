function route = buildCityRoute(info, seed, targetLength, legLength, cornerR, wobble)
% BUILDCITYROUTE  Long random tour through buildCityScene's street grid
% and hall lanes, as a followPath-compatible route (struct array of legs).
%
% A random walk on the street/lane graph: no U-turns, and a preference for
% edges driven less often (weight 1/(1+visits)^2), so the tour covers the
% whole map, revisits places after long excursions, and drives most
% streets in both directions. Corners are rounded inside the intersection
% (waypoints at +-cornerR along the incoming/outgoing street) so the
% Catmull-Rom path stays on the street.
%
% legLength only chops the tour into labelled legs ("leg-01", ...), each
% with its own lateral wobble amplitude - it is one continuous drive.
arguments
    info struct
    seed (1,1) double = 3
    targetLength (1,1) double = 1000
    legLength (1,1) double = 120
    cornerR (1,1) double = 1.3            % corner rounding (m along each street)
    wobble (1,2) double = [0.05, 0.25]    % per-leg lateral wobble amplitude range (m)
end
rng(seed);

% --- graph --------------------------------------------------------------
[GX, GY] = meshgrid(info.streetX, info.streetY);
nodes = [GX(:), GY(:)];
nx = numel(info.streetX); ny = numel(info.streetY);
gid = @(ix, iy) (ix - 1) * ny + iy;       % meshgrid(:) is column-major over x
edges = zeros(0, 2);
for ix = 1:nx
    for iy = 1:ny
        if ix < nx; edges(end+1, :) = [gid(ix, iy), gid(ix+1, iy)]; end %#ok<AGROW>
        if iy < ny; edges(end+1, :) = [gid(ix, iy), gid(ix, iy+1)]; end %#ok<AGROW>
    end
end
off = size(nodes, 1);
nodes = [nodes; info.hallNodes];
edges = [edges; info.hallEdges + off];
for k = 1:size(info.doorNodes, 1)
    [~, g] = min(vecnorm(nodes(1:off, :) - info.doorNodes(k, :), 2, 2));
    edges(end+1, :) = [g, info.doorHallIdx(k) + off]; %#ok<AGROW>
end
N = size(nodes, 1);
A = false(N); E = zeros(N);                % adjacency, edge index
for e = 1:size(edges, 1)
    A(edges(e,1), edges(e,2)) = true; A(edges(e,2), edges(e,1)) = true;
    E(edges(e,1), edges(e,2)) = e;    E(edges(e,2), edges(e,1)) = e;
end
visits = zeros(size(edges, 1), 1);

% --- random walk ----------------------------------------------------------
walk = [gid(1, 1); gid(2, 1)];             % start SW corner, heading east
visits(E(walk(1), walk(2))) = 1;
len = norm(nodes(walk(2), :) - nodes(walk(1), :));
while len < targetLength
    cur = walk(end); prev = walk(end-1);
    nb = find(A(cur, :));
    nb(nb == prev) = [];                   % no U-turns
    wgt = 1 ./ (1 + visits(E(cur, nb))).^2;
    nxt = nb(find(rand <= cumsum(wgt) / sum(wgt), 1));
    visits(E(cur, nxt)) = visits(E(cur, nxt)) + 1;
    len = len + norm(nodes(nxt, :) - nodes(cur, :));
    walk(end+1) = nxt; %#ok<AGROW>
end

% --- waypoints with rounded corners, chopped into legs --------------------
route = struct('name', {}, 'waypoints', {}, 'offset', {});
wp = nodes(walk(1), :); acc = 0; legNo = 1;
for k = 2:numel(walk) - 1
    c = nodes(walk(k), :);
    dIn = c - nodes(walk(k-1), :); dIn = dIn / norm(dIn);
    dOut = nodes(walk(k+1), :) - c; dOut = dOut / norm(dOut);
    if dot(dIn, dOut) > 0.99
        wp(end+1, :) = c; %#ok<AGROW>
    else
        wp(end+1, :) = c - cornerR * dIn; %#ok<AGROW>
        wp(end+1, :) = c + cornerR * dOut; %#ok<AGROW>
    end
    acc = acc + norm(nodes(walk(k), :) - nodes(walk(k-1), :));
    if acc >= legLength
        route(end+1) = struct('name', sprintf("leg-%02d", legNo), 'waypoints', wp, ...
                              'offset', wobble(1) + diff(wobble) * rand); %#ok<AGROW>
        wp = zeros(0, 2); acc = 0; legNo = legNo + 1;
    end
end
wp(end+1, :) = nodes(walk(end), :);
route(end+1) = struct('name', sprintf("leg-%02d", legNo), 'waypoints', wp, 'offset', wobble(1) + diff(wobble) * rand);
fprintf('city route: %d graph steps, %.0f m nominal, %d legs, edges covered %d/%d (%.0f%% driven twice+)\n', ...
    numel(walk) - 1, len, numel(route), nnz(visits), numel(visits), 100 * mean(visits >= 2));
end
