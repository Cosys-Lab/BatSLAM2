function route = buildRoute()
% BUILDROUTE  Hand-designed multi-loop route through buildHallScene.
%
% Returns a struct array, one entry per route segment ("lap"), each with
%   .name       label
%   .waypoints  [K x 2] corridor-centre waypoints (m)
%   .offset     lateral offset amplitude (m) for this lap, so revisits
%               are never pixel-identical to the first pass
%
% Corridor centre-lines (see buildHallScene):
%   south y = -3.1, north y = +3.1, east x = +6.5, west x = -6.5,
%   middle gap x = 0.
%
% Loop structure (what each lap tests):
%   1 outer CCW       - builds the first template map
%   2 outer CCW       - same direction revisit: loop closures everywhere
%   3 cloverleaf      - right island CCW (same heading as laps 1-2 ->
%                       closures), then left island CW (REVERSED heading
%                       on the south/west/north corridors -> must NOT close
%                       against lap-1 templates: heading-dependent views),
%                       both passes run south through the middle gap
%   4 outer CCW       - final revisit after the cloverleaf's drift
SW = [-6.5, -3.1]; S = [0, -3.1]; SE = [6.5, -3.1];
NE = [ 6.5,  3.1]; N = [0,  3.1]; NW = [-6.5, 3.1];
C  = [0, 0];

% Intermediate points keep the spline on the corridor centre-lines and
% round the corners inside the corridor (corners cut at ~1.2 m).
outer = [ ...
    -5.3, -3.1; S; 5.3, -3.1; ...        % south, eastbound
     6.5, -1.9; 6.5, 1.9; ...            % east, northbound
     5.3,  3.1; N; -5.3, 3.1; ...        % north, westbound
    -6.5,  1.9; -6.5, -1.9 ];            % west, southbound

rightClover = [ ...
    -5.3, -3.1; -2.5, -3.1; S; 5.3, -3.1; ...   % south, eastbound
     6.5, -1.9; 6.5, 1.9; 5.3, 3.1; ...         % east, northbound
     1.2,  3.1; 0, 2.0; C; 0, -2.0 ];            % north westbound, gap southbound
leftClover = [ ...
    -1.2, -3.1; -5.3, -3.1; ...                 % south, WESTBOUND (reversed)
    -6.5, -1.9; -6.5, 1.9; -5.3, 3.1; ...       % west, NORTHBOUND (reversed)
    -1.2,  3.1; 0, 2.0; C; 0, -2.0; ...         % north EASTBOUND, gap southbound
     1.2, -3.1; 5.3, -3.1; ...                  % back onto south, eastbound
     6.5, -1.9; 6.5, 1.9; 5.3, 3.1; N; -5.3, 3.1; ...
    -6.5,  1.9; -6.5, -1.9 ];                   % one more outer arc to rejoin

route = struct('name', {}, 'waypoints', {}, 'offset', {});
route(end+1) = struct('name', "outer-1", 'waypoints', outer, 'offset', 0.00);
route(end+1) = struct('name', "outer-2", 'waypoints', outer, 'offset', 0.20);
route(end+1) = struct('name', "clover-right", 'waypoints', rightClover, 'offset', 0.15);
route(end+1) = struct('name', "clover-left", 'waypoints', leftClover, 'offset', 0.15);
route(end+1) = struct('name', "outer-3", 'waypoints', [outer; -5.3, -3.1], 'offset', 0.25);
end
