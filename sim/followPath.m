function traj = followPath(route, ds, startPose)
% FOLLOWPATH  Sample a route at constant arc-length spacing ds - the robot
% is placed ON the path directly (no controller in the loop), heading
% along the path tangent.
%
% route      struct array from buildRoute (segments are concatenated)
% ds         arc-length spacing between pulses (m)
% startPose  optional [x y] to prepend (defaults to first waypoint)
%
% traj fields:
%   pose   [N x 3] x, y, theta (rad, CCW from +x, unwrapped)
%   lap    [N x 1] route segment index
%   s      [N x 1] arc length (m)
arguments
    route struct
    ds (1,1) double = 0.15
    startPose double = []
end

% Concatenate all segments into one waypoint chain, remembering which
% segment each waypoint came from and that segment's lateral offset.
P = zeros(0, 2); lapOfPt = zeros(0, 1); offOfPt = zeros(0, 1);
if ~isempty(startPose)
    P = startPose(1:2); lapOfPt = 1; offOfPt = route(1).offset;
end
for k = 1:numel(route)
    w = route(k).waypoints;
    P = [P; w]; %#ok<AGROW>
    lapOfPt = [lapOfPt; k * ones(size(w, 1), 1)]; %#ok<AGROW>
    offOfPt = [offOfPt; route(k).offset * ones(size(w, 1), 1)]; %#ok<AGROW>
end
% Drop exact duplicates (consecutive) - they break Catmull-Rom.
keep = [true; vecnorm(diff(P), 2, 2) > 1e-6];
P = P(keep, :); lapOfPt = lapOfPt(keep); offOfPt = offOfPt(keep);

% Dense centripetal Catmull-Rom through all waypoints.
dense = zeros(0, 2); denseLap = zeros(0, 1); denseOff = zeros(0, 1);
Pext = [2*P(1,:) - P(2,:); P; 2*P(end,:) - P(end-1,:)];
for i = 1:size(P, 1) - 1
    p0 = Pext(i, :); p1 = Pext(i+1, :); p2 = Pext(i+2, :); p3 = Pext(i+3, :);
    seg = catmullRom(p0, p1, p2, p3, 60);
    dense = [dense; seg(1:end-1, :)]; %#ok<AGROW>
    denseLap = [denseLap; lapOfPt(i+1) * ones(size(seg,1)-1, 1)]; %#ok<AGROW>
    a = linspace(offOfPt(i), offOfPt(i+1), size(seg,1))';
    denseOff = [denseOff; a(1:end-1)]; %#ok<AGROW>
end
dense = [dense; P(end, :)]; denseLap = [denseLap; lapOfPt(end)]; denseOff = [denseOff; offOfPt(end)];

% Arc-length parametrisation, resample every ds.
sDense = [0; cumsum(vecnorm(diff(dense), 2, 2))];
sQ = (0:ds:sDense(end))';
xy = interp1(sDense, dense, sQ, 'linear');
lap = interp1(sDense, denseLap, sQ, 'previous');
amp = interp1(sDense, denseOff, sQ, 'linear');

% Tangent heading (central differences on the dense curve).
tang = gradient(dense')';
hDense = unwrap(atan2(tang(:,2), tang(:,1)));
theta = interp1(sDense, hDense, sQ, 'linear');

% Lap-dependent lateral wobble: a slow sinusoid along the path normal, so
% each revisit passes the landmarks at a slightly different distance.
normal = [-sin(theta), cos(theta)];
wobble = amp .* sin(2*pi*sQ / 9.0);
xy = xy + wobble .* normal;
% Heading follows the wobbled path's own tangent.
dxy = gradient(xy')';
theta = unwrap(atan2(dxy(:,2), dxy(:,1)));

traj.pose = [xy, theta];
traj.lap = lap;
traj.s = sQ;
end

function C = catmullRom(p0, p1, p2, p3, n)
% Centripetal (alpha = 0.5) Catmull-Rom segment from p1 to p2.
alpha = 0.5;
t0 = 0;
t1 = t0 + norm(p1 - p0)^alpha;
t2 = t1 + norm(p2 - p1)^alpha;
t3 = t2 + norm(p3 - p2)^alpha;
t = linspace(t1, t2, n)';
A1 = (t1 - t)/(t1 - t0) .* p0 + (t - t0)/(t1 - t0) .* p1;
A2 = (t2 - t)/(t2 - t1) .* p1 + (t - t1)/(t2 - t1) .* p2;
A3 = (t3 - t)/(t3 - t2) .* p2 + (t - t2)/(t3 - t2) .* p3;
B1 = (t2 - t)/(t2 - t0) .* A1 + (t - t0)/(t2 - t0) .* A2;
B2 = (t3 - t)/(t3 - t1) .* A2 + (t - t1)/(t3 - t1) .* A3;
C  = (t2 - t)/(t2 - t1) .* B1 + (t - t1)/(t2 - t1) .* B2;
end
