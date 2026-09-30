classdef BatEnvironment < handle
    % Point, circle (vertical-cylinder), and wall (line-segment) reflectors,
    % each optionally carrying a non-flat spectral signature on top of its
    % scalar reflectivity.
    %
    % Circles and walls don't have a single fixed reflection point - where
    % the echo actually comes from depends on which position is looking at
    % them (the head, for emission; each ear, for reception), so callers
    % that need physically-correct geometry (SonarRenderer) must call
    % nearestPoint(idx, fromPos) per leg rather than reading Positions
    % directly. Positions still holds a nominal reference position for
    % every reflector (the point itself; a circle's center; a wall's
    % segment midpoint) for consumers that only need an approximate
    % location (range-gating, ground-truth matching, plotting) and
    % shouldn't have to care about reflector type.
    %
    % Both circles and walls are treated as infinite in z (a vertical
    % cylinder / an infinite vertical plane) - a deliberate simplification
    % that keeps their geometry 2D: the reflection point's height always
    % equals the *querying* position's own height, which is what an
    % infinite pillar or wall actually does away from its own top/bottom
    % edges.

    properties
        Positions (:,3) double = zeros(0,3)      % nominal [x y z] per reflector, meters
        Reflectivity (:,1) double = zeros(0,1)    % broadband scalar gain per reflector
        Type (:,1) string = strings(0,1)          % "point" | "circle" | "wall"
        Radius (:,1) double = zeros(0,1)          % circles only; NaN otherwise
        Endpoint1 (:,3) double = zeros(0,3)        % walls only; NaN otherwise
        Endpoint2 (:,3) double = zeros(0,3)        % walls only; NaN otherwise
        SpectralSignature cell = cell(0,1)         % {} per reflector, or a struct with
                                                    % fields freqAxis, gainShape (a
                                                    % frequency-dependent multiplier on
                                                    % top of Reflectivity)
    end

    methods
        function idx = addReflector(obj, position, reflectivity)
            % Add a point reflector (backward-compatible: existing callers
            % that only ever used addReflector keep working unchanged).
            arguments
                obj
                position (1,3) double
                reflectivity (1,1) double = 1.0
            end
            idx = obj.appendCommon(position, reflectivity, "point");
            obj.Radius(idx, 1) = nan;
            obj.Endpoint1(idx, :) = nan(1, 3);
            obj.Endpoint2(idx, :) = nan(1, 3);
        end

        function idx = addCircle(obj, x, y, radius, reflectivity, z)
            arguments
                obj
                x (1,1) double
                y (1,1) double
                radius (1,1) double
                reflectivity (1,1) double = 1.0
                z (1,1) double = 0.1
            end
            idx = obj.appendCommon([x, y, z], reflectivity, "circle");
            obj.Radius(idx, 1) = radius;
            obj.Endpoint1(idx, :) = nan(1, 3);
            obj.Endpoint2(idx, :) = nan(1, 3);
        end

        function idx = addWall(obj, x1, y1, x2, y2, reflectivity, z)
            arguments
                obj
                x1 (1,1) double
                y1 (1,1) double
                x2 (1,1) double
                y2 (1,1) double
                reflectivity (1,1) double = 1.0
                z (1,1) double = 0.1
            end
            midpoint = [(x1+x2)/2, (y1+y2)/2, z];
            idx = obj.appendCommon(midpoint, reflectivity, "wall");
            obj.Radius(idx, 1) = nan;
            obj.Endpoint1(idx, :) = [x1, y1, z];
            obj.Endpoint2(idx, :) = [x2, y2, z];
        end

        function addArenaWalls(obj, xmin, xmax, ymin, ymax, reflectivity)
            % Four walls forming a closed rectangular arena boundary,
            % plus a small circle reflector at each corner.
            %
            % The corner posts aren't decoration: two finite wall
            % segments that only meet at a shared endpoint leave a real
            % gap in nearestPoint's coverage exactly at outward corners -
            % a query position diagonally beyond the corner (e.g. past
            % both xmax and ymin at once for the "southeast" corner) has
            % its perpendicular foot fall outside BOTH adjacent walls'
            % segments simultaneously, so both return NaN (no echo) and
            % the position is invisible to the sensor from every
            % direction in that wedge - a robot passing through it can
            % drive straight out of a nominally closed arena undetected
            % (found by running the navigation demo headless: the robot
            % ended up meters outside the walls). A small circle at the
            % corner is visible from any angle (nearestPoint never
            % returns NaN for a circle), which plugs the wedge - and
            % isn't just a software patch: a real right-angle wall
            % junction does produce a corner-reflector-like echo, so this
            % is also the physically closer model of the two.
            arguments
                obj
                xmin (1,1) double
                xmax (1,1) double
                ymin (1,1) double
                ymax (1,1) double
                reflectivity (1,1) double = 1.0
            end
            obj.addWall(xmin, ymin, xmax, ymin, reflectivity);
            obj.addWall(xmax, ymin, xmax, ymax, reflectivity);
            obj.addWall(xmax, ymax, xmin, ymax, reflectivity);
            obj.addWall(xmin, ymax, xmin, ymin, reflectivity);

            cornerRadius = 0.12;
            obj.addCircle(xmin, ymin, cornerRadius, reflectivity);
            obj.addCircle(xmax, ymin, cornerRadius, reflectivity);
            obj.addCircle(xmax, ymax, cornerRadius, reflectivity);
            obj.addCircle(xmin, ymax, cornerRadius, reflectivity);
        end

        function setSpectralSignature(obj, idx, freqAxis, gainShape)
            % Attach a frequency-dependent reflectivity shape (multiplies
            % the reflector's existing scalar Reflectivity) to an already-
            % added reflector. gainShape is interpolated onto whatever
            % freqBins the renderer queries with at call time; frequencies
            % outside [freqAxis(1), freqAxis(end)] clamp to the nearest
            % defined endpoint rather than extrapolating.
            arguments
                obj
                idx (1,1) double
                freqAxis (1,:) double
                gainShape (1,:) double
            end
            obj.SpectralSignature{idx} = struct('freqAxis', freqAxis(:)', 'gainShape', gainShape(:)');
        end

        function g = reflectivityAt(obj, idx, freqBins)
            % Frequency-dependent reflectivity for reflector idx, same
            % size as freqBins. Reflectors with no signature return the
            % flat scalar Reflectivity broadcast to every bin (unchanged
            % behavior from before signatures existed).
            sig = obj.SpectralSignature{idx};
            if isempty(sig)
                g = obj.Reflectivity(idx) * ones(size(freqBins));
                return;
            end
            freqCol = freqBins(:);
            shape = interp1(sig.freqAxis, sig.gainShape, freqCol, 'linear');
            shape(freqCol < sig.freqAxis(1)) = sig.gainShape(1);
            shape(freqCol > sig.freqAxis(end)) = sig.gainShape(end);
            g = obj.Reflectivity(idx) * reshape(shape, size(freqBins));
        end

        function p = nearestPoint(obj, idx, fromPos)
            % The actual reflection point for reflector idx as seen from
            % fromPos (1x3 world position). NaN row = no valid echo from
            % this position (a wall whose perpendicular foot falls outside
            % its segment).
            switch obj.Type(idx)
                case "point"
                    p = obj.Positions(idx, :);
                case "circle"
                    c = obj.Positions(idx, :);
                    r = obj.Radius(idx);
                    dx = fromPos(1) - c(1);
                    dy = fromPos(2) - c(2);
                    dHoriz = hypot(dx, dy);
                    if dHoriz < eps
                        dx = 1; dy = 0; dHoriz = 1; % degenerate: query on the axis
                    end
                    p = [c(1) + r*dx/dHoriz, c(2) + r*dy/dHoriz, fromPos(3)];
                case "wall"
                    a = obj.Endpoint1(idx, :);
                    b = obj.Endpoint2(idx, :);
                    ab = [b(1)-a(1), b(2)-a(2)];
                    ap = [fromPos(1)-a(1), fromPos(2)-a(2)];
                    abLenSq = dot(ab, ab);
                    if abLenSq < eps
                        t = 0;
                    else
                        t = dot(ap, ab) / abLenSq;
                    end
                    if t < 0 || t > 1
                        p = nan(1, 3);
                    else
                        footXY = [a(1), a(2)] + t * ab;
                        p = [footXY, fromPos(3)];
                    end
                otherwise
                    error('BatEnvironment:badType', 'unknown reflector type "%s"', obj.Type(idx));
            end
        end

        function n = numReflectors(obj)
            n = size(obj.Positions, 1);
        end

        function [collided, idx] = checkCollision(obj, fromPos, radius)
            % Whether a body of the given radius at fromPos overlaps any
            % reflector - one distance-to-nearestPoint check handles all
            % three types uniformly (nearestPoint already encodes each
            % type's own geometry), unlike bespoke per-type collision
            % formulas. idx is the first colliding reflector found, or 0.
            collided = false;
            idx = 0;
            for i = 1:obj.numReflectors()
                p = obj.nearestPoint(i, fromPos);
                if any(isnan(p))
                    continue
                end
                if norm(p - fromPos) <= radius
                    collided = true;
                    idx = i;
                    return;
                end
            end
        end

        function posOut = resolveCollision(obj, posIn, radius)
            % Push posIn (XY only - reflectors are treated as infinite in
            % Z, so there's no vertical component to resolve) out of any
            % reflector it currently penetrates, along the line from that
            % reflector's nearest point to posIn.
            %
            % This is a hard physical constraint, deliberately separate
            % from BehaviorController's CA/OA: those steer using only
            % what EchoLocalizer actually detected, and a real sensor has
            % a genuine near-field blind zone (EchoLocalizer.MinRange)
            % plus discrete pulse-to-pulse latency - so a purely reactive
            % controller cannot *guarantee* it reacts before contact, only
            % make contact rare (see docs/PLAN_arena_behaviors_ear_policy.md's
            % status note and tests/test_arena_navigation.m, which
            % measures how rare). A wall is solid regardless of whether
            % the bat noticed it in time; callers that want that physical
            % guarantee call this once per kinematic step (after
            % BatRobot.step) as a position-correction pass, the same role
            % a physics engine's collision response plays relative to an
            % AI/control layer that only approximately avoids contact.
            %
            % Iterates a few passes so a position penetrating two
            % reflectors at once (e.g. a corner's wall + corner-post)
            % resolves against both instead of just whichever is found
            % first.
            %
            % This is position-based (checks where posIn ended up, not
            % the swept path from the previous position), so it assumes
            % per-step displacement stays well under radius - true for
            % this project's kinematic step sizes, but a caller moving a
            % body fast enough to fully cross a wall's zero-thickness
            % line within one step could get pushed out the wrong side
            % (a body already past the far side of a thin barrier looks,
            % to a nearest-point-and-push-out check, like it should be
            % pushed further across); this doesn't attempt swept/
            % continuous collision detection to guard against that.
            posOut = posIn;
            for pass = 1:4
                moved = false;
                for i = 1:obj.numReflectors()
                    p = obj.nearestPoint(i, posOut);
                    if any(isnan(p))
                        continue
                    end
                    d = posOut(1:2) - p(1:2);
                    dist = norm(d);
                    if dist < radius
                        if dist < eps
                            d = [1, 0]; dist = 1; % degenerate: exactly on the point, push arbitrarily
                        end
                        posOut(1:2) = p(1:2) + d / dist * radius;
                        moved = true;
                    end
                end
                if ~moved
                    break
                end
            end
        end
    end

    methods (Access = private)
        function idx = appendCommon(obj, nominalPos, reflectivity, type)
            obj.Positions(end+1, :) = nominalPos;
            obj.Reflectivity(end+1, 1) = reflectivity;
            obj.Type(end+1, 1) = type;
            obj.SpectralSignature{end+1, 1} = [];
            idx = obj.numReflectors();
        end
    end

    methods (Static)
        function [freqAxis, gainShape] = presetSignature(kind, freqMin, freqMax)
            % A couple of illustrative non-flat spectral signatures for
            % target-identification testing later. Not physically derived
            % from any real material/species - just distinctive shapes.
            arguments
                kind (1,1) string
                freqMin (1,1) double
                freqMax (1,1) double
            end
            freqAxis = linspace(freqMin, freqMax, 64);
            switch kind
                case "notch"
                    f0 = freqMin + 0.5*(freqMax-freqMin);
                    bw = 0.08*(freqMax-freqMin);
                    gainShape = 1 - 0.85*exp(-0.5*((freqAxis-f0)/bw).^2);
                case "highpass"
                    gainShape = (freqAxis - freqMin) / (freqMax - freqMin);
                    gainShape = 0.15 + 0.85*gainShape;
                case "lowpass"
                    gainShape = (freqMax - freqAxis) / (freqMax - freqMin);
                    gainShape = 0.15 + 0.85*gainShape;
                case "resonant"
                    f0 = freqMin + 0.35*(freqMax-freqMin);
                    bw = 0.06*(freqMax-freqMin);
                    gainShape = 0.3 + 0.9*exp(-0.5*((freqAxis-f0)/bw).^2);
                otherwise
                    error('BatEnvironment:badPreset', 'unknown preset "%s"', kind);
            end
        end
    end
end
