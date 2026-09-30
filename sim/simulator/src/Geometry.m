classdef Geometry
    % Static helpers for building head/ear orientation frames and
    % converting world-frame direction vectors into azimuth/elevation
    % pairs for HRTF lookup.
    %
    % Frame convention (all right-handed, degrees in/out unless noted):
    %   World: X, Y horizontal (Z up). Body heading theta measured CCW
    %   from +X (standard unicycle convention).
    %   Local (head/ear) rest frame: local +X = forward, +Y = left,
    %   +Z = up. A direction with azimuth az and elevation el (both deg,
    %   az>0 = left, el>0 = up) has local unit vector
    %       [cosd(el)*cosd(az), cosd(el)*sind(az), sind(el)]
    %   which is the same convention used to build az_vec/el_vec in the
    %   HRTF data (grid spans [-90,90] in both, i.e. the frontal
    %   hemisphere only).

    methods (Static)
        function R = rotZ(deg)
            c = cosd(deg); s = sind(deg);
            R = [c -s 0; s c 0; 0 0 1];
        end

        function R = rotY(deg)
            % Custom sign convention so that positive deg tilts +X
            % towards +Z (see class-level convention note).
            c = cosd(deg); s = sind(deg);
            R = [c 0 -s; 0 1 0; s 0 c];
        end

        function R = orientation(az_deg, el_deg)
            % Rotation matrix mapping local (forward,left,up) coordinates
            % to world/parent coordinates, for a frame pointed at
            % azimuth az_deg and elevation el_deg relative to its parent.
            R = Geometry.rotZ(az_deg) * Geometry.rotY(el_deg);
        end

        function [az_deg, el_deg] = dirToAzEl(vec_local)
            % vec_local: [N x 3] direction vectors expressed in the local
            % (forward,left,up) frame. Returns az/el in degrees.
            x = vec_local(:,1); y = vec_local(:,2); z = vec_local(:,3);
            az_deg = atan2d(y, x);
            el_deg = atan2d(z, hypot(x, y));
        end

        function vec_local = worldToLocal(R, originWorld, pointsWorld)
            % R: 3x3 orientation matrix (local->world) of the frame.
            % originWorld: 1x3 world position of the frame's origin.
            % pointsWorld: N x 3 world positions.
            % Returns N x 3 vectors expressed in the local frame.
            d = pointsWorld - originWorld;   % N x 3, world-frame offsets
            vec_local = d * R;                % R' * d' transposed -> d * R
        end
    end
end
