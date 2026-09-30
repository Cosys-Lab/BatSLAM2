classdef BatRobot < handle
    % 2D unicycle chassis (x, y, theta) carrying a head that can pan/tilt
    % relative to the body, and two ears that can each pan/tilt relative
    % to the head (independent pinna orientation). Reflectors live in 3D,
    % so the head/ears are placed at a fixed height above the ground
    % plane.

    properties
        X (1,1) double = 0
        Y (1,1) double = 0
        Theta (1,1) double = 0      % body heading [deg], CCW from +X
        Radius (1,1) double = 0.15  % [m] physical body radius, for collision checks

        HeadHeight (1,1) double = 0.05   % [m] head height above ground plane
        HeadAz (1,1) double = 0          % [deg] head yaw offset from body heading
        HeadEl (1,1) double = 0          % [deg] head pitch (positive = up)

        EarBaseline (1,1) double = 0.010 % [m] interaural distance (ear-to-ear)
        EarAz struct = struct('left', 0, 'right', 0)  % [deg] pinna yaw offset from head
        EarEl struct = struct('left', 0, 'right', 0)  % [deg] pinna pitch offset from head
    end

    methods
        function step(obj, vlin, wrot, dt)
            % Integrate unicycle kinematics forward by dt.
            obj.X = obj.X + vlin * cosd(obj.Theta) * dt;
            obj.Y = obj.Y + vlin * sind(obj.Theta) * dt;
            obj.Theta = obj.Theta + wrot * dt;
        end

        function setHeadPose(obj, az_deg, el_deg)
            obj.HeadAz = az_deg;
            obj.HeadEl = el_deg;
        end

        function setEarPose(obj, ear, az_deg, el_deg)
            ear = lower(ear);
            obj.EarAz.(ear) = az_deg;
            obj.EarEl.(ear) = el_deg;
        end

        function p = headPosition(obj)
            p = [obj.X, obj.Y, obj.HeadHeight];
        end

        function R = headOrientation(obj)
            R = Geometry.orientation(obj.Theta + obj.HeadAz, obj.HeadEl);
        end

        function R = earOrientation(obj, ear)
            ear = lower(ear);
            Rhead = obj.headOrientation();
            R = Rhead * Geometry.orientation(obj.EarAz.(ear), obj.EarEl.(ear));
        end

        function p = earPosition(obj, ear)
            ear = lower(ear);
            side = 1;
            if strcmp(ear, 'right'); side = -1; end
            Rhead = obj.headOrientation();
            offsetLocal = [0; side * obj.EarBaseline / 2; 0];
            p = obj.headPosition() + (Rhead * offsetLocal)';
        end
    end
end
