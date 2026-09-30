classdef HRTFModel
    % Loads a per-species bat_hrtf .mat file and exposes interpolated
    % emitter/receiver directional gain lookups plus call metadata.
    %
    % Expected fields in the file's `bat_hrtf` struct:
    %   hrtf_left, hrtf_right, hrtf_em : [Naz x Nel x Nfreq] linear gain
    %   az_vec, el_vec                 : [deg], frontal hemisphere only
    %   freq_vec                       : [Hz]
    %   (optional) call_spec, fmin, fmax, fs

    properties
        Name
        AzVec (1,:) double
        ElVec (1,:) double
        FreqVec (1,:) double
        Fs (1,1) double
        CallSpec double = []
        Fmin (1,1) double
        Fmax (1,1) double

        % A real reflector doesn't vanish the instant it crosses the last
        % measured angle - the measured directivity pattern just runs
        % out of data there, while the actual gain keeps fading, usually
        % faster than the in-grid trend alone would suggest. Rather than
        % the hard cliff to exactly 0 gain at +-max(AzVec)/+-max(ElVec)
        % this used to be, gain now linearly extrapolates the boundary
        % trend for up to ExtrapMarginDeg past the grid edge, then drops
        % to 0 beyond that. Frequency is never extrapolated this way - a
        % spectral gain curve can do far stranger things just past a
        % measured band's edge than a directivity pattern does just past
        % a measured angle's edge, so there's no similarly defensible
        % "keep extrapolating a bit" default for that dimension.
        ExtrapMarginDeg (1,1) double = 10  % deg

        % Raw [Naz x Nel x Nfreq] linear-gain grids, exposed (read-only in
        % spirit) for callers that need direct template access, e.g.
        % EchoLocalizer's grid-based direction matching, rather than the
        % single-point interpolated queries below.
        HrtfLeft double
        HrtfRight double
        HrtfEm double
    end

    properties (Access = private)
        GIem
        GIleft
        GIright
    end

    methods
        function obj = HRTFModel(matFile)
            s = load(matFile);
            bh = s.bat_hrtf;

            obj.AzVec = bh.az_vec(:)';
            obj.ElVec = bh.el_vec(:)';
            obj.FreqVec = bh.freq_vec(:)';

            % 'linear' extrapolation (not 'none') so a query just past the
            % grid edge continues the boundary trend instead of dropping
            % straight to NaN/0 - this applies to ALL three dimensions at
            % the interpolant level, so queryGain below explicitly re-
            % zeros anything past ExtrapMarginDeg in az/el, and anything
            % outside the measured band at all in frequency (frequency
            % was never meant to extrapolate - see ExtrapMarginDeg's doc).
            grid = {obj.AzVec, obj.ElVec, obj.FreqVec};
            obj.GIem    = griddedInterpolant(grid, bh.hrtf_em,    'linear', 'linear');
            obj.GIleft  = griddedInterpolant(grid, bh.hrtf_left,  'linear', 'linear');
            obj.GIright = griddedInterpolant(grid, bh.hrtf_right, 'linear', 'linear');

            obj.HrtfEm = bh.hrtf_em;
            obj.HrtfLeft = bh.hrtf_left;
            obj.HrtfRight = bh.hrtf_right;

            if isfield(bh, 'call_spec'); obj.CallSpec = bh.call_spec(:); end
            if isfield(bh, 'fmin'); obj.Fmin = bh.fmin; else; obj.Fmin = obj.FreqVec(1); end
            if isfield(bh, 'fmax'); obj.Fmax = bh.fmax; else; obj.Fmax = obj.FreqVec(end); end
            if isfield(bh, 'fs'); obj.Fs = bh.fs; else; obj.Fs = 2.2 * obj.FreqVec(end); end

            [~, obj.Name] = fileparts(matFile);
        end

        function g = emitterGain(obj, az_deg, el_deg, freq_hz)
            % az_deg, el_deg: scalars. freq_hz: vector. Returns gain
            % vector same size as freq_hz; frequencies outside the
            % measured band return 0, as do directions beyond
            % ExtrapMarginDeg past the measured az/el grid edge - within
            % that margin, gain is linearly extrapolated rather than cut
            % off (see ExtrapMarginDeg's doc).
            g = obj.queryGain(obj.GIem, az_deg, el_deg, freq_hz);
        end

        function g = receiverGain(obj, ear, az_deg, el_deg, freq_hz)
            switch lower(ear)
                case {'l', 'left'}
                    gi = obj.GIleft;
                case {'r', 'right'}
                    gi = obj.GIright;
                otherwise
                    error('HRTFModel:badEar', 'ear must be ''left'' or ''right''');
            end
            g = obj.queryGain(gi, az_deg, el_deg, freq_hz);
        end
    end

    methods (Access = private)
        function g = queryGain(obj, gi, az_deg, el_deg, freq_hz)
            freq_hz = freq_hz(:);
            n = numel(freq_hz);
            azq = repmat(az_deg, n, 1);
            elq = repmat(el_deg, n, 1);
            g = gi(azq, elq, freq_hz);
            g(isnan(g)) = 0;
            g = max(g, 0); % linear extrapolation of a falling gain can go negative - not physical

            azLimit = max(abs(obj.AzVec)) + obj.ExtrapMarginDeg;
            elLimit = max(abs(obj.ElVec)) + obj.ExtrapMarginDeg;
            if abs(az_deg) > azLimit || abs(el_deg) > elLimit
                g(:) = 0; % beyond the extrapolation margin entirely - a reflector back here is unseeable
            end
            g(freq_hz < obj.FreqVec(1) | freq_hz > obj.FreqVec(end)) = 0; % frequency never extrapolates
        end
    end
end
