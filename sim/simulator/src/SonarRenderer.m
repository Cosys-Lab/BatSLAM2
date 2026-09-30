classdef SonarRenderer < handle
    % Renders the binaural time-domain echo signals and spectrograms a
    % BatRobot would perceive from a BatEnvironment via a given
    % HRTFModel, for the robot's current pose.
    %
    % Pipeline per call: synthesize an FM sweep call (shaped by the
    % species' call_spec if provided), then for every reflector and each
    % ear multiply the call spectrum by
    %   emitter directivity x spherical spreading x atmospheric
    %   absorption x reflectivity x receiver directivity
    % (all frequency-dependent - reflectivity via BatEnvironment.
    % reflectivityAt, which is flat unless the reflector carries a
    % SpectralSignature), inverse-transform to a short echo waveform, and
    % add it into that ear's receive buffer at the round-trip delay
    % sample, then add optional white sensor noise (NoiseAmplitude). The
    % two buffers are then turned into spectrograms via a manual
    % (toolbox-free) STFT.
    %
    % Point, circle, and wall reflectors (BatEnvironment.Type) don't all
    % have a single fixed position: circles/walls' actual reflection point
    % depends on who's looking, so it's queried fresh per leg (head, then
    % each ear) via BatEnvironment.nearestPoint - a wall returns no echo
    % from a position whose perpendicular foot falls outside its segment.

    properties
        SpeedOfSound (1,1) double = 343      % m/s
        TemperatureC (1,1) double = 20        % deg C, for air absorption
        HumidityPct (1,1) double = 50         % relative humidity, %
        PressureKPa (1,1) double = 101.325    % ambient pressure, kPa
        CallDuration (1,1) double = 0.003     % s, emitted FM sweep duration

        WindowLength (1,1) double = 256       % samples, STFT window
        OverlapFraction (1,1) double = 0.75   % STFT window overlap

        MaxRange (1,1) double = Inf           % m, one-way cutoff; reflectors
                                               % beyond this are skipped and
                                               % excluded from buffer sizing

        NoiseAmplitude (1,1) double = 0       % linear std of additive white
                                               % sensor noise on each ear's
                                               % received signal (call peak
                                               % amplitude is ~1, so this is
                                               % directly comparable to echo
                                               % amplitudes); 0 = noiseless
    end

    methods
        function obj = SonarRenderer(varargin)
            for k = 1:2:numel(varargin)
                obj.(varargin{k}) = varargin{k+1};
            end
        end

        function [callWave, Fs] = synthesizeCall(obj, hrtf)
            % Linear FM downsweep from hrtf.Fmax to hrtf.Fmin, magnitude-
            % shaped to hrtf.CallSpec when the species file provides one.
            Fs = hrtf.Fs;
            T = obj.CallDuration;
            N = round(T * Fs);
            t = (0:N-1)' / Fs;
            f1 = hrtf.Fmax; f2 = hrtf.Fmin;
            phase = 2*pi*(f1*t + (f2-f1)*t.^2/(2*T));
            x = sin(phase);

            if ~isempty(hrtf.CallSpec)
                Nfft = N;
                X = fft(x, Nfft);
                halfN = floor(Nfft/2) + 1;
                fbins = (0:halfN-1)' * Fs / Nfft;
                targetMag = interp1(hrtf.FreqVec, hrtf.CallSpec, fbins, 'linear', 0);
                mag = abs(X(1:halfN));
                mag(mag < eps) = eps;
                X(1:halfN) = X(1:halfN) .* (targetMag ./ mag);
                X(halfN+1:end) = SonarRenderer.mirror(X(1:halfN), Nfft);
                x = real(ifft(X));
            end

            rampN = max(1, round(0.05 * N));
            ramp = 0.5 - 0.5 * cos(pi * (0:rampN-1)' / rampN);
            env = ones(N, 1);
            env(1:rampN) = ramp;
            env(end-rampN+1:end) = flipud(ramp);
            callWave = x .* env;
        end

        function [sigL, sigR, Fs] = renderEcho(obj, robot, env, hrtf)
            [callWave, Fs] = obj.synthesizeCall(hrtf);
            Ncall = numel(callWave);
            Nfft = 2^nextpow2(2 * Ncall);
            Scall = fft(callWave, Nfft);
            halfN = floor(Nfft/2) + 1;
            freqBins = (0:halfN-1)' * Fs / Nfft;

            c = obj.SpeedOfSound;
            headPos = robot.headPosition();
            Rhead = robot.headOrientation();

            % Coarse pre-filter on nominal position (head to reflector
            % center/midpoint), corrected for circle radius so a circle
            % whose near edge is in range isn't wrongly excluded by its
            % center distance alone. Walls use midpoint distance
            % uncorrected - a minor, accepted imprecision right at the
            % MaxRange boundary for long/oblique walls.
            distHead = vecnorm(env.Positions - headPos, 2, 2) - max(env.Radius, 0);
            inRange = distHead <= obj.MaxRange;
            nRef = sum(inRange);
            if nRef == 0
                sigL = obj.NoiseAmplitude * randn(Ncall, 1);
                sigR = obj.NoiseAmplitude * randn(Ncall, 1);
                return;
            end
            refIdx = find(inRange);

            % Per-ear round trip distance can exceed 2*r1 by up to the ear
            % baseline (triangle inequality on the reflector->ear leg),
            % so pad generously rather than sizing on head-to-reflector
            % distance alone.
            maxDelay = (2 * max(distHead(inRange)) + robot.EarBaseline) / c;
            bufLen = ceil(maxDelay * Fs) + Nfft + 1;
            sigL = zeros(bufLen, 1);
            sigR = zeros(bufLen, 1);

            atten_dB_per_m = atmosphericAbsorption(freqBins, obj.TemperatureC, obj.HumidityPct, obj.PressureKPa);

            ears = {'left', 'right'};
            for e = 1:2
                ear = ears{e};
                earPos = robot.earPosition(ear);
                Rear = robot.earOrientation(ear);

                for jj = 1:nRef
                    j = refIdx(jj);

                    % Circles/walls don't have one fixed reflection point -
                    % it depends on who's looking, so the emission leg (from
                    % the head) and reception leg (from this ear) each get
                    % their own nearest-point query. A wall returns NaN when
                    % the perpendicular foot falls outside its segment (no
                    % valid echo from this position); skip the reflector for
                    % this ear entirely in that case.
                    refPosEm = env.nearestPoint(j, headPos);
                    refPosRx = env.nearestPoint(j, earPos);
                    if any(isnan(refPosEm)) || any(isnan(refPosRx))
                        continue
                    end

                    vEmLocal = Geometry.worldToLocal(Rhead, headPos, refPosEm);
                    [azEm, elEm] = Geometry.dirToAzEl(vEmLocal);
                    r1 = norm(refPosEm - headPos);
                    Hem = hrtf.emitterGain(azEm, elEm, freqBins);

                    vRxLocal = Geometry.worldToLocal(Rear, earPos, refPosRx);
                    [azRx, elRx] = Geometry.dirToAzEl(vRxLocal);
                    r2 = norm(refPosRx - earPos);
                    Hrx = hrtf.receiverGain(ear, azRx, elRx, freqBins);

                    if r1 < eps || r2 < eps
                        continue % degenerate: reflector coincides with head/ear
                    end

                    spreading = 1 / (r1 * r2);
                    atten = 10 .^ (-atten_dB_per_m * (r1 + r2) / 20);

                    Htotal = Hem .* Hrx .* spreading .* env.reflectivityAt(j, freqBins) .* atten;

                    Hfull = zeros(Nfft, 1);
                    Hfull(1:halfN) = Htotal;
                    Hfull(halfN+1:end) = SonarRenderer.mirror(Htotal, Nfft);

                    echoTime = real(ifft(Scall .* Hfull));

                    delaySamples = round((r1 + r2) / c * Fs);
                    idx = delaySamples + (1:Nfft);

                    switch ear
                        case 'left'
                            sigL(idx) = sigL(idx) + echoTime;
                        case 'right'
                            sigR(idx) = sigR(idx) + echoTime;
                    end
                end
            end

            if obj.NoiseAmplitude > 0
                sigL = sigL + obj.NoiseAmplitude * randn(size(sigL));
                sigR = sigR + obj.NoiseAmplitude * randn(size(sigR));
            end
        end

        function [S, tAxis, fAxis] = stft(obj, x, Fs)
            W = obj.WindowLength;
            win = SonarRenderer.hannWindow(W);
            hop = max(1, round(W * (1 - obj.OverlapFraction)));
            N = numel(x);
            nFrames = max(1, floor((N - W) / hop) + 1);
            Nfft = 2^nextpow2(W);
            halfN = floor(Nfft/2) + 1;
            S = zeros(halfN, nFrames);
            for k = 1:nFrames
                idx = (k-1)*hop + (1:W);
                seg = x(idx) .* win;
                Xk = fft(seg, Nfft);
                S(:, k) = abs(Xk(1:halfN));
            end
            fAxis = (0:halfN-1)' * Fs / Nfft;
            tAxis = ((0:nFrames-1) * hop + W/2)' / Fs;
        end

        function [SL, SR, tAxis, fAxis] = renderObservation(obj, robot, env, hrtf)
            [sigL, sigR, Fs] = obj.renderEcho(robot, env, hrtf);
            [SL, tAxis, fAxis] = obj.stft(sigL, Fs);
            SR = obj.stft(sigR, Fs);
        end
    end

    methods (Static, Access = private)
        function mirrored = mirror(halfSpec, Nfft)
            if mod(Nfft, 2) == 0
                mirrored = conj(halfSpec(end-1:-1:2));
            else
                mirrored = conj(halfSpec(end:-1:2));
            end
        end

        function w = hannWindow(N)
            n = (0:N-1)';
            if N == 1
                w = 1;
            else
                w = 0.5 - 0.5 * cos(2*pi*n/(N-1));
            end
        end
    end
end
