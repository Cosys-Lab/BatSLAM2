function alpha_dB_per_m = atmosphericAbsorption(freq_hz, temp_C, relHumidity_pct, pressure_kPa)
% ATMOSPHERICABSORPTION  Air absorption coefficient per ISO 9613-1.
%   alpha_dB_per_m = atmosphericAbsorption(freq_hz, temp_C, relHumidity_pct, pressure_kPa)
%
% freq_hz can be a vector; the other inputs are scalars. Output is the
% same size as freq_hz, in dB/m. This matters a great deal at bat
% echolocation frequencies (tens of dB/m above ~100 kHz), unlike at
% audible frequencies where it is usually negligible over short ranges.

arguments
    freq_hz (:,1) double
    temp_C (1,1) double = 20
    relHumidity_pct (1,1) double = 50
    pressure_kPa (1,1) double = 101.325
end

T = temp_C + 273.15;        % Kelvin
T0 = 293.15;                % reference temperature (20 C)
T01 = 273.16;                % triple-point temperature of water
p0 = 101.325;                % reference pressure, kPa
pa = pressure_kPa;

f = freq_hz;

% Saturation vapor pressure ratio, then absolute humidity h (%).
psat_over_p0 = 10 .^ (-6.8346 * (T01 / T)^1.261 + 4.6151);
h = relHumidity_pct * psat_over_p0 * (p0 / pa);

% Oxygen and nitrogen relaxation frequencies (Hz).
frO = (pa / p0) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
frN = (pa / p0) * (T / T0)^(-0.5) * (9 + 280 * h * exp(-4.170 * ((T / T0)^(-1/3) - 1)));

term_classical = 1.84e-11 * (p0 / pa) * (T / T0)^0.5;
term_O = 0.01275 * exp(-2239.1 / T) .* (frO ./ (frO^2 + f.^2));
term_N = 0.1068 * exp(-3352.0 / T) .* (frN ./ (frN^2 + f.^2));

alpha_dB_per_m = 8.686 * f.^2 .* (term_classical + (T / T0)^(-2.5) * (term_O + term_N));

end
