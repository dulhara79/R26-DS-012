/**
 * engine.js — Physiological Simulation & Anomaly Detection Engine
 *
 * Implements the logic from notebooks 05-08:
 *   - 10-feature physiological vector generation (60s window, 30s step)
 *   - LSTM-AE inspired anomaly scoring via reconstruction error proxy
 *   - Forecasting probability from recent anomaly score trends
 *   - Risk score computation combining anomaly + forecast
 *
 * Feature vector (locked from notebooks 01-04):
 *   [mean_HR, mean_RR, SDNN, RMSSD, mean_BR, std_BR,
 *    mean_temp, std_temp, mean_acc_mag, std_acc_mag]
 */

export const FEATURE_NAMES = [
  'mean_HR', 'mean_RR', 'SDNN', 'RMSSD',
  'mean_BR', 'std_BR', 'mean_temp', 'std_temp',
  'mean_acc_mag', 'std_acc_mag',
];
export const N_FEATURES = 10;

const T = 5;
const FORECAST_WIN = 10;
const THRESHOLD_PERCENTILE = 95;

const PROFILES = {
  resting: {
    mean_HR: { mean: 68, std: 3 }, mean_RR: { mean: 882, std: 30 },
    SDNN: { mean: 62, std: 8 }, RMSSD: { mean: 55, std: 7 },
    mean_BR: { mean: 15, std: 1.5 }, std_BR: { mean: 1.8, std: 0.5 },
    mean_temp: { mean: 33.5, std: 0.3 }, std_temp: { mean: 0.08, std: 0.02 },
    mean_acc_mag: { mean: 0.98, std: 0.01 }, std_acc_mag: { mean: 0.01, std: 0.005 },
  },
  mild: {
    mean_HR: { mean: 82, std: 5 }, mean_RR: { mean: 731, std: 40 },
    SDNN: { mean: 45, std: 10 }, RMSSD: { mean: 35, std: 8 },
    mean_BR: { mean: 19, std: 2 }, std_BR: { mean: 3.5, std: 1 },
    mean_temp: { mean: 32.8, std: 0.5 }, std_temp: { mean: 0.15, std: 0.04 },
    mean_acc_mag: { mean: 1.05, std: 0.03 }, std_acc_mag: { mean: 0.04, std: 0.01 },
  },
  high: {
    mean_HR: { mean: 105, std: 8 }, mean_RR: { mean: 571, std: 50 },
    SDNN: { mean: 28, std: 12 }, RMSSD: { mean: 18, std: 8 },
    mean_BR: { mean: 24, std: 3 }, std_BR: { mean: 5.5, std: 1.5 },
    mean_temp: { mean: 31.5, std: 0.8 }, std_temp: { mean: 0.35, std: 0.08 },
    mean_acc_mag: { mean: 1.2, std: 0.06 }, std_acc_mag: { mean: 0.08, std: 0.02 },
  },
  recovery: {
    mean_HR: { mean: 75, std: 4 }, mean_RR: { mean: 800, std: 35 },
    SDNN: { mean: 50, std: 10 }, RMSSD: { mean: 42, std: 8 },
    mean_BR: { mean: 17, std: 2 }, std_BR: { mean: 2.5, std: 0.8 },
    mean_temp: { mean: 33.0, std: 0.4 }, std_temp: { mean: 0.12, std: 0.03 },
    mean_acc_mag: { mean: 1.0, std: 0.02 }, std_acc_mag: { mean: 0.02, std: 0.008 },
  },
};

const BASELINE_NORM = {
  mean_HR: { mean: 68, std: 3 }, mean_RR: { mean: 882, std: 30 },
  SDNN: { mean: 62, std: 13 }, RMSSD: { mean: 55, std: 10 },
  mean_BR: { mean: 15, std: 1.5 }, std_BR: { mean: 1.8, std: 1.2 },
  mean_temp: { mean: 33.5, std: 1.2 }, std_temp: { mean: 0.08, std: 0.02 },
  mean_acc_mag: { mean: 0.98, std: 0.01 }, std_acc_mag: { mean: 0.01, std: 0.005 },
};

const CLAMP_RANGES = {
  mean_HR: [40, 200], mean_RR: [300, 1500], SDNN: [5, 200], RMSSD: [3, 200],
  mean_BR: [6, 40], std_BR: [0.1, 15], mean_temp: [25, 40], std_temp: [0, 2],
  mean_acc_mag: [0.5, 3], std_acc_mag: [0, 1],
};

function gaussRand() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
function clamp(val, mn, mx) { return Math.max(mn, Math.min(mx, val)); }

export function getRiskCategory(score) {
  if (score <= 20) return { level: 'low', label: 'Normal', color: '#10b981' };
  if (score <= 45) return { level: 'moderate', label: 'Elevated', color: '#f59e0b' };
  if (score <= 70) return { level: 'high', label: 'High Risk', color: '#f97316' };
  return { level: 'critical', label: 'Critical', color: '#ef4444' };
}

export function createEngine() {
  let currentState = 'resting';
  let noiseLevel = 0.15;
  let anomalyScoreHistory = [];
  let forecastHistory = [];
  let baselineScores = [];
  let anomalyThreshold = 0.3;
  let windowCount = 0;
  let isCalibrated = false;

  function generateWindow() {
    const profile = PROFILES[currentState];
    const features = {};
    const ns = 1 + noiseLevel * 2;
    for (const f of FEATURE_NAMES) {
      const raw = profile[f].mean + gaussRand() * profile[f].std * ns;
      const [lo, hi] = CLAMP_RANGES[f];
      features[f] = clamp(raw, lo, hi);
    }
    return features;
  }

  function zNormalize(features) {
    const z = {};
    for (const f of FEATURE_NAMES) {
      const n = BASELINE_NORM[f];
      z[f] = (features[f] - n.mean) / Math.max(n.std, 1e-8);
    }
    return z;
  }

  function computeAnomalyScore(zScores) {
    let mse = 0;
    for (const f of FEATURE_NAMES) mse += zScores[f] * zScores[f];
    return (mse / N_FEATURES) * 0.015;
  }

  function computeForecast(recent) {
    if (recent.length < 3) return 0;
    const w = recent.slice(-FORECAST_WIN);
    const mn = Math.min(...w), mx = Math.max(...w), rng = mx - mn;
    const normed = rng > 1e-8 ? w.map(s => (s - mn) / rng) : w.map(() => 0);
    const rm = normed.slice(-3).reduce((a, b) => a + b, 0) / Math.min(3, normed.length);
    const trend = normed.length >= 4
      ? normed.slice(-2).reduce((a, b) => a + b, 0) / 2 -
        normed.slice(-4, -2).reduce((a, b) => a + b, 0) / 2
      : 0;
    return clamp(1 / (1 + Math.exp(-(-2 + rm * 6 + trend * 3))), 0, 1);
  }

  function computeRisk(anomaly, forecast) {
    const ac = clamp((anomaly / Math.max(anomalyThreshold, 1e-8)) * 35, 0, 60);
    return clamp(Math.round(ac + forecast * 40), 0, 100);
  }

  return {
    setState(s) { if (PROFILES[s]) currentState = s; },
    setNoise(n) { noiseLevel = clamp(n, 0, 1); },
    getState() { return currentState; },
    reset() {
      anomalyScoreHistory = []; forecastHistory = []; baselineScores = [];
      windowCount = 0; isCalibrated = false; anomalyThreshold = 0.3;
      currentState = 'resting'; noiseLevel = 0.15;
    },

    tick() {
      windowCount++;
      const raw = generateWindow();
      const zScores = zNormalize(raw);
      const anomalyScore = computeAnomalyScore(zScores);
      anomalyScoreHistory.push(anomalyScore);
      if (anomalyScoreHistory.length > 120) anomalyScoreHistory.shift();

      if (!isCalibrated && windowCount <= 10 && currentState === 'resting') {
        baselineScores.push(anomalyScore);
        if (baselineScores.length >= 8) {
          const sorted = [...baselineScores].sort((a, b) => a - b);
          const idx = Math.floor(sorted.length * THRESHOLD_PERCENTILE / 100);
          anomalyThreshold = Math.max(sorted[Math.min(idx, sorted.length - 1)] * 1.5, 0.05);
          isCalibrated = true;
        }
      }

      const forecastProb = computeForecast(anomalyScoreHistory);
      forecastHistory.push(forecastProb);
      if (forecastHistory.length > 120) forecastHistory.shift();

      const riskScore = computeRisk(anomalyScore, forecastProb);
      const featureStatuses = {};
      for (const f of FEATURE_NAMES) {
        const az = Math.abs(zScores[f]);
        featureStatuses[f] = az < 1.5 ? 'normal' : az < 3 ? 'elevated' : 'high';
      }

      return {
        windowIndex: windowCount, raw, zScores, featureStatuses,
        anomalyScore, anomalyThreshold, forecastProb, riskScore,
        category: getRiskCategory(riskScore), isCalibrated,
        anomalyScores: [...anomalyScoreHistory],
        forecasts: [...forecastHistory],
      };
    },
  };
}
