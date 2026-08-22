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

const PhysioEngine = (() => {

    // ── Locked feature definitions from notebooks ────────────────
    const FEATURE_NAMES = [
        'mean_HR', 'mean_RR', 'SDNN', 'RMSSD',
        'mean_BR', 'std_BR', 'mean_temp', 'std_temp',
        'mean_acc_mag', 'std_acc_mag'
    ];
    const N_FEATURES = 10;
    const T = 5;                // LSTM-AE sequence length
    const FORECAST_WIN = 10;    // Forecasting input window
    const THRESHOLD_PERCENTILE = 95;

    // ── Physiological state profiles (based on WESAD baseline/stress stats) ──
    const PROFILES = {
        resting: {
            mean_HR:      { mean: 68,   std: 3 },
            mean_RR:      { mean: 882,  std: 30 },
            SDNN:         { mean: 62,   std: 8 },
            RMSSD:        { mean: 55,   std: 7 },
            mean_BR:      { mean: 15,   std: 1.5 },
            std_BR:       { mean: 1.8,  std: 0.5 },
            mean_temp:    { mean: 33.5, std: 0.3 },
            std_temp:     { mean: 0.08, std: 0.02 },
            mean_acc_mag: { mean: 0.98, std: 0.01 },
            std_acc_mag:  { mean: 0.01, std: 0.005 }
        },
        mild: {
            mean_HR:      { mean: 82,   std: 5 },
            mean_RR:      { mean: 731,  std: 40 },
            SDNN:         { mean: 45,   std: 10 },
            RMSSD:        { mean: 35,   std: 8 },
            mean_BR:      { mean: 19,   std: 2 },
            std_BR:       { mean: 3.5,  std: 1 },
            mean_temp:    { mean: 32.8, std: 0.5 },
            std_temp:     { mean: 0.15, std: 0.04 },
            mean_acc_mag: { mean: 1.05, std: 0.03 },
            std_acc_mag:  { mean: 0.04, std: 0.01 }
        },
        high: {
            mean_HR:      { mean: 105,  std: 8 },
            mean_RR:      { mean: 571,  std: 50 },
            SDNN:         { mean: 28,   std: 12 },
            RMSSD:        { mean: 18,   std: 8 },
            mean_BR:      { mean: 24,   std: 3 },
            std_BR:       { mean: 5.5,  std: 1.5 },
            mean_temp:    { mean: 31.5, std: 0.8 },
            std_temp:     { mean: 0.35, std: 0.08 },
            mean_acc_mag: { mean: 1.2,  std: 0.06 },
            std_acc_mag:  { mean: 0.08, std: 0.02 }
        },
        recovery: {
            mean_HR:      { mean: 75,   std: 4 },
            mean_RR:      { mean: 800,  std: 35 },
            SDNN:         { mean: 50,   std: 10 },
            RMSSD:        { mean: 42,   std: 8 },
            mean_BR:      { mean: 17,   std: 2 },
            std_BR:       { mean: 2.5,  std: 0.8 },
            mean_temp:    { mean: 33.0, std: 0.4 },
            std_temp:     { mean: 0.12, std: 0.03 },
            mean_acc_mag: { mean: 1.0,  std: 0.02 },
            std_acc_mag:  { mean: 0.02, std: 0.008 }
        }
    };

    // ── Baseline normalization stats (from notebook 05 WESAD S2 baseline) ──
    const BASELINE_NORM = {
        mean_HR:      { mean: 68,    std: 3 },
        mean_RR:      { mean: 882,   std: 30 },
        SDNN:         { mean: 62,    std: 13 },
        RMSSD:        { mean: 55,    std: 10 },
        mean_BR:      { mean: 15,    std: 1.5 },
        std_BR:       { mean: 1.8,   std: 1.2 },
        mean_temp:    { mean: 33.5,  std: 1.2 },
        std_temp:     { mean: 0.08,  std: 0.02 },
        mean_acc_mag: { mean: 0.98,  std: 0.01 },
        std_acc_mag:  { mean: 0.01,  std: 0.005 }
    };

    // ── State ────────────────────────────────────────────────────
    let currentState = 'resting';
    let noiseLevel = 0.15;
    let windowHistory = [];       // Raw feature vectors
    let anomalyScoreHistory = []; // MSE scores
    let forecastHistory = [];     // P(stress) values
    let baselineScores = [];      // First N scores for threshold
    let anomalyThreshold = 0.3;   // Default, calibrated after baseline
    let windowCount = 0;
    let isCalibrated = false;

    // ── Utility ──────────────────────────────────────────────────
    function gaussRand() {
        let u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function clamp(val, min, max) { return Math.max(min, Math.min(max, val)); }

    function lerp(a, b, t) { return a + (b - a) * t; }

    // ── Generate one feature window ──────────────────────────────
    function generateWindow(state, noise) {
        const profile = PROFILES[state];
        const features = {};
        const noiseScale = 1 + noise * 2;

        for (const fname of FEATURE_NAMES) {
            const p = profile[fname];
            const raw = p.mean + gaussRand() * p.std * noiseScale;
            // Clamp to reasonable physiological ranges
            switch (fname) {
                case 'mean_HR':      features[fname] = clamp(raw, 40, 200); break;
                case 'mean_RR':      features[fname] = clamp(raw, 300, 1500); break;
                case 'SDNN':         features[fname] = clamp(raw, 5, 200); break;
                case 'RMSSD':        features[fname] = clamp(raw, 3, 200); break;
                case 'mean_BR':      features[fname] = clamp(raw, 6, 40); break;
                case 'std_BR':       features[fname] = clamp(raw, 0.1, 15); break;
                case 'mean_temp':    features[fname] = clamp(raw, 25, 40); break;
                case 'std_temp':     features[fname] = clamp(raw, 0, 2); break;
                case 'mean_acc_mag': features[fname] = clamp(raw, 0.5, 3); break;
                case 'std_acc_mag':  features[fname] = clamp(raw, 0, 1); break;
                default:             features[fname] = raw;
            }
        }
        return features;
    }

    // ── Z-score normalization (mirrors safe_normalise from notebook) ──
    function zNormalize(features) {
        const zScores = {};
        for (const fname of FEATURE_NAMES) {
            const norm = BASELINE_NORM[fname];
            const safeStd = Math.max(norm.std, 1e-8);
            zScores[fname] = (features[fname] - norm.mean) / safeStd;
        }
        return zScores;
    }

    // ── Anomaly score: simulated reconstruction error ────────────
    // Mirrors LSTM-AE behavior: low MSE for baseline, high for deviations
    function computeAnomalyScore(zScores) {
        let mse = 0;
        for (const fname of FEATURE_NAMES) {
            const z = zScores[fname];
            // Reconstruction error is proportional to squared z-score
            // Baseline data should reconstruct well (low z), stressed data poorly
            mse += z * z;
        }
        mse /= N_FEATURES;

        // Scale to match notebook ranges (baseline ~0.01-0.03, stress ~0.1-50)
        // Apply non-linear transform to simulate LSTM-AE behavior
        const score = mse * 0.015;
        return score;
    }

    // ── Forecasting: simulated stress onset probability ──────────
    // Mirrors notebook 07 ForecastingLSTM behavior
    function computeForecastProbability(recentScores) {
        if (recentScores.length < 3) return 0;

        const window = recentScores.slice(-FORECAST_WIN);
        
        // Normalize scores to [0,1] (min-max like notebook)
        const mn = Math.min(...window);
        const mx = Math.max(...window);
        const range = mx - mn;
        const normed = range > 1e-8 
            ? window.map(s => (s - mn) / range) 
            : window.map(() => 0);

        // Simulate LSTM forecaster:
        // - Rising trend in anomaly scores -> higher probability
        // - Recent high scores -> higher probability
        const recentMean = normed.slice(-3).reduce((a, b) => a + b, 0) / Math.min(3, normed.length);
        const trend = normed.length >= 4
            ? (normed.slice(-2).reduce((a, b) => a + b, 0) / 2) - 
              (normed.slice(-4, -2).reduce((a, b) => a + b, 0) / 2)
            : 0;

        // Sigmoid-like combination
        const logit = -2.0 + recentMean * 6.0 + trend * 3.0;
        const prob = 1.0 / (1.0 + Math.exp(-logit));
        
        return clamp(prob, 0, 1);
    }

    // ── Compute composite risk score (0-100) ─────────────────────
    function computeRiskScore(anomalyScore, forecastProb) {
        // Anomaly component: how far above threshold
        const threshRatio = anomalyScore / Math.max(anomalyThreshold, 1e-8);
        const anomalyComponent = clamp(threshRatio * 35, 0, 60);

        // Forecast component
        const forecastComponent = forecastProb * 40;

        return clamp(Math.round(anomalyComponent + forecastComponent), 0, 100);
    }

    // ── Risk category ────────────────────────────────────────────
    function getRiskCategory(score) {
        if (score <= 20) return { level: 'low', label: 'Normal', color: '#10b981' };
        if (score <= 45) return { level: 'moderate', label: 'Elevated', color: '#f59e0b' };
        if (score <= 70) return { level: 'high', label: 'High Risk', color: '#f97316' };
        return { level: 'critical', label: 'Critical', color: '#ef4444' };
    }

    // ── Process one tick ─────────────────────────────────────────
    function tick() {
        windowCount++;

        // Generate raw features
        const rawFeatures = generateWindow(currentState, noiseLevel);
        windowHistory.push(rawFeatures);
        if (windowHistory.length > 120) windowHistory.shift();

        // Z-normalize
        const zScores = zNormalize(rawFeatures);

        // Anomaly score
        const anomalyScore = computeAnomalyScore(zScores);
        anomalyScoreHistory.push(anomalyScore);
        if (anomalyScoreHistory.length > 120) anomalyScoreHistory.shift();

        // Calibrate threshold from first 10 baseline windows
        if (!isCalibrated && windowCount <= 10 && currentState === 'resting') {
            baselineScores.push(anomalyScore);
            if (baselineScores.length >= 8) {
                const sorted = [...baselineScores].sort((a, b) => a - b);
                const idx = Math.floor(sorted.length * THRESHOLD_PERCENTILE / 100);
                anomalyThreshold = sorted[Math.min(idx, sorted.length - 1)] * 1.5;
                anomalyThreshold = Math.max(anomalyThreshold, 0.05);
                isCalibrated = true;
            }
        }

        // Forecast probability
        const forecastProb = computeForecastProbability(anomalyScoreHistory);
        forecastHistory.push(forecastProb);
        if (forecastHistory.length > 120) forecastHistory.shift();

        // Risk score
        const riskScore = computeRiskScore(anomalyScore, forecastProb);
        const category = getRiskCategory(riskScore);

        // Z-score statuses
        const featureStatuses = {};
        for (const fname of FEATURE_NAMES) {
            const absZ = Math.abs(zScores[fname]);
            featureStatuses[fname] = absZ < 1.5 ? 'normal' : absZ < 3 ? 'elevated' : 'high';
        }

        return {
            windowIndex: windowCount,
            rawFeatures,
            zScores,
            featureStatuses,
            anomalyScore,
            anomalyThreshold,
            forecastProb,
            riskScore,
            category,
            isCalibrated,
            anomalyScoreHistory: [...anomalyScoreHistory],
            forecastHistory: [...forecastHistory],
        };
    }

    // ── Public API ───────────────────────────────────────────────
    return {
        FEATURE_NAMES,
        N_FEATURES,
        tick,
        setState(state) { if (PROFILES[state]) currentState = state; },
        setNoise(n) { noiseLevel = clamp(n, 0, 1); },
        getState() { return currentState; },
        reset() {
            windowHistory = [];
            anomalyScoreHistory = [];
            forecastHistory = [];
            baselineScores = [];
            windowCount = 0;
            isCalibrated = false;
            anomalyThreshold = 0.3;
            currentState = 'resting';
            noiseLevel = 0.15;
        },
        getHistory() {
            return { windows: windowHistory, scores: anomalyScoreHistory, forecasts: forecastHistory };
        }
    };
})();
