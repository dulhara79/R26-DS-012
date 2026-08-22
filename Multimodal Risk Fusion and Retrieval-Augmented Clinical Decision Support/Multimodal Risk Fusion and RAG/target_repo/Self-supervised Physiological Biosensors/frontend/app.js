/**
 * app.js — Dashboard UI Controller
 * Connects PhysioEngine to DOM elements, renders charts, handles controls.
 */

(function () {
    'use strict';

    // ── DOM References ───────────────────────────────────────────
    const $ = id => document.getElementById(id);
    const gaugeValue = $('gauge-value');
    const gaugeFill = $('gauge-fill');
    const gaugeLabel = $('gauge-label');
    const gaugeEl = $('risk-gauge');
    const anomalyVal = $('anomaly-score-value');
    const forecastVal = $('forecast-prob-value');
    const thresholdVal = $('threshold-value');
    const connStatus = $('connection-status');
    const ewPanel = $('early-warning-panel');
    const ewTitle = $('ew-title');
    const ewDetail = $('ew-detail');

    const kpiHR = $('kpi-hr-val');
    const kpiRR = $('kpi-rr-val');
    const kpiHRV = $('kpi-hrv-val');
    const kpiBR = $('kpi-br-val');
    const kpiTemp = $('kpi-temp-val');

    const canvasTimeline = $('canvas-timeline');
    const canvasRadar = $('canvas-radar');
    const canvasForecast = $('canvas-forecast');
    const featureTbody = $('feature-tbody');

    const monHR = $('mon-hr-val');
    const monBR = $('mon-br-val');
    const monTemp = $('mon-temp-val');
    const monMotion = $('mon-motion-val');
    const vitalsTime = $('vitals-time');

    const btnStart = $('btn-start');
    const btnStop = $('btn-stop');
    const btnReset = $('btn-reset');
    const noiseSlider = $('noise-slider');
    const noiseVal = $('noise-val');
    const speedSlider = $('speed-slider');
    const speedVal = $('speed-val');
    const simToggle = $('sim-toggle');
    const simControls = $('sim-controls');
    const simBody = $('sim-body');
    const stateBtns = document.querySelectorAll('.sim-btn[data-state]');

    // ── State ────────────────────────────────────────────────────
    let intervalId = null;
    let speed = 2000;
    let sparkData = { hr: [], rr: [], hrv: [], br: [], temp: [] };
    const SPARK_MAX = 20;
    const GAUGE_ARC_LEN = 345.58;

    // ── Initialize feature table ─────────────────────────────────
    function initFeatureTable() {
        featureTbody.innerHTML = '';
        PhysioEngine.FEATURE_NAMES.forEach(fname => {
            const tr = document.createElement('tr');
            tr.id = 'frow-' + fname;
            tr.innerHTML = `<td>${fname}</td><td class="fval">--</td><td class="fz">--</td><td><span class="status-badge normal">--</span></td>`;
            featureTbody.appendChild(tr);
        });
    }

    // ── Gauge update ─────────────────────────────────────────────
    function updateGauge(score, category) {
        const pct = score / 100;
        const offset = GAUGE_ARC_LEN * (1 - pct);
        gaugeFill.style.strokeDashoffset = offset;
        gaugeValue.textContent = score;
        gaugeLabel.textContent = category.label;
        gaugeValue.style.color = category.color;
        gaugeEl.className = 'risk-gauge risk-' + category.level;
    }

    // ── Sparkline renderer ───────────────────────────────────────
    function drawSpark(containerId, data, color) {
        const container = $(containerId);
        if (!container || data.length < 2) return;
        const w = container.offsetWidth || 120;
        const h = container.offsetHeight || 30;

        let canvas = container.querySelector('canvas');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.width = w * 2; canvas.height = h * 2;
            canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
            container.appendChild(canvas);
        }

        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const mn = Math.min(...data);
        const mx = Math.max(...data);
        const range = mx - mn || 1;
        const step = (canvas.width) / (data.length - 1);

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineJoin = 'round';

        data.forEach((v, i) => {
            const x = i * step;
            const y = canvas.height - ((v - mn) / range) * (canvas.height - 8) - 4;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();
    }

    // ── Timeline chart ───────────────────────────────────────────
    function drawTimeline(scores, threshold) {
        const canvas = canvasTimeline;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (scores.length < 2) {
            ctx.fillStyle = '#64748b'; ctx.font = '13px Inter';
            ctx.textAlign = 'center'; ctx.fillText('Awaiting data...', W / 2, H / 2);
            return;
        }

        const pad = { l: 50, r: 20, t: 15, b: 25 };
        const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
        const mx = Math.max(...scores, threshold * 1.5) * 1.1;

        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + (ch / 4) * i;
            ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
            ctx.fillStyle = '#475569'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'right';
            ctx.fillText((mx * (1 - i / 4)).toFixed(2), pad.l - 6, y + 3);
        }

        // Threshold line
        const threshY = pad.t + ch * (1 - threshold / mx);
        ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
        ctx.beginPath(); ctx.moveTo(pad.l, threshY); ctx.lineTo(W - pad.r, threshY); ctx.stroke();
        ctx.setLineDash([]);

        // Bars
        const barW = Math.max(2, (cw / scores.length) - 1);
        scores.forEach((s, i) => {
            const x = pad.l + (i / scores.length) * cw;
            const barH = (s / mx) * ch;
            const y = pad.t + ch - barH;
            const isAnomaly = s > threshold;
            ctx.fillStyle = isAnomaly ? 'rgba(239,68,68,0.7)' : 'rgba(16,185,129,0.5)';
            ctx.fillRect(x, y, barW, barH);
        });
    }

    // ── Radar chart ──────────────────────────────────────────────
    function drawRadar(zScores) {
        const canvas = canvasRadar;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const features = PhysioEngine.FEATURE_NAMES;
        const n = features.length;
        const cx = W / 2, cy = H / 2 + 10;
        const maxR = Math.min(cx, cy) - 40;
        const angleStep = (2 * Math.PI) / n;
        const startAngle = -Math.PI / 2;

        // Concentric rings
        [0.25, 0.5, 0.75, 1.0].forEach(frac => {
            ctx.beginPath();
            for (let i = 0; i <= n; i++) {
                const a = startAngle + i * angleStep;
                const x = cx + Math.cos(a) * maxR * frac;
                const y = cy + Math.sin(a) * maxR * frac;
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1; ctx.stroke();
        });

        // Spokes + labels
        features.forEach((fname, i) => {
            const a = startAngle + i * angleStep;
            const ex = cx + Math.cos(a) * maxR;
            const ey = cy + Math.sin(a) * maxR;
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey);
            ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.stroke();

            const lx = cx + Math.cos(a) * (maxR + 18);
            const ly = cy + Math.sin(a) * (maxR + 18);
            ctx.fillStyle = '#64748b'; ctx.font = '9px Inter'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            const short = fname.replace('mean_', '').replace('std_', 's_');
            ctx.fillText(short, lx, ly);
        });

        if (!zScores) return;

        // Data polygon
        ctx.beginPath();
        const maxZ = 5;
        features.forEach((fname, i) => {
            const z = Math.min(Math.abs(zScores[fname] || 0), maxZ);
            const r = (z / maxZ) * maxR;
            const a = startAngle + i * angleStep;
            const x = cx + Math.cos(a) * r;
            const y = cy + Math.sin(a) * r;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.fillStyle = 'rgba(99,102,241,0.15)'; ctx.fill();
        ctx.strokeStyle = '#6366f1'; ctx.lineWidth = 2; ctx.stroke();

        // Dots
        features.forEach((fname, i) => {
            const z = Math.min(Math.abs(zScores[fname] || 0), maxZ);
            const r = (z / maxZ) * maxR;
            const a = startAngle + i * angleStep;
            const x = cx + Math.cos(a) * r;
            const y = cy + Math.sin(a) * r;
            ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = z > 3 ? '#ef4444' : z > 1.5 ? '#f59e0b' : '#10b981'; ctx.fill();
        });
    }

    // ── Forecast chart ───────────────────────────────────────────
    function drawForecast(forecasts) {
        const canvas = canvasForecast;
        const ctx = canvas.getContext('2d');
        const W = canvas.width, H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        if (forecasts.length < 2) {
            ctx.fillStyle = '#64748b'; ctx.font = '13px Inter'; ctx.textAlign = 'center';
            ctx.fillText('Awaiting forecast data...', W / 2, H / 2);
            return;
        }

        const pad = { l: 40, r: 15, t: 15, b: 25 };
        const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;

        // Danger zone
        const dangerY = pad.t + ch * 0.5;
        ctx.fillStyle = 'rgba(239,68,68,0.04)';
        ctx.fillRect(pad.l, pad.t, cw, dangerY - pad.t);

        // 0.5 threshold
        ctx.strokeStyle = 'rgba(239,68,68,0.3)'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(pad.l, dangerY); ctx.lineTo(W - pad.r, dangerY); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#ef4444'; ctx.font = '9px JetBrains Mono'; ctx.textAlign = 'left';
        ctx.fillText('P=0.5', pad.l + 4, dangerY - 4);

        // Y labels
        ctx.fillStyle = '#475569'; ctx.font = '10px JetBrains Mono'; ctx.textAlign = 'right';
        ctx.fillText('1.0', pad.l - 5, pad.t + 4);
        ctx.fillText('0.0', pad.l - 5, pad.t + ch + 4);

        // Line
        const gradient = ctx.createLinearGradient(0, pad.t, 0, pad.t + ch);
        gradient.addColorStop(0, 'rgba(239,68,68,0.8)');
        gradient.addColorStop(0.5, 'rgba(245,158,11,0.8)');
        gradient.addColorStop(1, 'rgba(16,185,129,0.8)');

        ctx.beginPath(); ctx.strokeStyle = gradient; ctx.lineWidth = 2; ctx.lineJoin = 'round';
        forecasts.forEach((p, i) => {
            const x = pad.l + (i / (forecasts.length - 1)) * cw;
            const y = pad.t + (1 - p) * ch;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Area fill
        ctx.lineTo(pad.l + cw, pad.t + ch);
        ctx.lineTo(pad.l, pad.t + ch);
        ctx.closePath();
        ctx.fillStyle = 'rgba(99,102,241,0.06)'; ctx.fill();
    }

    // ── Update feature table ─────────────────────────────────────
    function updateFeatureTable(raw, zScores, statuses) {
        PhysioEngine.FEATURE_NAMES.forEach(fname => {
            const row = $('frow-' + fname);
            if (!row) return;
            const cells = row.querySelectorAll('td');
            cells[1].textContent = raw[fname].toFixed(2);
            cells[2].textContent = zScores[fname].toFixed(3);
            const badge = cells[3].querySelector('.status-badge');
            badge.textContent = statuses[fname];
            badge.className = 'status-badge ' + statuses[fname];
        });
    }

    // ── Update early warning panel ───────────────────────────────
    function updateEarlyWarning(riskScore, forecastProb, category) {
        ewPanel.className = 'early-warning-panel';
        if (riskScore > 70) {
            ewPanel.classList.add('danger');
            ewTitle.textContent = 'Acute Stress Detected';
            ewDetail.textContent = `P(stress onset) = ${(forecastProb * 100).toFixed(1)}% — Anomalous autonomic activation`;
        } else if (riskScore > 40) {
            ewPanel.classList.add('warning');
            ewTitle.textContent = 'Elevated Physiological Deviation';
            ewDetail.textContent = `P(stress onset) = ${(forecastProb * 100).toFixed(1)}% — Monitoring escalation patterns`;
        } else {
            ewTitle.textContent = 'Physiological Baseline — Normal';
            ewDetail.textContent = `System validated on 39 subjects | Combined AUROC: 0.97 | EWT: 11.83 min`;
        }
    }

    // ── Main tick handler ────────────────────────────────────────
    function onTick() {
        const result = PhysioEngine.tick();

        // KPI cards
        kpiHR.textContent = result.rawFeatures.mean_HR.toFixed(0);
        kpiRR.textContent = result.rawFeatures.mean_RR.toFixed(0);
        kpiHRV.textContent = result.rawFeatures.SDNN.toFixed(1);
        kpiBR.textContent = result.rawFeatures.mean_BR.toFixed(1);
        kpiTemp.textContent = result.rawFeatures.mean_temp.toFixed(1);

        // Sparklines
        sparkData.hr.push(result.rawFeatures.mean_HR);
        sparkData.rr.push(result.rawFeatures.mean_RR);
        sparkData.hrv.push(result.rawFeatures.SDNN);
        sparkData.br.push(result.rawFeatures.mean_BR);
        sparkData.temp.push(result.rawFeatures.mean_temp);
        Object.keys(sparkData).forEach(k => { if (sparkData[k].length > SPARK_MAX) sparkData[k].shift(); });

        drawSpark('spark-hr', sparkData.hr, '#f43f5e');
        drawSpark('spark-rr', sparkData.rr, '#6366f1');
        drawSpark('spark-hrv', sparkData.hrv, '#06b6d4');
        drawSpark('spark-br', sparkData.br, '#10b981');
        drawSpark('spark-temp', sparkData.temp, '#f59e0b');

        // Gauge
        updateGauge(result.riskScore, result.category);

        // Meta values
        anomalyVal.textContent = result.anomalyScore.toFixed(4);
        forecastVal.textContent = (result.forecastProb * 100).toFixed(1) + '%';
        thresholdVal.textContent = result.isCalibrated ? result.anomalyThreshold.toFixed(4) : 'calibrating...';

        // Charts
        drawTimeline(result.anomalyScoreHistory, result.anomalyThreshold);
        drawRadar(result.zScores);
        drawForecast(result.forecastHistory);

        // Feature table
        updateFeatureTable(result.rawFeatures, result.zScores, result.featureStatuses);

        // Early warning
        updateEarlyWarning(result.riskScore, result.forecastProb, result.category);

        // Patient Vitals Monitor
        monHR.textContent = result.rawFeatures.mean_HR.toFixed(0);
        monBR.textContent = result.rawFeatures.mean_BR.toFixed(1);
        monTemp.textContent = result.rawFeatures.mean_temp.toFixed(1);
        monMotion.textContent = result.rawFeatures.mean_acc_mag.toFixed(2);
        
        const now = new Date();
        vitalsTime.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    // ── Controls ─────────────────────────────────────────────────
    function startSimulation() {
        if (intervalId) return;
        connStatus.classList.add('connected');
        connStatus.querySelector('.status-text').textContent = 'Simulated Chest Strap';
        btnStart.classList.add('hidden');
        btnStop.classList.remove('hidden');
        intervalId = setInterval(onTick, speed);
        onTick(); // Immediate first tick
    }

    function stopSimulation() {
        clearInterval(intervalId); intervalId = null;
        connStatus.classList.remove('connected');
        connStatus.querySelector('.status-text').textContent = 'Paused';
        btnStart.classList.remove('hidden');
        btnStop.classList.add('hidden');
    }

    function resetSimulation() {
        stopSimulation();
        PhysioEngine.reset();
        sparkData = { hr: [], rr: [], hrv: [], br: [], temp: [] };
        gaugeValue.textContent = '--';
        gaugeFill.style.strokeDashoffset = GAUGE_ARC_LEN;
        gaugeLabel.textContent = 'Awaiting Data';
        gaugeValue.style.color = '#10b981';
        anomalyVal.textContent = '--';
        forecastVal.textContent = '--';
        thresholdVal.textContent = '95th %ile';
        kpiHR.textContent = '--'; kpiRR.textContent = '--';
        kpiHRV.textContent = '--'; kpiBR.textContent = '--'; kpiTemp.textContent = '--';
        monHR.textContent = '--'; monBR.textContent = '--'; monTemp.textContent = '--'; monMotion.textContent = '--';
        vitalsTime.textContent = 'Awaiting Data...';
        document.querySelectorAll('.kpi-spark canvas').forEach(c => { const ctx = c.getContext('2d'); ctx.clearRect(0, 0, c.width, c.height); });
        drawTimeline([], 0.3); drawRadar(null); drawForecast([]);
        initFeatureTable();
        connStatus.classList.remove('connected');
        connStatus.querySelector('.status-text').textContent = 'Chest Strap Disconnected';
        ewPanel.className = 'early-warning-panel';
        ewTitle.textContent = 'Early Warning System';
        ewDetail.textContent = 'Average EWT: 11.83 min before onset';

        stateBtns.forEach(b => b.classList.toggle('active', b.dataset.state === 'resting'));
        noiseSlider.value = 15; noiseVal.textContent = '15%';
        speedSlider.value = 2000; speedVal.textContent = '2.0s';
    }

    // ── Event Listeners ──────────────────────────────────────────
    btnStart.addEventListener('click', startSimulation);
    btnStop.addEventListener('click', stopSimulation);
    btnReset.addEventListener('click', resetSimulation);

    stateBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            stateBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            PhysioEngine.setState(btn.dataset.state);
        });
    });

    noiseSlider.addEventListener('input', () => {
        const v = parseInt(noiseSlider.value);
        noiseVal.textContent = v + '%';
        PhysioEngine.setNoise(v / 100);
    });

    speedSlider.addEventListener('input', () => {
        speed = parseInt(speedSlider.value);
        speedVal.textContent = (speed / 1000).toFixed(1) + 's';
        if (intervalId) { clearInterval(intervalId); intervalId = setInterval(onTick, speed); }
    });

    simToggle.addEventListener('click', () => simControls.classList.toggle('collapsed'));
    $('sim-header').addEventListener('click', (e) => { if (e.target === simToggle || simToggle.contains(e.target)) return; simControls.classList.toggle('collapsed'); });

    // ── Init ─────────────────────────────────────────────────────
    initFeatureTable();
    drawTimeline([], 0.3);
    drawRadar(null);
    drawForecast([]);
})();
