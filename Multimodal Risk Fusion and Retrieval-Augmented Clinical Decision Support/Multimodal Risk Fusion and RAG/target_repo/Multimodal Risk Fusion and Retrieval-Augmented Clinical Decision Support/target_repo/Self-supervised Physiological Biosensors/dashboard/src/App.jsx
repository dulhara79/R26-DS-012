import { useState, useEffect, useRef, useCallback } from 'react';
import { createEngine, FEATURE_NAMES } from './engine';
import { drawSpark, drawTimeline, drawRadar, drawForecast } from './charts';
import './App.css';

const GAUGE_ARC = 345.58;
const SPARK_MAX = 20;

const SPARK_COLORS = { hr: '#f43f5e', rr: '#6366f1', hrv: '#06b6d4', br: '#10b981', temp: '#f59e0b' };

const KPI_DEFS = [
  { key: 'hr', label: 'Heart Rate', unit: 'bpm', feat: 'mean_HR', icon: 'hr', dec: 0 },
  { key: 'rr', label: 'RR Interval', unit: 'ms', feat: 'mean_RR', icon: 'rr', dec: 0 },
  { key: 'hrv', label: 'HRV (SDNN)', unit: 'ms', feat: 'SDNN', icon: 'hrv', dec: 1 },
  { key: 'br', label: 'Breathing Rate', unit: 'rpm', feat: 'mean_BR', icon: 'br', dec: 1 },
  { key: 'temp', label: 'Skin Temp', unit: '°C', feat: 'mean_temp', icon: 'temp', dec: 1 },
];

const ICONS = {
  hr: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20.84 4.61a5.5 5.5 0 00-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 00-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 000-7.78z"/></svg>,
  rr: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>,
  hrv: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>,
  br: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22c-4-3-8-7-8-12a8 8 0 0116 0c0 5-4 9-8 12z"/><circle cx="12" cy="10" r="3"/></svg>,
  temp: <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 14.76V3.5a2.5 2.5 0 00-5 0v11.26a4.5 4.5 0 105 0z"/></svg>,
};



const STATES = ['resting', 'mild', 'high', 'recovery'];
const STATE_LABELS = { resting: 'Resting', mild: 'Mild Stress', high: 'High Stress', recovery: 'Recovery' };

export default function App() {
  const engineRef = useRef(null);
  const intervalRef = useRef(null);
  const sparkDataRef = useRef({ hr: [], rr: [], hrv: [], br: [], temp: [] });
  const sparkCanvasRefs = useRef({});
  const timelineRef = useRef(null);
  const radarRef = useRef(null);
  const forecastRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [state, setState] = useState('resting');
  const [noise, setNoise] = useState(15);
  const [speed, setSpeed] = useState(2000);
  const [collapsed, setCollapsed] = useState(false);
  const [data, setData] = useState(null);

  // Init engine once
  useEffect(() => { engineRef.current = createEngine(); }, []);

  const doTick = useCallback(() => {
    const eng = engineRef.current;
    if (!eng) return;
    const result = eng.tick();
    setData(result);

    // Update sparklines
    const sd = sparkDataRef.current;
    sd.hr.push(result.raw.mean_HR); sd.rr.push(result.raw.mean_RR);
    sd.hrv.push(result.raw.SDNN); sd.br.push(result.raw.mean_BR);
    sd.temp.push(result.raw.mean_temp);
    for (const k of Object.keys(sd)) { if (sd[k].length > SPARK_MAX) sd[k].shift(); }

    // Draw canvases
    for (const kpi of KPI_DEFS) {
      const c = sparkCanvasRefs.current[kpi.key];
      if (c) drawSpark(c, sd[kpi.key], SPARK_COLORS[kpi.key]);
    }
    drawTimeline(timelineRef.current, result.anomalyScores, result.anomalyThreshold);
    drawRadar(radarRef.current, result.zScores, FEATURE_NAMES);
    drawForecast(forecastRef.current, result.forecasts);
  }, []);

  const start = useCallback(() => {
    if (intervalRef.current) return;
    setRunning(true);
    doTick();
    intervalRef.current = setInterval(doTick, speed);
  }, [speed, doTick]);

  const stop = useCallback(() => {
    clearInterval(intervalRef.current); intervalRef.current = null;
    setRunning(false);
  }, []);

  const reset = useCallback(() => {
    stop();
    engineRef.current?.reset();
    sparkDataRef.current = { hr: [], rr: [], hrv: [], br: [], temp: [] };
    setData(null);
    setState('resting');
    setNoise(15);
    // Clear canvases
    for (const c of Object.values(sparkCanvasRefs.current)) {
      if (c) c.getContext('2d').clearRect(0, 0, c.width, c.height);
    }
    if (timelineRef.current) drawTimeline(timelineRef.current, [], 0.3);
    if (radarRef.current) drawRadar(radarRef.current, null, FEATURE_NAMES);
    if (forecastRef.current) drawForecast(forecastRef.current, []);
  }, [stop]);

  // Restart interval when speed changes while running
  useEffect(() => {
    if (running) {
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(doTick, speed);
    }
    return () => clearInterval(intervalRef.current);
  }, [speed, running, doTick]);

  // Sync state/noise to engine
  useEffect(() => { engineRef.current?.setState(state); }, [state]);
  useEffect(() => { engineRef.current?.setNoise(noise / 100); }, [noise]);

  // Initial canvas draw
  useEffect(() => {
    if (timelineRef.current) drawTimeline(timelineRef.current, [], 0.3);
    if (radarRef.current) drawRadar(radarRef.current, null, FEATURE_NAMES);
    if (forecastRef.current) drawForecast(forecastRef.current, []);
  }, []);

  const riskScore = data?.riskScore ?? null;
  const category = data?.category ?? { level: 'low', label: 'Awaiting Data', color: '#10b981' };
  const gaugeOffset = riskScore != null ? GAUGE_ARC * (1 - riskScore / 100) : GAUGE_ARC;

  let ewClass = '', ewIconClass = 'ew-icon safe', ewTitleText = 'Early Warning System', ewDetailText = 'System validated on 39 subjects | Combined AUROC: 0.97 | EWT: 11.83 min';
  if (riskScore != null) {
    if (riskScore > 70) { ewClass = ' danger'; ewIconClass = 'ew-icon danger'; ewTitleText = 'Acute Stress Detected'; ewDetailText = `P(stress onset) = ${(data.forecastProb * 100).toFixed(1)}% — Anomalous autonomic activation`; }
    else if (riskScore > 40) { ewClass = ' warning'; ewIconClass = 'ew-icon warn'; ewTitleText = 'Elevated Physiological Deviation'; ewDetailText = `P(stress onset) = ${(data.forecastProb * 100).toFixed(1)}% — Monitoring escalation patterns`; }
    else { ewTitleText = 'Physiological Baseline — Normal'; ewDetailText = 'System validated on 39 subjects | Combined AUROC: 0.97 | EWT: 11.83 min'; }
  }

  return (
    <>
      {/* Ambient */}
      <div className="ambient-bg"><div className="gradient-orb orb-1"/><div className="gradient-orb orb-2"/><div className="gradient-orb orb-3"/></div>

      {/* Nav */}
      <nav className="top-nav">
        <div className="nav-left">
          <div className="logo-container">
            <svg width="28" height="28" viewBox="0 0 28 28" fill="none"><path d="M14 2L2 8v12l12 6 12-6V8L14 2z" stroke="url(#lg)" strokeWidth="1.5" fill="none"/><path d="M14 8a6 6 0 100 12 6 6 0 000-12z" stroke="url(#lg)" strokeWidth="1.5" fill="none"/><circle cx="14" cy="14" r="2" fill="url(#lg)"/><defs><linearGradient id="lg" x1="2" y1="2" x2="26" y2="26"><stop stopColor="#6366f1"/><stop offset="1" stopColor="#06b6d4"/></linearGradient></defs></svg>
            <span className="logo-text">Aura</span>
            <span className="logo-badge">Bio Monitor</span>
          </div>
        </div>
        <div className={`connection-status${running ? ' connected' : ''}`}>
          <span className="status-dot"/>
          <span>{running ? 'Simulated Chest Strap' : data ? 'Paused' : 'Chest Strap Disconnected'}</span>
        </div>
        <div className="nav-right"/>
      </nav>

      {/* Dashboard */}
      <main className="dashboard">
        {/* Hero */}
        <section className="hero-section">
          <div className="risk-gauge" style={{ position: 'relative', width: 280, height: 180, margin: '0 auto' }}>
            <svg className="gauge-svg" viewBox="0 0 280 180" style={{ width: '100%', height: '100%' }}>
              <defs>
                <linearGradient id="gg" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" stopColor="#10b981"/><stop offset="35%" stopColor="#f59e0b"/><stop offset="65%" stopColor="#f97316"/><stop offset="100%" stopColor="#ef4444"/></linearGradient>
                <filter id="glow"><feGaussianBlur stdDeviation="3" result="cb"/><feMerge><feMergeNode in="cb"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
              </defs>
              <path d="M 30 160 A 110 110 0 0 1 250 160" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="16" strokeLinecap="round"/>
              <path d="M 30 160 A 110 110 0 0 1 250 160" fill="none" stroke="url(#gg)" strokeWidth="16" strokeLinecap="round" strokeDasharray={GAUGE_ARC} strokeDashoffset={gaugeOffset} filter="url(#glow)" style={{ transition: 'stroke-dashoffset .5s ease' }}/>
            </svg>
            <div className="gauge-value-container">
              <span className="gauge-value" style={{ color: category.color }}>{riskScore ?? '--'}</span>
              <span className="gauge-unit">/ 100</span>
            </div>
          </div>
          <div className="gauge-label">{category.label}</div>
          <div className="risk-meta">
            <div className="meta-item"><span className="meta-label">Anomaly Score</span><span className="meta-value">{data ? data.anomalyScore.toFixed(4) : '--'}</span></div>
            <div className="meta-divider"/>
            <div className="meta-item"><span className="meta-label">Forecast P(stress)</span><span className="meta-value">{data ? `${(data.forecastProb * 100).toFixed(1)}%` : '--'}</span></div>
            <div className="meta-divider"/>
            <div className="meta-item"><span className="meta-label">Threshold</span><span className="meta-value">{data?.isCalibrated ? data.anomalyThreshold.toFixed(4) : '95th %ile'}</span></div>
          </div>
          <div className={`early-warning-panel${ewClass}`}>
            <div className={ewIconClass}><svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
            <div className="ew-content"><span className="ew-title">{ewTitleText}</span><span className="ew-detail">{ewDetailText}</span></div>
          </div>
        </section>

        {/* KPI */}
        <section className="kpi-row">
          {KPI_DEFS.map(kpi => (
            <div className="kpi-card" key={kpi.key}>
              <div className={`kpi-icon ${kpi.icon}`}>{ICONS[kpi.icon]}</div>
              <div><span className="kpi-label">{kpi.label}</span><br/><span className="kpi-value">{data ? data.raw[kpi.feat].toFixed(kpi.dec) : '--'}</span> <span className="kpi-unit">{kpi.unit}</span></div>
              <div className="kpi-spark"><canvas ref={el => { sparkCanvasRefs.current[kpi.key] = el; }} width={240} height={60}/></div>
            </div>
          ))}
        </section>

        {/* Charts */}
        <section className="charts-row">
          <div className="chart-card">
            <div className="chart-header">
              <h3>Anomaly Score Timeline</h3>
              <div className="chart-legend">
                <span className="legend-item"><span className="legend-dot baseline"/> Baseline</span>
                <span className="legend-item"><span className="legend-dot anomaly"/> Anomalous</span>
                <span className="legend-item"><span className="legend-line threshold"/> 95th %ile</span>
              </div>
            </div>
            <div className="chart-body"><canvas ref={timelineRef} width={900} height={220}/></div>
          </div>
          <div className="chart-card">
            <div className="chart-header"><h3>Feature Deviation Radar</h3></div>
            <div className="chart-body"><canvas ref={radarRef} width={320} height={280}/></div>
          </div>
        </section>

        {/* Bottom */}
        <section className="bottom-row">
          <div className="chart-card">
            <div className="chart-header"><h3>Stress Onset Forecast</h3><span className="chart-subtitle">Probability of stress within next 5 min</span></div>
            <div className="chart-body"><canvas ref={forecastRef} width={500} height={180}/></div>
          </div>
          <div className="info-card">
            <div className="chart-header"><h3>Live Feature Vector</h3><span className="chart-subtitle">10 features / 60s window / 30s step</span></div>
            <div className="feature-table-container">
              <table className="feature-table">
                <thead><tr><th>Feature</th><th>Raw Value</th><th>Z-Score</th><th>Status</th></tr></thead>
                <tbody>
                  {FEATURE_NAMES.map(f => (
                    <tr key={f}>
                      <td>{f}</td>
                      <td>{data ? data.raw[f].toFixed(2) : '--'}</td>
                      <td>{data ? data.zScores[f].toFixed(3) : '--'}</td>
                      <td><span className={`status-badge ${data ? data.featureStatuses[f] : 'normal'}`}>{data ? data.featureStatuses[f] : '--'}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>


      </main>

      {/* Sim Controls */}
      <div className={`sim-controls${collapsed ? ' collapsed' : ''}`}>
        <div className="sim-header" onClick={() => setCollapsed(c => !c)}>
          <span className="sim-title">Simulation Controls</span>
          <button className="sim-toggle"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="6 9 12 15 18 9"/></svg></button>
        </div>
        <div className="sim-body">
          <div className="sim-group">
            <label>Physiological State</label>
            <div className="sim-btn-group">
              {STATES.map(s => <button key={s} className={`sim-btn${state === s ? ' active' : ''}`} onClick={() => setState(s)}>{STATE_LABELS[s]}</button>)}
            </div>
          </div>
          <div className="sim-group">
            <label>Noise Level</label>
            <input type="range" className="sim-slider" min="0" max="100" value={noise} onChange={e => setNoise(+e.target.value)}/>
            <span className="sim-slider-val">{noise}%</span>
          </div>
          <div className="sim-group">
            <label>Simulation Speed</label>
            <input type="range" className="sim-slider" min="500" max="5000" step="500" value={speed} onChange={e => setSpeed(+e.target.value)}/>
            <span className="sim-slider-val">{(speed / 1000).toFixed(1)}s</span>
          </div>
          <div className="sim-actions">
            {!running
              ? <button className="sim-action-btn start" onClick={start}>▶ Start Streaming</button>
              : <button className="sim-action-btn stop" onClick={stop}>⏸ Stop</button>}
            <button className="sim-action-btn reset" onClick={reset}>↺ Reset</button>
          </div>
        </div>
      </div>
    </>
  );
}
