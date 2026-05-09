import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  BarChart, Bar, PieChart, Pie, Cell
} from 'recharts';
import { 
  Activity, Users, Battery, Smartphone, Phone, AlertCircle, LayoutDashboard, 
  Settings, UserCircle, Bell, Search, Menu, X
} from 'lucide-react';
import { globalStats, subjectTimelineData, appUsageData, subjectsList } from './data/mockData';

const GlassCard = ({ title, icon: Icon, children, className = '' }) => (
  <div className={`glass-panel ${className}`}>
    <div className="card-header">
      <h3 className="card-title">
        {Icon && <Icon size={20} className="icon-primary" />}
        {title}
      </h3>
    </div>
    <div className="card-body">
      {children}
    </div>
  </div>
);

function App() {
  const [selectedSubject, setSelectedSubject] = useState(subjectsList[0].id);
  const [activeTab, setActiveTab] = useState('overview');
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // Close sidebar on mobile when tab changes
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth > 768) {
        setIsSidebarOpen(false);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setIsSidebarOpen(false); // Auto close on mobile
  };

  const COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div className="app-container">
      
      {/* Mobile Sidebar Overlay */}
      <div 
        className={`sidebar-overlay ${isSidebarOpen ? 'open' : ''}`}
        onClick={() => setIsSidebarOpen(false)}
      ></div>

      {/* Sidebar */}
      <aside className={`sidebar ${isSidebarOpen ? 'open' : ''}`}>
        <div className="sidebar-header flex justify-between items-center">
          <h1 className="logo-text">
            <Activity /> PhenoDash
          </h1>
          <button 
            className="mobile-only icon-btn" 
            onClick={() => setIsSidebarOpen(false)}
          >
            <X size={20} />
          </button>
        </div>
        
        <nav className="sidebar-nav">
          <button 
            onClick={() => handleTabChange('overview')}
            className={`nav-button ${activeTab === 'overview' ? 'active' : ''}`}
          >
            <LayoutDashboard size={20} />
            Global Overview
          </button>
          <button 
            onClick={() => handleTabChange('subjects')}
            className={`nav-button ${activeTab === 'subjects' ? 'active' : ''}`}
          >
            <Users size={20} />
            Subject Details
          </button>
        </nav>

        <div className="sidebar-footer">
          <button className="nav-button">
            <Settings size={20} />
            Settings
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        <header className="topbar">
          <div className="topbar-left">
            <button 
              className="icon-btn mobile-only" 
              onClick={() => setIsSidebarOpen(true)}
            >
              <Menu size={24} />
            </button>
            <div className="search-container desktop-only">
              <Search className="search-icon" size={18} />
              <input 
                type="text" 
                placeholder="Search subjects..." 
                className="search-input"
              />
            </div>
          </div>
          
          <div className="topbar-right">
            <button className="icon-btn notification-btn">
              <Bell size={20} />
              <span className="notification-dot"></span>
            </button>
            <div className="user-profile">
              <UserCircle size={32} className="user-avatar" />
              <div className="user-info desktop-only">
                <p className="user-name">Dr. Researcher</p>
                <p className="user-role">Admin</p>
              </div>
            </div>
          </div>
        </header>

        <div className="scroll-area">
          {activeTab === 'overview' && (
            <div className="view-container">
              <div className="view-header">
                <h2>Study Overview</h2>
                <p className="subtitle">High-level metrics for the Anxiety Digital Phenotyping study.</p>
              </div>

              <div className="stats-grid">
                <GlassCard title="Active Participants" icon={Users}>
                  <div className="stat-value">{globalStats.activeParticipants}</div>
                  <p className="trend-text success">↑ 12% from last month</p>
                </GlassCard>
                
                <GlassCard title="Total Data Points" icon={Activity}>
                  <div className="stat-value">{(globalStats.totalDataPoints / 1000).toFixed(1)}k</div>
                  <p className="trend-text neutral">Collected last 24h</p>
                </GlassCard>

                <GlassCard title="EMA Compliance" icon={AlertCircle}>
                  <div className="stat-value">{globalStats.complianceRateEMA}%</div>
                  <div className="progress-bar-container">
                    <div className="progress-bar success" style={{ width: `${globalStats.complianceRateEMA}%` }}></div>
                  </div>
                </GlassCard>

                <GlassCard title="PSS10 Compliance" icon={AlertCircle}>
                  <div className="stat-value">{globalStats.complianceRatePSS}%</div>
                  <div className="progress-bar-container">
                    <div className="progress-bar primary" style={{ width: `${globalStats.complianceRatePSS}%` }}></div>
                  </div>
                </GlassCard>
              </div>

              <div className="charts-grid-main">
                <GlassCard title="Aggregate Anxiety Trends" className="chart-span-2">
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={subjectTimelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorAnxiety" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#6366f1" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                        <XAxis dataKey="date" stroke="#94a3b8" tick={{fill: '#94a3b8', fontSize: 12}} tickLine={false} axisLine={false} />
                        <YAxis stroke="#94a3b8" tick={{fill: '#94a3b8', fontSize: 12}} tickLine={false} axisLine={false} />
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px' }}
                          itemStyle={{ color: '#e2e8f0' }}
                        />
                        <Area type="monotone" dataKey="anxietyScore" stroke="#818cf8" strokeWidth={3} fillOpacity={1} fill="url(#colorAnxiety)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </GlassCard>

                <GlassCard title="App Usage">
                  <div className="chart-container-pie">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={appUsageData}
                          cx="50%"
                          cy="50%"
                          innerRadius={50}
                          outerRadius={70}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {appUsageData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="legend-container">
                    {appUsageData.map((item, i) => (
                      <div key={item.name} className="legend-item">
                        <span className="legend-color" style={{ backgroundColor: COLORS[i] }}></span>
                        {item.name}
                      </div>
                    ))}
                  </div>
                </GlassCard>
              </div>
            </div>
          )}

          {activeTab === 'subjects' && (
            <div className="view-container">
              <div className="view-header-row">
                <div>
                  <h2>Subject Detail View</h2>
                  <p className="subtitle">Deep dive into individual digital phenotyping data.</p>
                </div>
                
                <div className="select-container">
                  <select 
                    value={selectedSubject}
                    onChange={(e) => setSelectedSubject(e.target.value)}
                    className="subject-select"
                  >
                    {subjectsList.map(s => (
                      <option key={s.id} value={s.id}>{s.id} ({s.status})</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="charts-grid-half">
                
                <GlassCard title="Clinical Metrics (EMA)" icon={Activity}>
                  <p className="chart-subtitle">Daily Anxiety vs Mood correlation</p>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={subjectTimelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" stroke="#64748b" tick={{fontSize: 10}} />
                        <YAxis stroke="#64748b" tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f111a', border: '1px solid #334155' }} />
                        <Line type="monotone" dataKey="anxietyScore" stroke="#ef4444" strokeWidth={2} dot={{r: 3}} name="Anxiety Level" />
                        <Line type="monotone" dataKey="moodScore" stroke="#10b981" strokeWidth={2} dot={{r: 3}} name="Mood Level" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </GlassCard>

                <GlassCard title="Sociability Index" icon={Phone}>
                  <p className="chart-subtitle">Daily interactions (Calls & SMS)</p>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={subjectTimelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis dataKey="date" stroke="#64748b" tick={{fontSize: 10}} />
                        <YAxis stroke="#64748b" tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f111a', border: '1px solid #334155' }} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
                        <Bar dataKey="socialInteractions" fill="#6366f1" radius={[4, 4, 0, 0]} name="Interactions" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </GlassCard>

                <GlassCard title="Physical Activity (Motion)" icon={Activity}>
                  <p className="chart-subtitle">High motion events per day</p>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={subjectTimelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorMotion" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3}/>
                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" stroke="#64748b" tick={{fontSize: 10}} />
                        <YAxis stroke="#64748b" tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f111a', border: '1px solid #334155' }} />
                        <Area type="step" dataKey="highMotionEvents" stroke="#f59e0b" fill="url(#colorMotion)" name="Motion Events" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                </GlassCard>

                <GlassCard title="Digital Engagement" icon={Smartphone}>
                  <p className="chart-subtitle">Total Screen Time (Hours)</p>
                  <div className="chart-container">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={subjectTimelineData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                        <XAxis dataKey="date" stroke="#64748b" tick={{fontSize: 10}} />
                        <YAxis stroke="#64748b" tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{ backgroundColor: '#0f111a', border: '1px solid #334155' }} />
                        <Line type="monotone" dataKey="screenTimeHours" stroke="#0ea5e9" strokeWidth={3} name="Screen Time (h)" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </GlassCard>

              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
