// Mock Data for Digital Phenotyping Dashboard

// Generate dates for the last 14 days
const generateDates = (days) => {
  const dates = [];
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date();
    d.setDate(d.getDate() - i);
    dates.push(d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
  }
  return dates;
};

const labels14Days = generateDates(14);

export const globalStats = {
  activeParticipants: 42,
  totalDataPoints: 124500,
  complianceRateEMA: 85,
  complianceRatePSS: 92,
};

export const subjectTimelineData = labels14Days.map((date, index) => ({
  date,
  anxietyScore: Math.floor(Math.random() * 4) + Math.random() * 2, // 0-5 scale
  moodScore: Math.floor(Math.random() * 3) + 2, // 2-5 scale
  highMotionEvents: Math.floor(Math.random() * 50) + 10,
  screenTimeHours: (Math.random() * 4 + 2).toFixed(1),
  socialInteractions: Math.floor(Math.random() * 15) + 2, // Calls + SMS
  batteryDrainPerDay: Math.floor(Math.random() * 100) + 50, // %
}));

export const appUsageData = [
  { name: 'Social Media', value: 35 },
  { name: 'Messaging', value: 25 },
  { name: 'Browser', value: 20 },
  { name: 'Entertainment', value: 15 },
  { name: 'Other', value: 5 },
];

export const subjectsList = [
  { id: 'SUBJ-001', status: 'Active', compliance: 95, riskLevel: 'Low' },
  { id: 'SUBJ-002', status: 'Active', compliance: 80, riskLevel: 'Medium' },
  { id: 'SUBJ-003', status: 'Inactive', compliance: 40, riskLevel: 'High' },
  { id: 'SUBJ-004', status: 'Active', compliance: 100, riskLevel: 'Low' },
];
