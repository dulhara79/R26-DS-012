/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter Tight"', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
      colors: {
        canvas: '#F7F8FA',
        surface: '#FFFFFF',
        ink: '#101828',
        muted: '#5E6673',
        rule: '#D9DEE7',
        navy: '#132A4A',
        deepNavy: '#0E1A2B',
        indigo: '#4658D9',
        physiology: '#16A6A0',
        clinical: '#DB6B68',
        evidence: '#4A9CC7',
      },
      boxShadow: {
        subtle: '0 10px 30px rgba(16, 24, 40, 0.08)',
      },
    },
  },
  plugins: [],
}
