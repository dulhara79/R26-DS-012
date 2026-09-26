/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: { sans: ['Inter', 'sans-serif'] },
      colors: {
        canvas: '#F3F6F4', ink: '#10201F', muted: '#63706E', rule: '#CBD7D3',
        teal: '#4E9892', cyan: '#71B9C6', mist: '#A9C8D5', lavender: '#AAA7D6', peach: '#E8B9A9',
      },
    },
  },
  plugins: [],
}
