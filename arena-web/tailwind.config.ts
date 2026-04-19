import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        ink: {
          50: '#f8f8f7',
          100: '#eeeeec',
          200: '#d4d4d0',
          300: '#a8a89f',
          400: '#7a7a6f',
          500: '#52524a',
          600: '#3a3a32',
          700: '#262621',
          800: '#16161a',
          900: '#0a0a0c',
        },
        accent: {
          DEFAULT: '#7cc4ff',
          strong: '#3ca2ff',
        },
        good: '#7ad39a',
        warn: '#f5c46b',
        bad: '#f07070',
      },
    },
  },
  plugins: [],
};

export default config;
