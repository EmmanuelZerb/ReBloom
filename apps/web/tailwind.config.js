/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        // Luminous warm palette
        cream: '#FFFDFB',
        coral: {
          DEFAULT: '#FF7F66',
          light: '#FF9B85',
          dark: '#E86B52',
          50: '#FFF5F3',
          100: '#FFE8E3',
        },
        sage: {
          DEFAULT: '#86B09B',
          light: '#A8C9B8',
          dark: '#6A9680',
          50: '#F3F8F5',
        },
        peach: {
          DEFAULT: '#FFB794',
          light: '#FFCDB8',
          50: '#FFF8F4',
        },
      },
      fontFamily: {
        sans: ['var(--font-outfit)', 'Outfit', 'system-ui', 'sans-serif'],
        mono: ['var(--font-mono)', 'JetBrains Mono', 'monospace'],
        display: ['var(--font-outfit)', 'Outfit', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fade-in 0.4s ease-out',
        'slide-up': 'slide-up 0.5s cubic-bezier(0.16, 1, 0.3, 1)',
        'shimmer': 'shimmer 2s linear infinite',
      },
      boxShadow: {
        'soft': '0 2px 8px rgba(0, 0, 0, 0.04)',
        'medium': '0 4px 16px rgba(0, 0, 0, 0.06)',
        'lifted': '0 8px 24px rgba(0, 0, 0, 0.08)',
      },
      borderRadius: {
        '4xl': '2rem',
      },
    },
  },
  plugins: [],
};
