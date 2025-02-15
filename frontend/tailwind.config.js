/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: "#012921",
        secondary: "#F0FF3D",
        "secondary-dark": "#CEE009FF",
        tertiary: "#013D31",
        quartiary: "#5D76D3",
        "black-100": "#100d25",
        "black-200": "#090325",
        "white-100": "#f3f3f3",
      },
      boxShadow: {
        card: "0px 35px 120px -15px #013D31",
      },
      screens: {
        xs: "450px",
        sm: "640px",
        md: "768px",
        lg: "1024px",
        xl: "1280px",
      },
      fontSize: {
        tiny: '.875rem',
        base: '1rem',
        lg: '1.125rem',
        xl: '1.25rem',
        '2xl': '1.5rem',
        '5xl': '3rem',
        '6xl': '4rem',
        '7xl': '5rem',
      },
      backgroundImage: {
        "logo": "url('/src/assets/logo.svg')",
      }
    },
  },
  plugins: [],
}

