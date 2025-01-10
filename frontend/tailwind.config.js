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
        tertiary: "#013D31",
        quartiary: "#5D76D3",
        "black-100": "#100d25",
        "black-200": "#090325",
        "white-100": "#f3f3f3",
      },
      boxShadow: {
        card: "0px 35px 120px -15px #013D31",
      },
      screes: {
        xs: "450px",
        sm: "640px",
        md: "768px",
        lg: "1024px",
        xl: "1280px",
      },
      backgroundImage: {
        "logo": "url('/src/assets/logo.svg')",
      }
    },
  },
  plugins: [],
}

