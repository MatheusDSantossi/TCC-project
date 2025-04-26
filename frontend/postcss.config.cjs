module.exports = {
  plugins: {
    'postcss-import': {},
    '@tailwindcss/postcss': {
      config: './tailwind.config.js'
    },
    'postcss-nesting': {},
    autoprefixer: {},
    'postcss-custom-properties': {
      preserve: false  // inlines final colors, strips var(...) calls
    }

  },
}
