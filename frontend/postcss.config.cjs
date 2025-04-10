module.exports = {
  plugins: {
    'postcss-import': {},
    '@tailwindcss/postcss': {
      config: './tailwind.config.js'
    },
    'postcss-nesting': {},
    autoprefixer: {}
  },
}
