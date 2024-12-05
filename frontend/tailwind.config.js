/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      backgroundImage: {
        'custom-radial': 'radial-gradient(circle at center, #e6b7fe 10%, #5049c2 20%, rgba(87,78,255,0) 60%)',
        'custom-bg': 'radial-gradient(circle at right bottom, #e6b7fe 10%, #5049c2 20%, rgba(87,78,255,0) 50%)',
        'custom-bgt': 'radial-gradient(circle at left top, #e6b7fe 10%, #5049c2 20%, rgba(87,78,255,0) 50%)',
      },
    },
  },
  plugins: [],
}