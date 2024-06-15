/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/*.html"],
  theme: {
    extend: {    
      colors: {
        chatblack: {50: '#212121'},
        sideblack: {50: '#181414'},
        hovergrey: {50: '#302c2c'}
    }
  },

  },
  plugins: [],
}

