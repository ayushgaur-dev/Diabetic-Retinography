/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        paper: "#FAF6EF",
        ink: "#16130F",
        coal: "#12100D",
        arterial: "#B3352B",
        amber: "#C98A1B",
        moss: "#5A6650",
      },
      fontFamily: {
        display: ["Didot", "\"Bodoni MT\"", "\"Playfair Display\"", "Georgia", "serif"],
        body: ["system-ui", "-apple-system", "\"Segoe UI\"", "Inter", "sans-serif"],
      },
    },
  },
  plugins: [],
};
