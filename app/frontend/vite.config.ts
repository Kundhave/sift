import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = process.env.SIFT_API ?? "http://127.0.0.1:8000";
const proxy = { "/api": { target: api, changeOrigin: true, timeout: 600000 } };

export default defineConfig({
  plugins: [react()],
  server: { proxy },
  preview: { proxy },
});
