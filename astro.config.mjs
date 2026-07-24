// @ts-check
import { defineConfig } from 'astro/config';

import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://dileepadev.github.io',
  base: '/learn-ai',
  vite: {
    plugins: [tailwindcss()],
  },
});