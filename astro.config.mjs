// @ts-check
import { defineConfig } from 'astro/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import tailwindcss from '@tailwindcss/vite';
import { getDocDirectory, rewriteDocLink } from './src/utils/rewriteDocLinks.js';

const site = 'https://dileepadev.github.io';
const base = '/learn-ai';
const docsRoot = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  'src',
  'content',
  'docs',
);

function rewriteInternalDocLinks() {
  return (tree, file) => {
    const currentDir = getDocDirectory(file.path, docsRoot);

    if (!currentDir) {
      return;
    }

    const nodes = [tree];

    while (nodes.length > 0) {
      const node = nodes.pop();

      if (!node) {
        continue;
      }

      if (node.type === 'link' && typeof node.url === 'string') {
        node.url = rewriteDocLink(node.url, currentDir, base);
      }

      if (Array.isArray(node.children)) {
        nodes.push(...node.children);
      }
    }
  };
}

// https://astro.build/config
export default defineConfig({
  site,
  base,
  markdown: {
    remarkPlugins: [rewriteInternalDocLinks],
  },
  vite: {
    plugins: [tailwindcss()],
  },
});