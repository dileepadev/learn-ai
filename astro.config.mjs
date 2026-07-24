// @ts-check
import { defineConfig } from 'astro/config';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import tailwindcss from '@tailwindcss/vite';

const site = 'https://dileepadev.github.io';
const base = '/learn-ai';
const docsRoot = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  'src',
  'content',
  'docs',
);

const INTERNAL_LINK_PATTERN = /^(?:[a-z]+:|\/|#|\?)/i;

function rewriteInternalDocLinks() {
  return (tree, file) => {
    const filePath = file.path;

    if (!filePath?.startsWith(docsRoot)) {
      return;
    }

    const relativeFilePath = path
      .relative(docsRoot, filePath)
      .split(path.sep)
      .join(path.posix.sep);
    const currentDir = path.posix.dirname(relativeFilePath);
    const normalizedBase = base.replace(/\/$/, '');

    const visit = (node) => {
      if (node.type === 'link' && typeof node.url === 'string') {
        const url = node.url.trim();

        if (!INTERNAL_LINK_PATTERN.test(url)) {
          const match = url.match(/^([^?#]+)(\?[^#]*)?(#.*)?$/);

          if (match) {
            const [, pathname, search = '', hash = ''] = match;
            const resolvedPath = path.posix
              .join('/', currentDir, pathname)
              .replace(/\.(md|mdx)$/, '')
              .replace(/\/index$/, '/');

            node.url = `${normalizedBase}${resolvedPath}${search}${hash}`;
          }
        }
      }

      if (Array.isArray(node.children)) {
        node.children.forEach(visit);
      }
    };

    visit(tree);
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