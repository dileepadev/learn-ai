import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';

import {
  getDocDirectory,
  rewriteDocLink,
} from './rewriteDocLinks.js';

const docsRoot = path.join(path.sep, 'repo', 'src', 'content', 'docs');

test('getDocDirectory returns null for files outside the docs root', () => {
  assert.equal(
    getDocDirectory(path.join(path.sep, 'repo', 'src', 'pages', 'index.astro'), docsRoot),
    null,
  );
});

test('rewriteDocLink prefixes the base path for section index links', () => {
  assert.equal(
    rewriteDocLink('what-is-ai', 'introduction', '/learn-ai'),
    '/learn-ai/introduction/what-is-ai',
  );
});

test('rewriteDocLink resolves markdown extensions and preserves query/hash parts', () => {
  assert.equal(
    rewriteDocLink('./direct-preference-optimization.md?view=full#examples', 'generative-ai', '/learn-ai'),
    '/learn-ai/generative-ai/direct-preference-optimization?view=full#examples',
  );
});

test('rewriteDocLink skips protocol-relative, fragment-only, and query-only URLs', () => {
  assert.equal(rewriteDocLink('//cdn.example.com/doc', 'introduction', '/learn-ai'), '//cdn.example.com/doc');
  assert.equal(rewriteDocLink('#topics', 'introduction', '/learn-ai'), '#topics');
  assert.equal(rewriteDocLink('?view=compact', 'introduction', '/learn-ai'), '?view=compact');
});
